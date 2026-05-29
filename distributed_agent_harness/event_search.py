"""
event_search — queryable view over an ``EventLog``.

The same query interface is reused by:

- The runtime's built-in ``search_event_log`` tool (surfaced to the LLM
  alongside ``@actions`` and ``consult_<name>`` subagent tools).
- A CLI helper for humans browsing the project history.
- Other agents inspecting a project over A2A.

Keeping the search logic in one module avoids three slightly-different
implementations and one source of subtle drift.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .eventlog import Event, EventLog


# --------------------------------------------------------------------------- #
# Query                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class EventQuery:
    """A filter over a project's event log.

    All fields are optional. Multiple fields compose with logical AND.

    Attributes
    ----------
    action_name_glob:
        ``fnmatch``-style glob over ``Event.action_name`` (case-sensitive
        because action names are code identifiers). ``"consult_*"`` matches
        every subagent consult; ``"register_data_subject"`` matches that one
        action exactly.
    grep:
        Case-insensitive substring matched against a rendered line:
        ``"<action_name>(<args>) <result_summary>"``. Use for free-text
        searches like "Acme" or "ICO".
    actor:
        Exact match on ``Event.actor``. Common values: ``"agent"``,
        ``"human"``, ``"subagent:<name>"``.
    since, until:
        Inclusive lower / exclusive upper bound on ``Event.timestamp``.
    limit:
        Maximum number of results. Default 20.
    offset_from:
        When set, only events at ``Event.offset >= offset_from`` are
        considered. Use for pagination: re-query with
        ``offset_from = last_seen_offset + 1`` to fetch the next page.
    """
    action_name_glob: str | None = None
    grep: str | None = None
    actor: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 20
    offset_from: int | None = None


# --------------------------------------------------------------------------- #
# Search                                                                       #
# --------------------------------------------------------------------------- #

async def search_events(
    eventlog: "EventLog",
    project_id: str,
    query: EventQuery,
) -> list["Event"]:
    """Return events matching ``query``, sorted by offset descending.

    Reads the whole project log into memory then filters. Sufficient for
    every backend we ship today (in-memory log; Kafka with sane retention
    for v1). When/if a backend gets retention-sensitive, this is where the
    pagination contract gets pushed down into the log impl.
    """
    from_offset = query.offset_from or 0
    events = await eventlog.read_events(project_id, from_offset=from_offset)

    filtered = [e for e in events if _matches(e, query)]
    # Most-recent first so the LLM sees the freshest context at the top.
    filtered.sort(key=lambda e: (e.offset if e.offset is not None else -1), reverse=True)
    return filtered[: max(0, query.limit)]


def _matches(event: "Event", query: EventQuery) -> bool:
    if query.action_name_glob is not None and not fnmatch.fnmatchcase(
        event.action_name, query.action_name_glob,
    ):
        return False
    if query.actor is not None and event.actor != query.actor:
        return False
    if query.since is not None and event.timestamp < query.since:
        return False
    if query.until is not None and event.timestamp >= query.until:
        return False
    if query.grep is not None:
        needle = query.grep.lower()
        haystack = _render_line(event).lower()
        if needle not in haystack:
            return False
    return True


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def render_events_markdown(events: list["Event"]) -> str:
    """Render search results in the same one-line format as ``event_log.md``.

    Includes the offset so the LLM can paginate or quote a specific event
    when reporting back to the user.
    """
    if not events:
        return "_(no matching events)_"
    return "\n".join(f"- {_render_line(e)}" for e in events)


def _render_line(event: "Event") -> str:
    """One human-readable line per event, identical shape to ``event_log.md``."""
    args_str = _format_args(event)
    ts_short = (
        event.timestamp.isoformat()
        .replace("T", " ")
        .split(".")[0]
        .replace("+00:00", " UTC")
    )
    parts = [
        f"offset {event.offset}",
        ts_short,
        f"by {event.actor}",
        f"`{event.action_name}({args_str})`",
    ]
    if event.result_summary:
        parts.append(f"→ {event.result_summary}")
    return " — ".join(parts)


_MAX_ARG_REPR_LEN = 60


def _format_args(event: "Event") -> str:
    def _truncate(value: object) -> str:
        rep = repr(value)
        if len(rep) > _MAX_ARG_REPR_LEN:
            return rep[: _MAX_ARG_REPR_LEN - 1] + "…"
        return rep

    parts = [_truncate(a) for a in event.args]
    parts.extend(f"{k}={_truncate(v)}" for k, v in event.kwargs.items())
    return ", ".join(parts)
