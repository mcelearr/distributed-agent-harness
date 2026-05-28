"""
BaseWorldEnvironment — the core of the Distributed Agent Harness.

Subclass this to define a domain-specific world model. The base class
handles persistence, event sourcing, and audit logging transparently.

Every ``@action`` method is wrapped in an optimistic-CAS transaction:

    1. Read current_offset from the EventLog
    2. Re-hydrate state from the snapshot if it is stale
    3. Evaluate the action's precondition / relevance predicates
    4. Execute the method body against the in-memory state
    5. Append an Event to the log with expected_offset
       - on success: flush the snapshot (state.json + _meta.last_offset),
         write audit + human-readable event log lines, return the result
       - on conflict: raise ``ConcurrentUpdate`` for the runtime to handle
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Tuple, Type

from pydantic import BaseModel

from .conflict import ConcurrentUpdate
from .eventlog import Appended, Conflict, Event, EventLog
from .namespace import NamespaceAdapter

if TYPE_CHECKING:
    from .transport import TriggerEvent


Predicate = Callable[[BaseModel, "TriggerEvent | None"], bool]


# --------------------------------------------------------------------------- #
# PreconditionViolation — raised by the @action wrapper when a hard predicate  #
# returns False. The AgentRuntime catches it and surfaces a blocking TOOL      #
# message to the LLM (same shape as a pre_action hook BlockDecision).          #
# --------------------------------------------------------------------------- #

class PreconditionViolation(Exception):
    """Raised by an @action wrapper when its ``precondition`` predicate
    returns False (or raises). Surfaced to the LLM as a blocking TOOL message.
    """
    def __init__(self, action_name: str, reason: str) -> None:
        self.action_name = action_name
        self.reason = reason
        super().__init__(f"{action_name}: {reason}")


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

_MAX_ARG_REPR_LEN = 60
_META_KEY = "_meta"


def _format_args_for_log(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Render args/kwargs inline for the markdown event log."""
    def _truncate(value: Any) -> str:
        rep = repr(value)
        if len(rep) > _MAX_ARG_REPR_LEN:
            return rep[: _MAX_ARG_REPR_LEN - 1] + "…"
        return rep

    parts = [_truncate(a) for a in args]
    parts.extend(f"{k}={_truncate(v)}" for k, v in kwargs.items())
    return ", ".join(parts)


def _run_sync(coro):
    """Run an awaitable to completion from sync code.

    @action methods are synchronous (so subclasses don't have to deal with
    asyncio). When the runtime invokes them it does so via
    ``asyncio.to_thread`` — the worker thread has no running loop, so
    ``asyncio.run`` is safe. Direct callers from sync code also work.
    """
    return asyncio.run(coro)


def _split_meta(raw: str) -> Tuple[str, int]:
    """Parse a stored state.json into (state_json_without_meta, last_offset).

    The on-disk shape is ``{ ...state..., "_meta": {"last_offset": N} }``.
    Earlier writes that predate ``_meta`` are treated as ``last_offset = 0``.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw, 0
    if not isinstance(parsed, dict):
        return raw, 0
    meta = parsed.pop(_META_KEY, None) or {}
    last_offset = int(meta.get("last_offset", 0))
    return json.dumps(parsed), last_offset


def _embed_meta(state_json: str, last_offset: int) -> str:
    """Embed ``_meta.last_offset`` into a serialised state JSON document."""
    parsed = json.loads(state_json)
    if not isinstance(parsed, dict):
        return state_json
    parsed[_META_KEY] = {"last_offset": last_offset}
    return json.dumps(parsed, indent=2)


# --------------------------------------------------------------------------- #
# @action decorator                                                             #
# --------------------------------------------------------------------------- #

def action(
    method: Any = None,
    *,
    precondition: Predicate | None = None,
    relevance: Predicate | None = None,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
) -> Any:
    """Mark a WorldEnvironment method as an auditable, event-sourced action.

    The wrapper handles the full event-sourced transaction cycle:

        read current_offset → catch up snapshot → evaluate precondition →
        execute method → append Event with expected_offset → on success,
        flush snapshot + audit + event_log; on conflict, raise
        ConcurrentUpdate for the runtime to resolve.

    Parameters
    ----------
    precondition: Predicate | None
        Hard contract. If False, action is hidden from the prompt and
        raises ``PreconditionViolation`` at call time.
    relevance: Predicate | None
        Soft hint. If False, action is shown in the Latent prompt tier but
        is still callable.
    reads, writes: tuple[str, ...]
        Top-level state field names the action depends on / mutates. Used
        by the runtime's structural conflict pre-check to skip the LLM
        round-trip when an intervening event cannot have invalidated this
        plan. When omitted, the action is treated as touching every field
        (conservative: every conflict invokes the resolver).
    """
    def decorator(target_method: Any) -> Any:
        try:
            source = inspect.getsource(target_method)
        except OSError:
            source = ""

        @functools.wraps(target_method)
        def wrapper(self: "BaseWorldEnvironment", *args: Any, **kwargs: Any) -> Any:
            # Catch up to the latest snapshot. ``_last_seen_offset`` is the
            # CAS token for our append: events that exist beyond it are
            # exactly the ones we'd need to reason about as a conflict.
            current_offset = _run_sync(self._eventlog.current_offset(self._project_id))
            if current_offset != self._last_seen_offset:
                self._hydrate()

            # Evaluate precondition against the freshly-hydrated state.
            if precondition is not None:
                event = getattr(self, "_pending_trigger", None)
                try:
                    ok = bool(precondition(self.state, event))
                except Exception as exc:  # noqa: BLE001
                    raise PreconditionViolation(
                        target_method.__name__,
                        f"precondition raised {type(exc).__name__}: {exc}",
                    ) from exc
                if not ok:
                    raise PreconditionViolation(
                        target_method.__name__,
                        "precondition not satisfied for the current state",
                    )

            # Execute against in-memory state.
            try:
                result = target_method(self, *args, **kwargs)
            except Exception as exc:
                # Re-read the snapshot so the in-memory state matches disk —
                # we never persist a half-applied action.
                self._hydrate()
                self._write_audit(target_method.__name__, args, kwargs, error=str(exc))
                raise

            # Append the event. CAS via expected_offset.
            event = Event(
                project_id=self._project_id,
                action_name=target_method.__name__,
                args=[_serialisable(a) for a in args],
                kwargs={k: _serialisable(v) for k, v in kwargs.items()},
                actor=getattr(self, "_actor", "agent"),
                result_summary=_summarise_result(result),
            )
            append_result = _run_sync(self._eventlog.append(
                self._project_id, event, expected_offset=self._last_seen_offset,
            ))

            if isinstance(append_result, Conflict):
                # Discard the in-memory mutation — re-hydrate from snapshot.
                self._hydrate()
                raise ConcurrentUpdate(
                    action_name=target_method.__name__,
                    last_seen_offset=self._last_seen_offset,
                    current_offset=append_result.current_offset,
                    intervening_events=append_result.intervening_events,
                )

            assert isinstance(append_result, Appended)
            self._last_seen_offset = append_result.new_offset
            self._flush()
            self._write_audit(target_method.__name__, args, kwargs, result=result)
            return result

        wrapper._is_action = True               # type: ignore[attr-defined]
        wrapper._source = source                # type: ignore[attr-defined]
        wrapper._precondition = precondition    # type: ignore[attr-defined]
        wrapper._relevance = relevance          # type: ignore[attr-defined]
        wrapper._reads = tuple(reads)           # type: ignore[attr-defined]
        wrapper._writes = tuple(writes)         # type: ignore[attr-defined]
        return wrapper

    if method is None:
        return decorator
    return decorator(method)


def _serialisable(value: Any) -> Any:
    """Best-effort coercion of an argument to a JSON-friendly shape."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialisable(v) for k, v in value.items()}
    return repr(value)


def _summarise_result(result: Any) -> str | None:
    if result is None:
        return None
    if hasattr(result, "model_dump_json"):
        return type(result).__name__
    return type(result).__name__


# --------------------------------------------------------------------------- #
# BaseWorldEnvironment                                                          #
# --------------------------------------------------------------------------- #

class BaseWorldEnvironment:
    """Abstract base class for all World Environments.

    How to use
    ----------
    1. Define a Pydantic model for your domain state.
    2. Subclass ``BaseWorldEnvironment``, set ``State``, write ``@action`` methods.
    3. Instantiate with pluggable namespace and event log backends::

        world = MyWorld(
            project_id="my-project",
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
        )

    The base class handles snapshot persistence, event-log appends, and
    optimistic-CAS conflict surfacing.
    """

    State: ClassVar[Type[BaseModel]]

    # Standard documents written into every project namespace.
    _STATE_DOC: ClassVar[str] = "state.json"       # snapshot + _meta.last_offset
    _AUDIT_DOC: ClassVar[str] = "audit.jsonl"      # machine audit log
    _SUMMARY_DOC: ClassVar[str] = "summary.md"     # human-readable card view
    _EVENT_LOG_DOC: ClassVar[str] = "event_log.md" # human-readable activity

    def __init__(
        self,
        project_id: str,
        namespace: NamespaceAdapter,
        eventlog: EventLog,
        actor: str = "agent",
    ) -> None:
        if not hasattr(self.__class__, "State"):
            raise TypeError(
                f"{type(self).__name__} must define a 'State' class attribute "
                f"pointing to a Pydantic BaseModel subclass."
            )
        self._project_id = project_id
        self._namespace = namespace
        self._eventlog = eventlog
        self._actor = actor
        self._pending_trigger: "TriggerEvent | None" = None
        self._last_seen_offset: int = 0
        self.state: BaseModel = self.__class__.State()
        self._hydrate()

    # ----------------------------------------------------------------------- #
    # Path helpers                                                             #
    # ----------------------------------------------------------------------- #

    def _state_path(self) -> str:
        return f"{self._project_id}/{self._STATE_DOC}"

    def _audit_path(self) -> str:
        return f"{self._project_id}/{self._AUDIT_DOC}"

    def _summary_path(self) -> str:
        return f"{self._project_id}/{self._SUMMARY_DOC}"

    def _event_log_path(self) -> str:
        return f"{self._project_id}/{self._EVENT_LOG_DOC}"

    # ----------------------------------------------------------------------- #
    # Snapshot lifecycle                                                       #
    # ----------------------------------------------------------------------- #

    def _hydrate(self) -> None:
        """Load the latest snapshot from the namespace into ``self.state``.

        Reads ``state.json``, splits off ``_meta.last_offset``, and validates
        the remaining JSON into the domain state model.
        """
        raw = self._namespace.read_doc(self._state_path())
        if not raw:
            # No snapshot yet — keep the default-constructed state.
            return
        state_json, last_offset = _split_meta(raw)
        self.state = self.__class__.State.model_validate_json(state_json)
        self._last_seen_offset = last_offset

    def _flush(self) -> None:
        """Persist current state as a snapshot and refresh the summary doc."""
        state_json = self.state.model_dump_json(indent=2)
        snapshot = _embed_meta(state_json, self._last_seen_offset)
        self._namespace.write_doc(self._state_path(), snapshot)
        try:
            summary = self.render_summary()
        except Exception:  # noqa: BLE001 — buggy subclass should not break state writes
            log.exception("render_summary() raised; using default summary")
            summary = self._default_summary()
        self._namespace.write_doc(self._summary_path(), summary)

    # ----------------------------------------------------------------------- #
    # Audit + human-readable event log                                         #
    # ----------------------------------------------------------------------- #

    def _write_audit(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Record one action in both the JSONL audit log and the markdown event log."""
        timestamp = datetime.now(timezone.utc).isoformat()

        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "project_id": self._project_id,
            "method": method_name,
            "args": [repr(a) for a in args],
            "kwargs": {k: repr(v) for k, v in kwargs.items()},
            "offset": self._last_seen_offset,
        }
        if error is not None:
            entry["error"] = error
        else:
            entry["result_type"] = type(result).__name__

        existing = self._namespace.read_doc(self._audit_path()) or ""
        self._namespace.write_doc(
            self._audit_path(),
            existing + json.dumps(entry) + "\n",
        )

        self._append_event_log(timestamp, method_name, args, kwargs, error=error)

    def _append_event_log(
        self,
        timestamp: str,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        error: str | None,
    ) -> None:
        existing = self._namespace.read_doc(self._event_log_path())
        if not existing:
            existing = (
                "# Event Log\n\n"
                "_Append-only record of every action taken on this project. "
                "The most recent entries are lifted into the agent's system "
                "prompt to help it avoid repeating itself._\n\n"
            )

        ts_short = timestamp.replace("T", " ").split(".")[0].replace("+00:00", " UTC")
        arg_str = _format_args_for_log(args, kwargs)
        outcome = f"FAILED: {error}" if error else "OK"
        line = f"- **{ts_short}** — `{method_name}({arg_str})` — {outcome}\n"
        self._namespace.write_doc(self._event_log_path(), existing + line)

    # ----------------------------------------------------------------------- #
    # Summary rendering — override on subclasses for narrative output          #
    # ----------------------------------------------------------------------- #

    def render_summary(self) -> str:
        """Render the project's "card view" as Markdown.

        Subclasses should override to produce a domain-specific narrative.
        The default implementation dumps the state JSON.
        """
        return self._default_summary()

    def _default_summary(self) -> str:
        return (
            f"# {type(self).__name__}\n\n"
            f"_Project: `{self._project_id}`_\n\n"
            "_No `render_summary()` override defined; showing raw state._\n\n"
            f"```json\n{self.state.model_dump_json(indent=2)}\n```\n"
        )

    # ----------------------------------------------------------------------- #
    # Introspection (used by PromptBuilder and runtime)                        #
    # ----------------------------------------------------------------------- #

    @classmethod
    def get_actions(cls) -> dict[str, Any]:
        """Return all ``@action``-decorated callables defined on this class."""
        return {
            name: member
            for name, member in inspect.getmembers(cls, predicate=callable)
            if getattr(member, "_is_action", False)
        }

    @classmethod
    def action_writes_by_name(cls) -> dict[str, tuple[str, ...]]:
        """Map of action name → declared ``writes`` field set.

        Used by the conflict pre-check. Actions that declared neither
        ``reads`` nor ``writes`` are reported with ``("__unknown__",)`` so the
        pre-check stays conservative.
        """
        result: dict[str, tuple[str, ...]] = {}
        for name, method in cls.get_actions().items():
            reads = getattr(method, "_reads", ())
            writes = getattr(method, "_writes", ())
            if not reads and not writes:
                result[name] = ("__unknown__",)
            else:
                result[name] = tuple(writes)
        return result
