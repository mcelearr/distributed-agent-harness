"""Tests for the event_search module — filtering, ordering, rendering."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from distributed_agent_harness.event_search import (
    EventQuery,
    render_events_markdown,
    search_events,
)
from distributed_agent_harness.eventlog import Event, InMemoryEventLog


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _ts(minutes: int) -> datetime:
    """Deterministic timestamp helper."""
    return datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


async def _seed(log: InMemoryEventLog) -> None:
    """Append a mixed sequence of agent / human / subagent events."""
    events = [
        Event(project_id="p", action_name="add_item", kwargs={"text": "buy milk"},
              actor="agent", timestamp=_ts(0)),
        Event(project_id="p", action_name="consult_legal_research",
              kwargs={"message": "what is Art. 33?"}, actor="agent",
              result_summary="status=completed session=ctx-a content=Within 72h…",
              timestamp=_ts(1)),
        Event(project_id="p", action_name="register_data_subject",
              kwargs={"name": "Bob"}, actor="agent", timestamp=_ts(2)),
        Event(project_id="p", action_name="consult_doc_drafting",
              kwargs={"message": "draft a privacy policy"}, actor="agent",
              result_summary="status=completed session=ctx-b content=Privacy policy…",
              timestamp=_ts(3)),
        Event(project_id="p", action_name="consult_legal_research",
              kwargs={"message": "follow-up on Acme matter", "session_id": "ctx-a"},
              actor="agent",
              result_summary="status=input-required session=ctx-a content=Need date…",
              timestamp=_ts(4)),
    ]
    for ev in events:
        offset = await log.current_offset("p")
        await log.append("p", ev, expected_offset=offset)


# --------------------------------------------------------------------------- #
# Filtering                                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_glob_filters_to_subagent_events() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    results = await search_events(log, "p", EventQuery(action_name_glob="consult_*"))
    names = [e.action_name for e in results]
    assert names == [
        "consult_legal_research",       # offset 4 (newest)
        "consult_doc_drafting",         # offset 3
        "consult_legal_research",       # offset 1
    ]


@pytest.mark.asyncio
async def test_glob_filters_to_specific_subagent() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    results = await search_events(
        log, "p", EventQuery(action_name_glob="consult_legal_research"),
    )
    assert len(results) == 2
    assert all(e.action_name == "consult_legal_research" for e in results)


@pytest.mark.asyncio
async def test_grep_is_case_insensitive_substring() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    results = await search_events(log, "p", EventQuery(grep="acme"))
    assert len(results) == 1
    assert "Acme" in results[0].kwargs["message"]


@pytest.mark.asyncio
async def test_time_window_inclusive_lower_exclusive_upper() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    results = await search_events(
        log, "p",
        EventQuery(since=_ts(1), until=_ts(3)),
    )
    # offsets at minute 1 and 2 — minute 3 excluded by `until`
    assert {e.offset for e in results} == {1, 2}


@pytest.mark.asyncio
async def test_actor_filter() -> None:
    log = InMemoryEventLog()
    await log.append("p", Event(project_id="p", action_name="x", actor="human"), 0)
    await log.append("p", Event(project_id="p", action_name="y", actor="agent"), 1)
    results = await search_events(log, "p", EventQuery(actor="human"))
    assert [e.action_name for e in results] == ["x"]


@pytest.mark.asyncio
async def test_results_are_sorted_newest_first() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    results = await search_events(log, "p", EventQuery())
    offsets = [e.offset for e in results]
    assert offsets == sorted(offsets, reverse=True)


@pytest.mark.asyncio
async def test_limit_caps_results() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    results = await search_events(log, "p", EventQuery(limit=2))
    assert len(results) == 2


@pytest.mark.asyncio
async def test_offset_from_paginates() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    # Skip everything at offset < 3
    results = await search_events(log, "p", EventQuery(offset_from=3))
    assert {e.offset for e in results} == {3, 4}


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_render_includes_offset_and_action() -> None:
    log = InMemoryEventLog()
    await _seed(log)
    results = await search_events(log, "p", EventQuery(action_name_glob="consult_*"))
    md = render_events_markdown(results)
    assert "offset 4" in md
    assert "consult_legal_research" in md
    assert "consult_doc_drafting" in md


def test_render_handles_empty() -> None:
    assert render_events_markdown([]) == "_(no matching events)_"
