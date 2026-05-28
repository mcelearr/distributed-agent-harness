"""
Tests for the event-sourced primitives — Event, EventLog, InMemoryEventLog,
and the optimistic-CAS append contract.
"""
from __future__ import annotations

import threading

import pytest

from distributed_agent_harness.eventlog import (
    Appended,
    Conflict,
    Event,
    InMemoryEventLog,
)


# --------------------------------------------------------------------------- #
# InMemoryEventLog                                                             #
# --------------------------------------------------------------------------- #

class TestInMemoryEventLog:
    @pytest.mark.asyncio
    async def test_initial_offset_is_zero(self) -> None:
        log = InMemoryEventLog()
        assert await log.current_offset("p") == 0

    @pytest.mark.asyncio
    async def test_append_with_matching_offset_succeeds(self) -> None:
        log = InMemoryEventLog()
        event = Event(project_id="p", action_name="do")
        result = await log.append("p", event, expected_offset=0)
        assert isinstance(result, Appended)
        assert result.new_offset == 1
        assert event.offset == 0

    @pytest.mark.asyncio
    async def test_append_with_stale_offset_conflicts(self) -> None:
        log = InMemoryEventLog()
        # Two appends at offset 0; the second one stale.
        await log.append("p", Event(project_id="p", action_name="a"), 0)
        result = await log.append("p", Event(project_id="p", action_name="b"), 0)
        assert isinstance(result, Conflict)
        assert result.current_offset == 1
        assert len(result.intervening_events) == 1
        assert result.intervening_events[0].action_name == "a"

    @pytest.mark.asyncio
    async def test_offsets_are_per_project(self) -> None:
        log = InMemoryEventLog()
        await log.append("p1", Event(project_id="p1", action_name="a"), 0)
        # p2 is still at offset 0 — independent.
        assert await log.current_offset("p2") == 0

    @pytest.mark.asyncio
    async def test_read_events_returns_tail(self) -> None:
        log = InMemoryEventLog()
        for name in ("a", "b", "c"):
            await log.append("p", Event(project_id="p", action_name=name), await log.current_offset("p"))
        tail = await log.read_events("p", from_offset=1)
        assert [e.action_name for e in tail] == ["b", "c"]

    def test_thread_safety_under_contention(self) -> None:
        """Mixed appends from many threads — winners' count matches log size."""
        import asyncio as _asyncio

        log = InMemoryEventLog()
        successes = 0
        success_lock = threading.Lock()

        def run() -> None:
            nonlocal successes
            for _ in range(20):
                offset = _asyncio.run(log.current_offset("p"))
                result = _asyncio.run(
                    log.append("p", Event(project_id="p", action_name="x"), offset)
                )
                if isinstance(result, Appended):
                    with success_lock:
                        successes += 1

        threads = [threading.Thread(target=run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = _asyncio.run(log.current_offset("p"))
        assert final == successes
        # At least *some* appends won under contention.
        assert successes > 0
