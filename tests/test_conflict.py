"""
Tests for conflict resolution: structural pre-check, scripted resolver,
runtime retry caps, Continue/Recover/Abandon flows.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.conflict import (
    Abandon,
    Continue,
    Recover,
    ScriptedResolver,
    fields_disjoint,
)
from distributed_agent_harness.eventlog import Event, InMemoryEventLog
from distributed_agent_harness.llm import (
    CompletionChunk,
    LLMProvider,
    Message,
    Role,
    ToolCall,
    ToolSchema,
)
from distributed_agent_harness.runtime import AgentRuntime
from distributed_agent_harness.transport import (
    OutputChannel,
    OutputEvent,
    OutputEventKind,
    TriggerEvent,
    TriggerKind,
)
from distributed_agent_harness.world import BaseWorldEnvironment, action


# --------------------------------------------------------------------------- #
# Structural pre-check                                                         #
# --------------------------------------------------------------------------- #

class TestStructuralPreCheck:
    def test_disjoint_field_sets_return_true(self) -> None:
        intervening = [Event(project_id="p", action_name="touch_b", offset=0)]
        writes = {"touch_b": ("b",)}
        assert fields_disjoint(
            planned_reads=("a",),
            planned_writes=("a",),
            intervening_events=intervening,
            action_writes_by_name=writes,
        )

    def test_overlapping_fields_return_false(self) -> None:
        intervening = [Event(project_id="p", action_name="touch_a", offset=0)]
        writes = {"touch_a": ("a",)}
        assert not fields_disjoint(
            planned_reads=("a",),
            planned_writes=("a",),
            intervening_events=intervening,
            action_writes_by_name=writes,
        )

    def test_unknown_intervening_action_is_conservative(self) -> None:
        intervening = [Event(project_id="p", action_name="mystery", offset=0)]
        assert not fields_disjoint(
            planned_reads=("a",),
            planned_writes=(),
            intervening_events=intervening,
            action_writes_by_name={},  # unknown
        )

    def test_unknown_writes_marker_is_conservative(self) -> None:
        intervening = [Event(project_id="p", action_name="opaque", offset=0)]
        assert not fields_disjoint(
            planned_reads=("a",),
            planned_writes=(),
            intervening_events=intervening,
            action_writes_by_name={"opaque": ("__unknown__",)},
        )


# --------------------------------------------------------------------------- #
# Runtime integration — conflict resolution end-to-end                         #
# --------------------------------------------------------------------------- #

class CounterState(BaseModel):
    counter: int = 0
    other: int = 0


class CounterWorld(BaseWorldEnvironment):
    State = CounterState

    @action(writes=("counter",))
    def bump_counter(self, by: int = 1) -> int:
        """Increment counter."""
        self.state.counter += by
        return self.state.counter

    @action(writes=("other",))
    def bump_other(self) -> int:
        """Increment other field."""
        self.state.other += 1
        return self.state.other


class FakeLLM(LLMProvider):
    def __init__(self, scripted: list[Message]):
        self._scripted = list(scripted)
        self.calls: list[tuple[list[Message], list[ToolSchema] | None]] = []

    async def chat_complete(self, messages, tools=None, **kwargs):
        self.calls.append((list(messages), tools))
        return self._scripted.pop(0)

    async def chat_complete_stream(self, messages, tools=None, **kwargs) -> AsyncIterator[CompletionChunk]:
        if False:  # pragma: no cover
            yield CompletionChunk()


class RecordingChannel(OutputChannel):
    def __init__(self) -> None:
        self.events: list[OutputEvent] = []

    async def emit(self, event: OutputEvent) -> None:
        self.events.append(event)


def _chat(text: str, channel: OutputChannel | None = None) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="p",
        reply_to=channel,
    )


async def _seed_conflict(eventlog: InMemoryEventLog, action_name: str = "bump_other") -> None:
    """Pre-seed the log with one event so the runtime's first append conflicts.

    Default action is ``bump_other`` (writes a different field, so the
    structural pre-check will auto-Continue). Pass ``bump_counter`` to force
    overlap with the planned action and exercise the resolver path.
    """
    await eventlog.append(
        "p",
        Event(project_id="p", action_name=action_name),
        expected_offset=0,
    )


@pytest.mark.asyncio
async def test_continue_decision_retries_and_succeeds() -> None:
    """When the resolver returns Continue, the action re-runs against the
    new state and the second append wins."""
    namespace = InMemoryNamespace()
    eventlog = InMemoryEventLog()
    # Use an overlapping seed (bump_counter) so the structural pre-check
    # does NOT auto-Continue — we want to exercise the resolver path.
    await _seed_conflict(eventlog, action_name="bump_counter")

    llm = FakeLLM([
        Message(role=Role.ASSISTANT, content=None, tool_calls=[
            ToolCall(id="c1", name="bump_counter", arguments={"by": 5}),
        ]),
        Message(role=Role.ASSISTANT, content="done"),
    ])
    resolver = ScriptedResolver([Continue()])
    runtime = AgentRuntime(
        world_class=CounterWorld,
        namespace=namespace,
        eventlog=eventlog,
        llm=llm,
        conflict_resolver=resolver,
    )
    await runtime.handle(_chat("bump 5"))

    # Resolver was called once.
    assert len(resolver.calls) == 1
    # Final log has two events: seeded + bump_counter.
    assert await eventlog.current_offset("p") == 2


@pytest.mark.asyncio
async def test_structural_precheck_auto_continues_without_resolver() -> None:
    """Disjoint writes auto-Continue without invoking the resolver."""
    namespace = InMemoryNamespace()
    eventlog = InMemoryEventLog()
    await _seed_conflict(eventlog)  # bumps `other`

    llm = FakeLLM([
        Message(role=Role.ASSISTANT, content=None, tool_calls=[
            ToolCall(id="c1", name="bump_counter", arguments={"by": 1}),
        ]),
        Message(role=Role.ASSISTANT, content="done"),
    ])
    resolver = ScriptedResolver([])  # would Abandon if consulted
    runtime = AgentRuntime(
        world_class=CounterWorld,
        namespace=namespace,
        eventlog=eventlog,
        llm=llm,
        conflict_resolver=resolver,
    )
    await runtime.handle(_chat("bump 1"))

    # Resolver was NOT called — structural pre-check handled it.
    assert resolver.calls == []
    assert await eventlog.current_offset("p") == 2


@pytest.mark.asyncio
async def test_recover_decision_drops_turn_and_replans() -> None:
    """Recover pops the stale assistant turn, re-prompts, the LLM re-plans."""
    namespace = InMemoryNamespace()
    eventlog = InMemoryEventLog()
    await _seed_conflict(eventlog, action_name="bump_counter")

    llm = FakeLLM([
        # Turn 1: plan bump_counter (will conflict + Recover)
        Message(role=Role.ASSISTANT, content=None, tool_calls=[
            ToolCall(id="c1", name="bump_counter", arguments={"by": 3}),
        ]),
        # Turn 2: re-plan after Recover, succeed
        Message(role=Role.ASSISTANT, content=None, tool_calls=[
            ToolCall(id="c2", name="bump_counter", arguments={"by": 3}),
        ]),
        Message(role=Role.ASSISTANT, content="done"),
    ])
    resolver = ScriptedResolver([Recover()])
    runtime = AgentRuntime(
        world_class=CounterWorld,
        namespace=namespace,
        eventlog=eventlog,
        llm=llm,
        conflict_resolver=resolver,
    )
    await runtime.handle(_chat("bump 3"))

    assert len(resolver.calls) == 1
    # Three LLM calls: initial plan, re-plan after Recover, then final summary
    assert len(llm.calls) == 3
    # The re-plan call's messages contain the synthetic recover note.
    recover_notes = [
        m for m in llm.calls[1][0]
        if m.role == Role.SYSTEM and "Concurrent state changes" in (m.content or "")
    ]
    assert recover_notes


@pytest.mark.asyncio
async def test_abandon_decision_emits_assistant_message() -> None:
    namespace = InMemoryNamespace()
    eventlog = InMemoryEventLog()
    await _seed_conflict(eventlog, action_name="bump_counter")

    llm = FakeLLM([
        Message(role=Role.ASSISTANT, content=None, tool_calls=[
            ToolCall(id="c1", name="bump_counter", arguments={"by": 1}),
        ]),
    ])
    resolver = ScriptedResolver([Abandon(reason="too risky")])
    runtime = AgentRuntime(
        world_class=CounterWorld,
        namespace=namespace,
        eventlog=eventlog,
        llm=llm,
        conflict_resolver=resolver,
    )
    final = await runtime.handle(_chat("bump 1"))

    assert "abandoned" in (final.content or "")
    assert "too risky" in (final.content or "")


@pytest.mark.asyncio
async def test_recover_cap_escalates_to_abandon() -> None:
    """Three Recover cycles in one user turn → Abandon."""
    namespace = InMemoryNamespace()
    eventlog = InMemoryEventLog()
    await _seed_conflict(eventlog, action_name="bump_counter")

    # The LLM keeps planning bump_counter; every attempt conflicts.
    llm = FakeLLM([
        Message(role=Role.ASSISTANT, content=None, tool_calls=[
            ToolCall(id=f"c{i}", name="bump_counter", arguments={"by": 1}),
        ])
        for i in range(10)
    ])

    # Inject a new conflict before each retry by re-seeding.
    class AlwaysConflictResolver:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def resolve(self, ctx, llm=None):
            self.calls.append(ctx)
            # Sneak in another competing event so the next attempt also conflicts.
            await eventlog.append(
                "p",
                Event(project_id="p", action_name="bump_counter"),
                expected_offset=ctx.current_offset,
            )
            return Recover()

    resolver = AlwaysConflictResolver()
    runtime = AgentRuntime(
        world_class=CounterWorld,
        namespace=namespace,
        eventlog=eventlog,
        llm=llm,
        conflict_resolver=resolver,  # type: ignore[arg-type]
    )
    final = await runtime.handle(_chat("bump"))

    assert "abandoned" in (final.content or "")
    assert "retry cap" in (final.content or "")
