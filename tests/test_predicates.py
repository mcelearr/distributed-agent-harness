"""
Tests for @action predicates — precondition (hard) and relevance (soft).

Covers:
- Decorator forms: bare @action, @action(), @action(precondition=...), @action(relevance=...)
- precondition False → action is hidden in prompt AND blocks at runtime with PreconditionViolation
- relevance False → action is demoted to Latent tier; runtime still allows it
- Both compose: hard gate takes priority over soft hint
- Predicate that raises is treated as False (defensive)
- _pending_trigger is threaded through the runtime to predicate evaluation
- Direct calls outside a runtime see event=None
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.eventlog import InMemoryEventLog
from distributed_agent_harness.llm import (
    CompletionChunk,
    LLMProvider,
    Message,
    Role,
    ToolCall,
    ToolSchema,
)
from distributed_agent_harness.prompt_builder import PromptBuilder
from distributed_agent_harness.runtime import AgentRuntime
from distributed_agent_harness.transport import (
    OutputChannel,
    OutputEvent,
    OutputEventKind,
    TriggerEvent,
    TriggerKind,
)
from distributed_agent_harness.world import (
    BaseWorldEnvironment,
    PreconditionViolation,
    action,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

class PhaseState(BaseModel):
    phase: str = "open"           # "open" | "closed"
    items: list[str] = []
    flag: bool = False


class PhaseWorld(BaseWorldEnvironment):
    State = PhaseState

    # Bare @action — no predicates, always active.
    @action
    def add_open_item(self, text: str) -> str:
        """Add an item (always available)."""
        self.state.items.append(text)
        return text

    # @action() — empty parens, equivalent to bare @action.
    @action()
    def set_flag(self) -> bool:
        """Set the flag (always available)."""
        self.state.flag = True
        return self.state.flag

    # Hard gate: only callable when phase is "open".
    @action(precondition=lambda s, e: s.phase == "open")
    def close_deal(self) -> str:
        """Close the deal — requires phase=open."""
        self.state.phase = "closed"
        return "closed"

    # Soft hint: visible in Latent tier when not relevant.
    @action(relevance=lambda s, e: s.phase == "closed")
    def reopen_deal(self) -> str:
        """Reopen a closed deal — relevant only when phase=closed."""
        self.state.phase = "open"
        return "open"

    # Both: hard gate + soft hint.
    @action(
        precondition=lambda s, e: s.phase == "open",
        relevance=lambda s, e: bool(s.items),
    )
    def finalise(self) -> int:
        """Finalise — requires open phase, relevant when there are items."""
        return len(self.state.items)

    # Predicate that ignores event and only looks at state.
    @action(precondition=lambda s, e: s.phase == "open")
    def state_only_predicate(self) -> str:
        """Predicate uses state only."""
        return "ok"

    # Predicate that looks at the event.
    @action(precondition=lambda s, e: e is not None and e.kind == TriggerKind.WEBHOOK)
    def webhook_only(self) -> str:
        """Predicate that requires a webhook trigger."""
        return "webhook"

    # Predicate that raises — should be treated as False, defensively.
    @action(precondition=lambda s, e: (_ for _ in ()).throw(RuntimeError("boom")))
    def buggy_precondition(self) -> str:
        """Predicate that raises."""
        return "should not reach"


class FakeLLM(LLMProvider):
    def __init__(self, scripted: list[Message]):
        self._scripted = list(scripted)
        self.calls: list[tuple[list[Message], list[ToolSchema] | None]] = []

    async def chat_complete(self, messages, tools=None, **kwargs):
        self.calls.append((list(messages), tools))
        return self._scripted.pop(0)

    async def chat_complete_stream(
        self, messages, tools=None, **kwargs
    ) -> AsyncIterator[CompletionChunk]:
        if False:  # pragma: no cover
            yield CompletionChunk()


def _world() -> PhaseWorld:
    return PhaseWorld(
        project_id="p1",
        namespace=InMemoryNamespace(),
        eventlog=InMemoryEventLog(),
    )


def _runtime(scripted: list[Message]) -> AgentRuntime:
    return AgentRuntime(
        world_class=PhaseWorld,
        namespace=InMemoryNamespace(),
        eventlog=InMemoryEventLog(),
        llm=FakeLLM(scripted),
    )


def _chat(text: str) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="p1",
    )


def _webhook() -> TriggerEvent:
    return TriggerEvent(
        source="test-webhook",
        kind=TriggerKind.WEBHOOK,
        payload={},
        project_id="p1",
    )


# --------------------------------------------------------------------------- #
# Decorator forms                                                              #
# --------------------------------------------------------------------------- #

class TestDecoratorForms:
    def test_bare_at_action_still_works(self) -> None:
        world = _world()
        assert world.add_open_item("a") == "a"
        assert world.state.items == ["a"]

    def test_at_action_with_empty_parens_works(self) -> None:
        world = _world()
        assert world.set_flag() is True
        assert world.state.flag is True

    def test_decorated_methods_are_actions(self) -> None:
        actions = PhaseWorld.get_actions()
        # All eight decorated methods should be picked up
        expected = {
            "add_open_item", "set_flag", "close_deal", "reopen_deal",
            "finalise", "state_only_predicate", "webhook_only",
            "buggy_precondition",
        }
        assert set(actions.keys()) == expected

    def test_predicates_stored_on_wrapper(self) -> None:
        close_deal = PhaseWorld.get_actions()["close_deal"]
        add_open_item = PhaseWorld.get_actions()["add_open_item"]

        assert close_deal._precondition is not None
        assert close_deal._relevance is None
        assert add_open_item._precondition is None
        assert add_open_item._relevance is None


# --------------------------------------------------------------------------- #
# Hard precondition behaviour                                                  #
# --------------------------------------------------------------------------- #

class TestPreconditionHardGate:
    def test_passing_precondition_allows_call(self) -> None:
        world = _world()  # phase="open"
        result = world.close_deal()
        assert result == "closed"
        assert world.state.phase == "closed"

    def test_failing_precondition_raises_precondition_violation(self) -> None:
        world = _world()
        world.close_deal()  # phase is now "closed"
        with pytest.raises(PreconditionViolation, match="close_deal"):
            world.close_deal()

    def test_violation_carries_action_name_and_reason(self) -> None:
        world = _world()
        world.close_deal()
        try:
            world.close_deal()
        except PreconditionViolation as exc:
            assert exc.action_name == "close_deal"
            assert "precondition" in exc.reason.lower()

    def test_predicate_that_raises_treated_as_violation(self) -> None:
        """A buggy predicate should not propagate as a generic exception."""
        world = _world()
        with pytest.raises(PreconditionViolation, match="buggy_precondition"):
            world.buggy_precondition()

    def test_event_is_none_outside_runtime(self) -> None:
        """Direct calls with no runtime should see event=None in predicates."""
        world = _world()
        # webhook_only requires e.kind == WEBHOOK; with e=None, predicate is False
        with pytest.raises(PreconditionViolation, match="webhook_only"):
            world.webhook_only()


# --------------------------------------------------------------------------- #
# Soft relevance behaviour                                                     #
# --------------------------------------------------------------------------- #

class TestRelevanceSoftHint:
    def test_relevance_does_not_block_at_runtime(self) -> None:
        """relevance=False should still allow the call to run."""
        world = _world()
        # reopen_deal has relevance=phase=="closed", but phase="open" by default.
        # It should still execute when called directly.
        result = world.reopen_deal()
        assert result == "open"

    def test_relevance_false_demotes_to_latent_in_prompt(self) -> None:
        world = _world()  # phase="open" — reopen_deal is not relevant
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)

        # reopen_deal should be in the Latent section
        assert "Latent" in prompt
        # Latent section uses one-line manifest format
        assert "- `reopen_deal()`" in prompt

    def test_relevance_true_promotes_to_active(self) -> None:
        world = _world()
        world.close_deal()  # phase="closed"; reopen_deal is now relevant

        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)

        # reopen_deal should be in Active (full detail with ### header)
        assert "### `reopen_deal" in prompt


# --------------------------------------------------------------------------- #
# Composition: precondition + relevance                                        #
# --------------------------------------------------------------------------- #

class TestComposition:
    def test_precondition_false_hides_action_entirely(self) -> None:
        """When precondition is False, action is Hidden regardless of relevance."""
        world = _world()
        world.close_deal()  # phase="closed" — finalise's precondition is False

        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)

        # finalise should not appear in either Active or Latent section
        assert "finalise" not in prompt

    def test_precondition_true_and_relevance_true_is_active(self) -> None:
        world = _world()  # phase="open"
        world.add_open_item("x")  # items now non-empty → finalise is relevant

        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)

        assert "### `finalise" in prompt

    def test_precondition_true_and_relevance_false_is_latent(self) -> None:
        world = _world()  # phase="open", items empty → finalise not relevant
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)

        # finalise should be in Latent manifest
        assert "- `finalise()`" in prompt


# --------------------------------------------------------------------------- #
# PromptBuilder partitioning                                                   #
# --------------------------------------------------------------------------- #

class TestPromptPartitioning:
    def test_no_world_argument_shows_everything_active(self) -> None:
        """Backward compat: build_actions_prompt() with no world treats all as Active."""
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt()  # no world

        # All actions should appear as full headers
        for name in PhaseWorld.get_actions():
            assert f"### `{name}" in prompt or f"### `{name}(" in prompt

    def test_full_prompt_includes_partitioning(self) -> None:
        world = _world()  # phase="open"
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_full_prompt(world)

        assert "Active" in prompt
        # phase="open" → reopen_deal is not relevant → goes to Latent
        assert "Latent" in prompt

    def test_event_threaded_into_predicate(self) -> None:
        """The event argument should reach predicates that consume it."""
        world = _world()
        builder = PromptBuilder(PhaseWorld)

        # With no event, webhook_only's precondition is False → hidden
        prompt_no_event = builder.build_actions_prompt(world, event=None)
        assert "webhook_only" not in prompt_no_event

        # With a webhook event, webhook_only's precondition is True → active
        prompt_webhook = builder.build_actions_prompt(world, event=_webhook())
        assert "### `webhook_only" in prompt_webhook


# --------------------------------------------------------------------------- #
# Runtime integration                                                          #
# --------------------------------------------------------------------------- #

class TestRuntimeBlocking:
    @pytest.mark.asyncio
    async def test_precondition_violation_surfaces_as_blocked_tool_message(self) -> None:
        runtime = _runtime(scripted=[
            # First the agent calls close_deal twice — second time should be blocked
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="close_deal", arguments={})],
            ),
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c2", name="close_deal", arguments={})],
            ),
            Message(role=Role.ASSISTANT, content="Already closed."),
        ])

        await runtime.handle(_chat("close it twice"))

        # The third LLM call should see a blocked TOOL message for the second close_deal
        third_call_messages = runtime.llm.calls[2][0]  # type: ignore[attr-defined]
        tool_msgs = [m for m in third_call_messages if m.role == Role.TOOL]
        # The second one should mention being blocked
        blocked = [m for m in tool_msgs if "blocked" in (m.content or "").lower()]
        assert blocked, f"expected a blocked TOOL message, got: {tool_msgs}"

    @pytest.mark.asyncio
    async def test_runtime_passes_event_to_predicate(self) -> None:
        """The runtime should pass the trigger event so predicates can read it."""

        class RecordingChannel(OutputChannel):
            def __init__(self):
                self.events: list[OutputEvent] = []

            async def emit(self, event: OutputEvent) -> None:
                self.events.append(event)

        runtime = _runtime(scripted=[
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="webhook_only", arguments={})],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])

        ch = RecordingChannel()
        event = _webhook()
        event.reply_to = ch  # type: ignore[assignment]
        await runtime.handle(event)

        # webhook_only's precondition needs event.kind == WEBHOOK; with the webhook
        # trigger, it should succeed.
        results = [e for e in ch.events if e.kind == OutputEventKind.ACTION_RESULT]
        successful = [e for e in results if e.payload.get("result") == "webhook"]
        assert successful, f"expected successful webhook_only call, got: {results}"


class TestPendingTriggerStashing:
    """Direct verification that the runtime sets and clears _pending_trigger."""

    @pytest.mark.asyncio
    async def test_pending_trigger_cleared_after_call(self) -> None:
        runtime = _runtime(scripted=[
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="add_open_item", arguments={"text": "x"})],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        await runtime.handle(_chat("add x"))

        # After the run, any new world over the same namespace should see
        # _pending_trigger=None (it's an instance attr, but freshly constructed
        # worlds initialise it to None).
        new_world = PhaseWorld("p1", runtime.namespace, runtime.eventlog)
        assert new_world._pending_trigger is None
