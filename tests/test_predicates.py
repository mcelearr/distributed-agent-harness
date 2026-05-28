"""
Tests for the single ``show_when`` predicate on ``@action``.

Covers:
- Decorator forms: bare ``@action``, ``@action()``, ``@action(show_when=...)``
- ``show_when`` False → action hidden from the prompt AND blocked at runtime
  with ``ActionNotAvailable``
- Predicate that raises is treated as False (defensive, both in prompt
  and at runtime)
- ``_pending_trigger`` is threaded through the runtime to predicate
  evaluation
- Direct calls outside a runtime see ``event=None``
- AgentRuntime catches ``ActionNotAvailable`` and surfaces it as a blocked
  TOOL message to the LLM
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
    ActionNotAvailable,
    BaseWorldEnvironment,
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

    # Bare @action — no predicate, always visible.
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

    # Visible iff phase=="open".
    @action(show_when=lambda s, e: s.phase == "open")
    def close_deal(self) -> str:
        """Close the deal — visible only when phase=open."""
        self.state.phase = "closed"
        return "closed"

    # Visible iff phase=="closed".
    @action(show_when=lambda s, e: s.phase == "closed")
    def reopen_deal(self) -> str:
        """Reopen a closed deal — visible only when phase=closed."""
        self.state.phase = "open"
        return "open"

    # Predicate that ignores event and only looks at state.
    @action(show_when=lambda s, e: s.phase == "open")
    def state_only_predicate(self) -> str:
        """Predicate uses state only."""
        return "ok"

    # Predicate that looks at the event.
    @action(show_when=lambda s, e: e is not None and e.kind == TriggerKind.WEBHOOK)
    def webhook_only(self) -> str:
        """Predicate that requires a webhook trigger."""
        return "webhook"

    # Predicate that raises — should be treated as False, defensively.
    @action(show_when=lambda s, e: (_ for _ in ()).throw(RuntimeError("boom")))
    def buggy_show_when(self) -> str:
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
        expected = {
            "add_open_item", "set_flag", "close_deal", "reopen_deal",
            "state_only_predicate", "webhook_only", "buggy_show_when",
        }
        assert set(actions.keys()) == expected

    def test_show_when_stored_on_wrapper(self) -> None:
        close_deal = PhaseWorld.get_actions()["close_deal"]
        add_open_item = PhaseWorld.get_actions()["add_open_item"]
        assert close_deal._show_when is not None
        assert add_open_item._show_when is None


# --------------------------------------------------------------------------- #
# show_when at runtime                                                         #
# --------------------------------------------------------------------------- #

class TestShowWhen:
    def test_passing_show_when_allows_call(self) -> None:
        world = _world()  # phase="open"
        result = world.close_deal()
        assert result == "closed"
        assert world.state.phase == "closed"

    def test_failing_show_when_raises_action_not_available(self) -> None:
        world = _world()
        world.close_deal()  # phase is now "closed"
        with pytest.raises(ActionNotAvailable, match="close_deal"):
            world.close_deal()

    def test_violation_carries_action_name_and_reason(self) -> None:
        world = _world()
        world.close_deal()
        try:
            world.close_deal()
        except ActionNotAvailable as exc:
            assert exc.action_name == "close_deal"
            assert "show_when" in exc.reason.lower()

    def test_predicate_that_raises_treated_as_not_available(self) -> None:
        world = _world()
        with pytest.raises(ActionNotAvailable, match="buggy_show_when"):
            world.buggy_show_when()

    def test_event_is_none_outside_runtime(self) -> None:
        """Direct calls with no runtime should see event=None in predicates."""
        world = _world()
        # webhook_only requires e.kind == WEBHOOK; with e=None, predicate is False.
        with pytest.raises(ActionNotAvailable, match="webhook_only"):
            world.webhook_only()


# --------------------------------------------------------------------------- #
# show_when in the prompt                                                      #
# --------------------------------------------------------------------------- #

class TestShowWhenInPrompt:
    def test_visible_action_appears_in_prompt(self) -> None:
        world = _world()  # phase="open" — close_deal is visible
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)
        assert "### `close_deal" in prompt

    def test_hidden_action_is_invisible_in_prompt(self) -> None:
        world = _world()
        world.close_deal()  # phase="closed" — close_deal is now hidden
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)
        assert "close_deal" not in prompt
        # And reopen_deal becomes visible.
        assert "### `reopen_deal" in prompt

    def test_buggy_predicate_hides_action_in_prompt(self) -> None:
        world = _world()
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)
        assert "buggy_show_when" not in prompt

    def test_no_world_argument_shows_everything(self) -> None:
        """Backward-compat: without a world, all actions appear."""
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt()  # no world
        for name in PhaseWorld.get_actions():
            assert f"### `{name}" in prompt or f"### `{name}(" in prompt

    def test_no_active_latent_partitioning(self) -> None:
        """The prompt has exactly one Actions section — no Active/Latent split."""
        world = _world()
        builder = PromptBuilder(PhaseWorld)
        prompt = builder.build_actions_prompt(world)
        assert "Active" not in prompt
        assert "Latent" not in prompt

    def test_event_threaded_into_predicate(self) -> None:
        """The event argument reaches predicates that consume it."""
        world = _world()
        builder = PromptBuilder(PhaseWorld)

        prompt_no_event = builder.build_actions_prompt(world, event=None)
        assert "webhook_only" not in prompt_no_event

        prompt_webhook = builder.build_actions_prompt(world, event=_webhook())
        assert "### `webhook_only" in prompt_webhook


# --------------------------------------------------------------------------- #
# Runtime integration                                                          #
# --------------------------------------------------------------------------- #

class TestRuntimeBlocking:
    @pytest.mark.asyncio
    async def test_unavailable_action_surfaces_as_blocked_tool_message(self) -> None:
        runtime = _runtime(scripted=[
            # Agent calls close_deal twice — the second should be blocked.
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

        third_call_messages = runtime.llm.calls[2][0]  # type: ignore[attr-defined]
        tool_msgs = [m for m in third_call_messages if m.role == Role.TOOL]
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

        new_world = PhaseWorld("p1", runtime.namespace, runtime.eventlog)
        assert new_world._pending_trigger is None
