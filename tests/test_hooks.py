"""
Tests for the lifecycle hook system.

Covers:
- pre_action: blocking and pass-through
- pre_action: per-action filtering vs wildcard (None)
- post_action: fires with result, exceptions swallowed
- action_error: fires when @action raises
- pre_trigger: blocks the entire run before world is even created
- run_complete: fires on every run termination path (success, block, etc.)
- Registration order is preserved
- AgentRuntime decorator passthroughs work
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.eventlog import InMemoryEventLog
from distributed_agent_harness.hooks import (
    ActionContext,
    BlockDecision,
    HookRegistry,
)
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
# Test fixtures                                                                #
# --------------------------------------------------------------------------- #

class ItemState(BaseModel):
    items: list[str] = []
    counter: int = 0


class ItemWorld(BaseWorldEnvironment):
    State = ItemState

    @action
    def add_item(self, text: str) -> str:
        """Add an item."""
        self.state.items.append(text)
        return text

    @action
    def increment(self, by: int = 1) -> int:
        """Increment counter."""
        self.state.counter += by
        return self.state.counter

    @action
    def boom(self) -> None:
        """Always raises."""
        raise RuntimeError("intentional failure")


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


class RecordingChannel(OutputChannel):
    def __init__(self) -> None:
        self.events: list[OutputEvent] = []

    async def emit(self, event: OutputEvent) -> None:
        self.events.append(event)


def _make_runtime(scripted: list[Message], hooks: HookRegistry | None = None) -> AgentRuntime:
    return AgentRuntime(
        world_class=ItemWorld,
        namespace=InMemoryNamespace(),
        eventlog=InMemoryEventLog(),
        llm=FakeLLM(scripted),
        hooks=hooks,
    )


def _chat_event(text: str, channel: OutputChannel | None = None) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="p1",
        reply_to=channel,
    )


# --------------------------------------------------------------------------- #
# HookRegistry — unit tests in isolation                                       #
# --------------------------------------------------------------------------- #

class TestHookRegistryFiring:
    @pytest.mark.asyncio
    async def test_pre_action_passes_when_hook_returns_none(self) -> None:
        reg = HookRegistry()

        @reg.on_pre_action("add_item")
        async def hook(ctx: ActionContext) -> None:
            return None

        ctx = ActionContext(project_id="p", action_name="add_item")
        result = await reg.fire_pre_action(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_action_blocks_when_hook_returns_decision(self) -> None:
        reg = HookRegistry()

        @reg.on_pre_action("add_item")
        async def hook(ctx: ActionContext) -> BlockDecision:
            return BlockDecision(reason="not allowed")

        ctx = ActionContext(project_id="p", action_name="add_item")
        result = await reg.fire_pre_action(ctx)
        assert isinstance(result, BlockDecision)
        assert result.reason == "not allowed"

    @pytest.mark.asyncio
    async def test_pre_action_filters_by_action_name(self) -> None:
        reg = HookRegistry()
        fired_for: list[str] = []

        @reg.on_pre_action("add_item")
        async def only_add(ctx: ActionContext) -> None:
            fired_for.append(ctx.action_name)

        await reg.fire_pre_action(ActionContext(project_id="p", action_name="add_item"))
        await reg.fire_pre_action(ActionContext(project_id="p", action_name="increment"))

        assert fired_for == ["add_item"]

    @pytest.mark.asyncio
    async def test_wildcard_pre_action_fires_for_every_action(self) -> None:
        reg = HookRegistry()
        fired_for: list[str] = []

        @reg.on_pre_action()  # None — wildcard
        async def for_all(ctx: ActionContext) -> None:
            fired_for.append(ctx.action_name)

        await reg.fire_pre_action(ActionContext(project_id="p", action_name="a"))
        await reg.fire_pre_action(ActionContext(project_id="p", action_name="b"))

        assert fired_for == ["a", "b"]

    @pytest.mark.asyncio
    async def test_specific_hook_runs_before_wildcard(self) -> None:
        """Specific hooks fire before wildcard so they can block first."""
        reg = HookRegistry()
        call_order: list[str] = []

        @reg.on_pre_action("add_item")
        async def specific(ctx: ActionContext) -> None:
            call_order.append("specific")

        @reg.on_pre_action()
        async def wildcard(ctx: ActionContext) -> None:
            call_order.append("wildcard")

        await reg.fire_pre_action(ActionContext(project_id="p", action_name="add_item"))
        assert call_order == ["specific", "wildcard"]

    @pytest.mark.asyncio
    async def test_first_block_decision_short_circuits(self) -> None:
        reg = HookRegistry()
        call_order: list[str] = []

        @reg.on_pre_action("a")
        async def first(ctx: ActionContext) -> BlockDecision:
            call_order.append("first")
            return BlockDecision(reason="stop here")

        @reg.on_pre_action("a")
        async def second(ctx: ActionContext) -> None:
            call_order.append("second")  # pragma: no cover

        result = await reg.fire_pre_action(ActionContext(project_id="p", action_name="a"))
        assert isinstance(result, BlockDecision)
        assert call_order == ["first"]

    @pytest.mark.asyncio
    async def test_post_action_receives_result(self) -> None:
        reg = HookRegistry()
        seen: list[Any] = []

        @reg.on_post_action("add_item")
        async def hook(ctx: ActionContext, result: Any) -> None:
            seen.append(result)

        await reg.fire_post_action(
            ActionContext(project_id="p", action_name="add_item"),
            result="hello",
        )
        assert seen == ["hello"]

    @pytest.mark.asyncio
    async def test_post_action_exceptions_are_swallowed(self) -> None:
        reg = HookRegistry()
        called_after: list[bool] = []

        @reg.on_post_action()
        async def buggy(ctx: ActionContext, result: Any) -> None:
            raise RuntimeError("oops")

        @reg.on_post_action()
        async def healthy(ctx: ActionContext, result: Any) -> None:
            called_after.append(True)

        # Should NOT raise — buggy hook is logged and swallowed,
        # and the healthy hook still fires.
        await reg.fire_post_action(
            ActionContext(project_id="p", action_name="a"),
            result=None,
        )
        assert called_after == [True]

    @pytest.mark.asyncio
    async def test_action_error_receives_exception(self) -> None:
        reg = HookRegistry()
        seen: list[BaseException] = []

        @reg.on_action_error()
        async def hook(ctx: ActionContext, exc: BaseException) -> None:
            seen.append(exc)

        err = ValueError("bad")
        await reg.fire_action_error(
            ActionContext(project_id="p", action_name="boom"), exc=err
        )
        assert seen == [err]


# --------------------------------------------------------------------------- #
# AgentRuntime integration                                                     #
# --------------------------------------------------------------------------- #

class TestRuntimePreAction:
    @pytest.mark.asyncio
    async def test_block_prevents_action_execution(self) -> None:
        """A blocked action should not mutate state."""
        hooks = HookRegistry()

        @hooks.on_pre_action("add_item")
        async def block(ctx: ActionContext) -> BlockDecision:
            return BlockDecision(reason="denied")

        runtime = _make_runtime(
            scripted=[
                Message(
                    role=Role.ASSISTANT, content=None,
                    tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "x"})],
                ),
                Message(role=Role.ASSISTANT, content="Sorry."),
            ],
            hooks=hooks,
        )

        await runtime.handle(_chat_event("add x"))

        # The world's state.json should NOT contain 'x'
        raw = runtime.namespace.read_doc("p1/state.json")
        assert raw is None or "x" not in raw

    @pytest.mark.asyncio
    async def test_block_reason_surfaces_to_llm(self) -> None:
        hooks = HookRegistry()

        @hooks.on_pre_action("add_item")
        async def block(ctx: ActionContext) -> BlockDecision:
            return BlockDecision(reason="needs partner approval")

        llm_messages = [
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "x"})],
            ),
            Message(role=Role.ASSISTANT, content="OK, will wait."),
        ]
        runtime = _make_runtime(scripted=llm_messages, hooks=hooks)
        await runtime.handle(_chat_event("add x"))

        # The LLM's second call should have seen the block reason in a TOOL message
        second_call_messages = runtime.llm.calls[1][0]  # type: ignore[attr-defined]
        tool_msgs = [m for m in second_call_messages if m.role == Role.TOOL]
        assert any("needs partner approval" in (m.content or "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_block_emits_blocked_output_event(self) -> None:
        hooks = HookRegistry()

        @hooks.on_pre_action("add_item")
        async def block(ctx: ActionContext) -> BlockDecision:
            return BlockDecision(reason="denied")

        runtime = _make_runtime(
            scripted=[
                Message(
                    role=Role.ASSISTANT, content=None,
                    tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "x"})],
                ),
                Message(role=Role.ASSISTANT, content="ok"),
            ],
            hooks=hooks,
        )
        ch = RecordingChannel()
        await runtime.handle(_chat_event("add x", ch))

        blocked_events = [
            e for e in ch.events
            if e.kind == OutputEventKind.ACTION_RESULT and e.payload.get("blocked")
        ]
        assert len(blocked_events) == 1
        assert "denied" in blocked_events[0].payload["error"]

    @pytest.mark.asyncio
    async def test_context_carries_trigger_and_kwargs(self) -> None:
        hooks = HookRegistry()
        captured: list[ActionContext] = []

        @hooks.on_pre_action("add_item")
        async def capture(ctx: ActionContext) -> None:
            captured.append(ctx)

        runtime = _make_runtime(
            scripted=[
                Message(
                    role=Role.ASSISTANT, content=None,
                    tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "hello"})],
                ),
                Message(role=Role.ASSISTANT, content="done"),
            ],
            hooks=hooks,
        )
        event = _chat_event("add hello")
        await runtime.handle(event)

        assert len(captured) == 1
        ctx = captured[0]
        assert ctx.action_name == "add_item"
        assert ctx.kwargs == {"text": "hello"}
        assert ctx.project_id == "p1"
        assert ctx.trigger is event


class TestRuntimePostAction:
    @pytest.mark.asyncio
    async def test_post_action_fires_with_real_result(self) -> None:
        hooks = HookRegistry()
        seen: list[tuple[str, Any]] = []

        @hooks.on_post_action("increment")
        async def record(ctx: ActionContext, result: Any) -> None:
            seen.append((ctx.action_name, result))

        runtime = _make_runtime(
            scripted=[
                Message(
                    role=Role.ASSISTANT, content=None,
                    tool_calls=[ToolCall(id="c1", name="increment", arguments={"by": 5})],
                ),
                Message(role=Role.ASSISTANT, content="incremented"),
            ],
            hooks=hooks,
        )
        await runtime.handle(_chat_event("bump 5"))

        assert seen == [("increment", 5)]

    @pytest.mark.asyncio
    async def test_post_action_does_not_fire_when_action_raises(self) -> None:
        hooks = HookRegistry()
        post_calls: list[Any] = []
        err_calls: list[BaseException] = []

        @hooks.on_post_action("boom")
        async def post(ctx: ActionContext, result: Any) -> None:
            post_calls.append(result)  # pragma: no cover

        @hooks.on_action_error("boom")
        async def err(ctx: ActionContext, exc: BaseException) -> None:
            err_calls.append(exc)

        runtime = _make_runtime(
            scripted=[
                Message(
                    role=Role.ASSISTANT, content=None,
                    tool_calls=[ToolCall(id="c1", name="boom", arguments={})],
                ),
                Message(role=Role.ASSISTANT, content="oh well"),
            ],
            hooks=hooks,
        )
        await runtime.handle(_chat_event("explode"))

        assert post_calls == []
        assert len(err_calls) == 1
        assert isinstance(err_calls[0], RuntimeError)


class TestRuntimePreTrigger:
    @pytest.mark.asyncio
    async def test_pre_trigger_block_stops_the_run(self) -> None:
        hooks = HookRegistry()

        @hooks.on_pre_trigger
        async def gate(event: TriggerEvent) -> BlockDecision:
            return BlockDecision(reason="ratelimit")

        runtime = _make_runtime(
            scripted=[Message(role=Role.ASSISTANT, content="never called")],
            hooks=hooks,
        )
        result = await runtime.handle(_chat_event("hi"))

        # LLM should never have been called
        assert runtime.llm.calls == []  # type: ignore[attr-defined]
        assert "ratelimit" in (result.content or "")

    @pytest.mark.asyncio
    async def test_pre_trigger_emits_error_event(self) -> None:
        hooks = HookRegistry()

        @hooks.on_pre_trigger
        async def gate(event: TriggerEvent) -> BlockDecision:
            return BlockDecision(reason="bad source")

        runtime = _make_runtime(
            scripted=[Message(role=Role.ASSISTANT, content="x")],
            hooks=hooks,
        )
        ch = RecordingChannel()
        await runtime.handle(_chat_event("hi", ch))

        errors = [e for e in ch.events if e.kind == OutputEventKind.ERROR]
        assert errors
        assert "bad source" in errors[0].payload["error"]


class TestRuntimeRunComplete:
    @pytest.mark.asyncio
    async def test_run_complete_fires_once_on_success(self) -> None:
        hooks = HookRegistry()
        calls: list[tuple[TriggerEvent, Message]] = []

        @hooks.on_run_complete
        async def track(event: TriggerEvent, final: Message) -> None:
            calls.append((event, final))

        runtime = _make_runtime(
            scripted=[Message(role=Role.ASSISTANT, content="done")],
            hooks=hooks,
        )
        event = _chat_event("hi")
        await runtime.handle(event)

        assert len(calls) == 1
        assert calls[0][0] is event
        assert calls[0][1].content == "done"

    @pytest.mark.asyncio
    async def test_run_complete_fires_even_when_pre_trigger_blocks(self) -> None:
        hooks = HookRegistry()
        completed: list[bool] = []

        @hooks.on_pre_trigger
        async def block(event: TriggerEvent) -> BlockDecision:
            return BlockDecision(reason="no")

        @hooks.on_run_complete
        async def fire(event: TriggerEvent, final: Message) -> None:
            completed.append(True)

        runtime = _make_runtime(
            scripted=[Message(role=Role.ASSISTANT, content="x")],
            hooks=hooks,
        )
        await runtime.handle(_chat_event("hi"))

        assert completed == [True]


class TestRuntimeDecoratorPassthroughs:
    @pytest.mark.asyncio
    async def test_runtime_on_pre_action_delegates_to_registry(self) -> None:
        runtime = _make_runtime(
            scripted=[
                Message(
                    role=Role.ASSISTANT, content=None,
                    tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "x"})],
                ),
                Message(role=Role.ASSISTANT, content="ok"),
            ],
        )

        @runtime.on_pre_action("add_item")
        async def block(ctx: ActionContext) -> BlockDecision:
            return BlockDecision(reason="blocked via runtime")

        await runtime.handle(_chat_event("add x"))

        # Verify the block went through
        second_call_messages = runtime.llm.calls[1][0]  # type: ignore[attr-defined]
        tool_msgs = [m for m in second_call_messages if m.role == Role.TOOL]
        assert any("blocked via runtime" in (m.content or "") for m in tool_msgs)
