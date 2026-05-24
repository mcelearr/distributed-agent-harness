"""
Tests for AgentRuntime — the LLM ↔ WorldEnvironment loop.

We use a scripted FakeLLM that returns pre-baked responses so we can drive
the runtime deterministically through tool-call cycles.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.concurrency_handlers import InProcessLock
from distributed_agent_harness.llm import (
    CompletionChunk,
    LLMProvider,
    Message,
    Role,
    ToolCall,
    ToolSchema,
)
from distributed_agent_harness.runtime import AgentRuntime, _build_tool_schemas
from distributed_agent_harness.transport import (
    OutputChannel,
    OutputEvent,
    OutputEventKind,
    TriggerEvent,
    TriggerKind,
)
from distributed_agent_harness.world import BaseWorldEnvironment, action


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

class TodoState(BaseModel):
    items: list[str] = []


class TodoWorld(BaseWorldEnvironment):
    State = TodoState

    @action
    def add_item(self, text: str) -> str:
        """Add an item to the todo list."""
        self.state.items.append(text)
        return text

    @action
    def count_items(self) -> int:
        """Return the number of items in the list."""
        return len(self.state.items)


class FakeLLM(LLMProvider):
    """LLM that returns pre-scripted responses in order."""

    def __init__(self, scripted: list[Message]):
        self._scripted = list(scripted)
        self.calls: list[tuple[list[Message], list[ToolSchema] | None]] = []

    async def chat_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> Message:
        self.calls.append((list(messages), tools))
        return self._scripted.pop(0)

    async def chat_complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        # Not exercised in these tests, but satisfy the ABC.
        if False:  # pragma: no cover
            yield CompletionChunk()


class RecordingChannel(OutputChannel):
    def __init__(self) -> None:
        self.events: list[OutputEvent] = []

    async def emit(self, event: OutputEvent) -> None:
        self.events.append(event)


def _runtime(world_class, llm: LLMProvider, **kwargs) -> AgentRuntime:
    return AgentRuntime(
        world_class=world_class,
        namespace=InMemoryNamespace(),
        concurrency=InProcessLock(),
        llm=llm,
        **kwargs,
    )


def _chat_event(text: str, channel: OutputChannel | None = None) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="test-project",
        reply_to=channel,
    )


# --------------------------------------------------------------------------- #
# Tool schema generation                                                       #
# --------------------------------------------------------------------------- #

class TestToolSchemas:
    def test_schemas_built_for_all_actions(self) -> None:
        schemas, models = _build_tool_schemas(TodoWorld)
        names = {s.name for s in schemas}
        assert names == {"add_item", "count_items"}

    def test_schema_has_correct_parameters(self) -> None:
        schemas, _ = _build_tool_schemas(TodoWorld)
        add_item = next(s for s in schemas if s.name == "add_item")
        assert add_item.parameters["type"] == "object"
        assert add_item.parameters["properties"]["text"]["type"] == "string"
        assert "text" in add_item.parameters["required"]

    def test_no_args_action_has_empty_properties(self) -> None:
        schemas, _ = _build_tool_schemas(TodoWorld)
        count = next(s for s in schemas if s.name == "count_items")
        assert count.parameters["type"] == "object"
        assert count.parameters.get("properties", {}) == {}

    def test_docstring_used_as_description(self) -> None:
        schemas, _ = _build_tool_schemas(TodoWorld)
        add_item = next(s for s in schemas if s.name == "add_item")
        assert "todo list" in add_item.description.lower()


# --------------------------------------------------------------------------- #
# End-to-end runtime loops                                                     #
# --------------------------------------------------------------------------- #

class TestRuntimeLoop:
    @pytest.mark.asyncio
    async def test_immediate_final_response(self) -> None:
        llm = FakeLLM([Message(role=Role.ASSISTANT, content="hello back")])
        runtime = _runtime(TodoWorld, llm)
        channel = RecordingChannel()
        result = await runtime.handle(_chat_event("hi", channel))

        assert result.content == "hello back"
        kinds = [e.kind for e in channel.events]
        assert OutputEventKind.MESSAGE in kinds
        assert kinds[-1] == OutputEventKind.FINAL

    @pytest.mark.asyncio
    async def test_single_tool_call_then_final(self) -> None:
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "buy milk"})],
            ),
            Message(role=Role.ASSISTANT, content="Added 'buy milk'."),
        ])
        runtime = _runtime(TodoWorld, llm)
        channel = RecordingChannel()
        await runtime.handle(_chat_event("add buy milk", channel))

        # Find the action events
        actions = [e for e in channel.events if e.kind == OutputEventKind.ACTION_CALLED]
        results = [e for e in channel.events if e.kind == OutputEventKind.ACTION_RESULT]
        assert len(actions) == 1
        assert actions[0].payload["name"] == "add_item"
        assert results[0].payload["result"] == "buy milk"

    @pytest.mark.asyncio
    async def test_state_persisted_across_tool_calls(self) -> None:
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "a"})],
            ),
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c2", name="add_item", arguments={"text": "b"})],
            ),
            Message(role=Role.ASSISTANT, content="Done."),
        ])
        ns = InMemoryNamespace()
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=ns,
            concurrency=InProcessLock(),
            llm=llm,
        )
        await runtime.handle(_chat_event("add two items"))

        # Re-read the state by constructing a new world over the same namespace
        world = TodoWorld("test-project", ns, InProcessLock())
        assert world.state.items == ["a", "b"]

    @pytest.mark.asyncio
    async def test_unknown_action_surfaces_error_to_llm(self) -> None:
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="nonexistent", arguments={})],
            ),
            Message(role=Role.ASSISTANT, content="Sorry."),
        ])
        runtime = _runtime(TodoWorld, llm)
        await runtime.handle(_chat_event("do bad thing"))

        # Second LLM call should include the tool message describing the error
        second_call_messages = llm.calls[1][0]
        tool_msgs = [m for m in second_call_messages if m.role == Role.TOOL]
        assert any("Unknown action" in (m.content or "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_invalid_args_surface_validation_error(self) -> None:
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="add_item", arguments={})],
            ),
            Message(role=Role.ASSISTANT, content="OK."),
        ])
        runtime = _runtime(TodoWorld, llm)
        await runtime.handle(_chat_event("add"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        assert any("Invalid arguments" in (m.content or "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_method_exception_surfaces_to_llm(self) -> None:
        """A ValueError raised inside an @action becomes a TOOL message, not a crash."""

        class StrictState(BaseModel):
            x: int = 0

        class StrictWorld(BaseWorldEnvironment):
            State = StrictState

            @action
            def set_x(self, value: int) -> int:
                """Set x to a positive integer."""
                if value <= 0:
                    raise ValueError("must be positive")
                self.state.x = value
                return value

        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="set_x", arguments={"value": -1})],
            ),
            Message(role=Role.ASSISTANT, content="Sorry, will retry."),
        ])
        runtime = _runtime(StrictWorld, llm)
        await runtime.handle(_chat_event("set x to -1"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        assert any("must be positive" in (m.content or "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_max_iterations_terminates(self) -> None:
        """If the LLM never stops calling tools, the runtime exits cleanly."""
        # 20 tool calls in a row, runtime cap is 3
        infinite = [
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id=f"c{i}", name="count_items", arguments={})],
            )
            for i in range(20)
        ]
        llm = FakeLLM(infinite)
        runtime = _runtime(TodoWorld, llm, max_iterations=3)
        channel = RecordingChannel()
        await runtime.handle(_chat_event("loop", channel))

        errors = [e for e in channel.events if e.kind == OutputEventKind.ERROR]
        assert errors
        assert "max_iterations" in errors[0].payload["error"]


# --------------------------------------------------------------------------- #
# System prompt content                                                        #
# --------------------------------------------------------------------------- #

class TestSystemPrompt:
    @pytest.mark.asyncio
    async def test_system_prompt_includes_all_standard_sections(self) -> None:
        llm = FakeLLM([Message(role=Role.ASSISTANT, content="ok")])
        runtime = _runtime(TodoWorld, llm)
        await runtime.handle(_chat_event("hi"))

        system_msg = llm.calls[0][0][0]
        assert system_msg.role == Role.SYSTEM
        # All four standardised sections must appear, plus our action.
        for section in [
            "Project Summary",
            "Available Actions",
            "Recent Activity",
            "Current State",
        ]:
            assert section in system_msg.content, f"missing section: {section}"
        assert "add_item" in system_msg.content

    @pytest.mark.asyncio
    async def test_recent_activity_visible_in_subsequent_call(self) -> None:
        """After one action, the next LLM call must see it in Recent Activity."""
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="add_item", arguments={"text": "buy milk"})],
            ),
            Message(role=Role.ASSISTANT, content="Added."),
        ])
        runtime = _runtime(TodoWorld, llm)
        await runtime.handle(_chat_event("add buy milk"))

        # The second LLM call (after the tool call) should have the event log
        # visible in its system prompt.
        second_call_system = llm.calls[1][0][0]
        assert "add_item" in second_call_system.content
        assert "buy milk" in second_call_system.content
        # And the loop-prevention guidance is on the preamble.
        assert "Never repeat an action" in second_call_system.content
