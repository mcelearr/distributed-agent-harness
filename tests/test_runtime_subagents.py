"""
Tests for runtime integration of subagents and the search_event_log meta-tool.

Covers:
- consult_<name> tool schema appears when a subagent is registered
- search_event_log tool schema is always present
- Runtime dispatches consult_<name> to the subagent and records an event
- input-required surfaces with instructions to follow up via session_id
- pre/post_subagent_call hooks fire; pre_action/post_action do NOT
- search_event_log dispatches and returns rendered markdown
- search_event_log does NOT append an event
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
    SubagentContext,
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
from distributed_agent_harness.subagents.base import (
    SubagentClient,
    SubagentResponse,
)
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
        """Add an item to the list."""
        self.state.items.append(text)
        return text


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


class FakeSubagent(SubagentClient):
    """Scripted subagent — returns pre-baked responses."""

    def __init__(
        self,
        name: str,
        responses: list[SubagentResponse],
        progress: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = f"Fake {name}"
        self.show_when = None
        self._responses = list(responses)
        self._progress_chunks = list(progress or [])
        self.calls: list[tuple[str, str | None]] = []

    async def consult(
        self,
        message: str,
        session_id: str | None = None,
        timeout: float = 60.0,
        on_progress=None,
    ) -> SubagentResponse:
        self.calls.append((message, session_id))
        if on_progress is not None:
            for chunk in self._progress_chunks:
                await on_progress(chunk)
        return self._responses.pop(0)


def _runtime(world_class, llm: LLMProvider, hooks: HookRegistry | None = None) -> AgentRuntime:
    return AgentRuntime(
        world_class=world_class,
        namespace=InMemoryNamespace(),
        eventlog=InMemoryEventLog(),
        llm=llm,
        hooks=hooks,
    )


def _chat(text: str, channel: OutputChannel | None = None) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="p",
        reply_to=channel,
    )


# --------------------------------------------------------------------------- #
# Tool schema composition                                                      #
# --------------------------------------------------------------------------- #

class TestToolSchemas:
    @pytest.mark.asyncio
    async def test_search_event_log_is_always_present(self) -> None:
        llm = FakeLLM([Message(role=Role.ASSISTANT, content="done")])
        runtime = _runtime(TodoWorld, llm)
        await runtime.handle(_chat("hi"))

        _, tools = llm.calls[0]
        assert tools is not None
        names = {t.name for t in tools}
        assert "search_event_log" in names

    @pytest.mark.asyncio
    async def test_consult_appears_when_subagent_registered(self) -> None:
        llm = FakeLLM([Message(role=Role.ASSISTANT, content="ok")])
        runtime = _runtime(TodoWorld, llm)
        runtime.subagents.register(
            FakeSubagent("legal_research", [SubagentResponse(
                status="completed", content="ok",
            )]),
        )

        await runtime.handle(_chat("hi"))

        _, tools = llm.calls[0]
        assert tools is not None
        names = {t.name for t in tools}
        assert "consult_legal_research" in names

    @pytest.mark.asyncio
    async def test_consult_absent_when_no_subagents(self) -> None:
        llm = FakeLLM([Message(role=Role.ASSISTANT, content="ok")])
        runtime = _runtime(TodoWorld, llm)
        await runtime.handle(_chat("hi"))

        _, tools = llm.calls[0]
        names = {t.name for t in tools or []}
        assert not any(n.startswith("consult_") for n in names)


# --------------------------------------------------------------------------- #
# consult_<name> dispatch                                                      #
# --------------------------------------------------------------------------- #

class TestConsultDispatch:
    @pytest.mark.asyncio
    async def test_consult_dispatched_to_subagent(self) -> None:
        sub = FakeSubagent("legal_research", [SubagentResponse(
            status="completed", content="72 hours per Art. 33",
            session_id="ctx-a",
        )])
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "What does Art. 33 require?"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="Reported back."),
        ])
        runtime = _runtime(TodoWorld, llm)
        runtime.subagents.register(sub)

        await runtime.handle(_chat("ask legal"))

        assert sub.calls == [("What does Art. 33 require?", None)]
        # TOOL message in turn 2
        tool_msgs = [
            m for m in llm.calls[1][0] if m.role == Role.TOOL
        ]
        assert any("72 hours" in (m.content or "") for m in tool_msgs)
        assert any("session_id=ctx-a" in (m.content or "") for m in tool_msgs)
        assert any("[completed]" in (m.content or "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_consult_records_event(self) -> None:
        sub = FakeSubagent("legal_research", [SubagentResponse(
            status="completed", content="done", session_id="ctx-a",
        )])
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "hi"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = _runtime(TodoWorld, llm)
        runtime.subagents.register(sub)
        await runtime.handle(_chat("ask"))

        events = await runtime.eventlog.read_events("p")
        names = [e.action_name for e in events]
        assert "consult_legal_research" in names

    @pytest.mark.asyncio
    async def test_session_id_passed_through(self) -> None:
        sub = FakeSubagent("legal_research", [SubagentResponse(
            status="completed", content="ok", session_id="ctx-z",
        )])
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "follow-up", "session_id": "ctx-z"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = _runtime(TodoWorld, llm)
        runtime.subagents.register(sub)
        await runtime.handle(_chat("follow up"))

        assert sub.calls == [("follow-up", "ctx-z")]

    @pytest.mark.asyncio
    async def test_input_required_surfaces_follow_up_instruction(self) -> None:
        sub = FakeSubagent("legal_research", [SubagentResponse(
            status="input-required",
            content="Which jurisdiction — UK or EU?",
            session_id="ctx-b",
        )])
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "Help me with GDPR"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="OK."),
        ])
        runtime = _runtime(TodoWorld, llm)
        runtime.subagents.register(sub)
        await runtime.handle(_chat("ask"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "[input-required]" in body
        assert "ctx-b" in body
        assert "consult_legal_research" in body

    @pytest.mark.asyncio
    async def test_working_deltas_forwarded_as_thinking_events(self) -> None:
        sub = FakeSubagent(
            "legal_research",
            [SubagentResponse(status="completed", content="ok")],
            progress=["thinking… ", "almost… "],
        )
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "hi"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = _runtime(TodoWorld, llm)
        runtime.subagents.register(sub)

        ch = RecordingChannel()
        await runtime.handle(_chat("ask", ch))

        thinking = [e for e in ch.events if e.kind == OutputEventKind.THINKING]
        deltas = [e.payload.get("delta") for e in thinking]
        assert deltas == ["thinking… ", "almost… "]


# --------------------------------------------------------------------------- #
# Hooks                                                                        #
# --------------------------------------------------------------------------- #

class TestSubagentHooks:
    @pytest.mark.asyncio
    async def test_pre_and_post_subagent_call_fire(self) -> None:
        hooks = HookRegistry()
        pre_seen: list[SubagentContext] = []
        post_seen: list[tuple[SubagentContext, Any]] = []

        @hooks.on_pre_subagent_call("legal_research")
        async def pre(ctx: SubagentContext) -> None:
            pre_seen.append(ctx)

        @hooks.on_post_subagent_call("legal_research")
        async def post(ctx: SubagentContext, response: Any) -> None:
            post_seen.append((ctx, response))

        sub = FakeSubagent("legal_research", [SubagentResponse(
            status="completed", content="ok",
        )])
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "hi"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = _runtime(TodoWorld, llm, hooks=hooks)
        runtime.subagents.register(sub)
        await runtime.handle(_chat("ask"))

        assert len(pre_seen) == 1
        assert pre_seen[0].subagent_name == "legal_research"
        assert pre_seen[0].message == "hi"
        assert len(post_seen) == 1
        assert post_seen[0][1].status == "completed"

    @pytest.mark.asyncio
    async def test_pre_action_does_NOT_fire_for_subagent(self) -> None:
        hooks = HookRegistry()
        pre_actions: list[ActionContext] = []

        @hooks.on_pre_action()  # wildcard
        async def pre(ctx: ActionContext) -> None:
            pre_actions.append(ctx)

        sub = FakeSubagent("legal_research", [SubagentResponse(
            status="completed", content="ok",
        )])
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "hi"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = _runtime(TodoWorld, llm, hooks=hooks)
        runtime.subagents.register(sub)
        await runtime.handle(_chat("ask"))

        # No @action calls happened in this run — pre_action stays empty.
        assert pre_actions == []

    @pytest.mark.asyncio
    async def test_pre_subagent_call_can_block(self) -> None:
        hooks = HookRegistry()

        @hooks.on_pre_subagent_call("legal_research")
        async def block(ctx: SubagentContext) -> BlockDecision:
            return BlockDecision(reason="not approved")

        sub = FakeSubagent("legal_research", [SubagentResponse(
            status="completed", content="ok",
        )])
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_legal_research",
                    arguments={"message": "hi"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = _runtime(TodoWorld, llm, hooks=hooks)
        runtime.subagents.register(sub)
        await runtime.handle(_chat("ask"))

        # Subagent was never called.
        assert sub.calls == []
        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        assert any("not approved" in (m.content or "") for m in tool_msgs)


# --------------------------------------------------------------------------- #
# search_event_log dispatch                                                    #
# --------------------------------------------------------------------------- #

class TestSearchEventLog:
    @pytest.mark.asyncio
    async def test_search_returns_filtered_markdown(self) -> None:
        # Pre-seed two events so the search has something to find.
        from distributed_agent_harness.eventlog import Event

        eventlog = InMemoryEventLog()
        await eventlog.append("p", Event(
            project_id="p", action_name="consult_legal_research",
            kwargs={"message": "Art 33?"}, actor="agent",
            result_summary="status=completed session=ctx-x",
        ), 0)
        await eventlog.append("p", Event(
            project_id="p", action_name="add_item",
            kwargs={"text": "buy milk"}, actor="agent",
        ), 1)

        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="search_event_log",
                    arguments={"action_name_glob": "consult_*"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="here you go"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=InMemoryNamespace(),
            eventlog=eventlog,
            llm=llm,
        )
        await runtime.handle(_chat("search please"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "consult_legal_research" in body
        assert "add_item" not in body

    @pytest.mark.asyncio
    async def test_search_does_not_append_event(self) -> None:
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="search_event_log", arguments={},
                )],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        runtime = _runtime(TodoWorld, llm)
        offset_before = await runtime.eventlog.current_offset("p")
        await runtime.handle(_chat("search"))
        offset_after = await runtime.eventlog.current_offset("p")
        assert offset_after == offset_before
