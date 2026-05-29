"""
Tests for in-process subagents — AsyncSubagent and MessagingSubagent +
InMemoryMessageBus — plus runtime artefact persistence.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.eventlog import InMemoryEventLog
from distributed_agent_harness.hooks import (
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
from distributed_agent_harness.subagents import (
    Artefact,
    AsyncSubagent,
    InMemoryMessageBus,
    InProcessSubagent,
    MessagingSubagent,
    MessagingSubagentWorker,
    NoSubscriberError,
    SubagentResponse,
    SubagentTimeout,
)
from distributed_agent_harness.transport import (
    OutputEventKind,
    OutputChannel,
    OutputEvent,
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
        """Add an item."""
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


def _chat(text: str, channel: OutputChannel | None = None) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="demo",
        reply_to=channel,
    )


# --------------------------------------------------------------------------- #
# AsyncSubagent — direct                                                       #
# --------------------------------------------------------------------------- #

class TestAsyncSubagent:
    def test_is_in_process_marker(self) -> None:
        async def handler(message, session_id, on_progress):
            return SubagentResponse(status="completed", content="ok")

        sub = AsyncSubagent("worker", "Local worker", handler)
        assert isinstance(sub, InProcessSubagent)
        assert sub.name == "worker"

    @pytest.mark.asyncio
    async def test_handler_invoked_and_response_returned(self) -> None:
        async def handler(message, session_id, on_progress):
            return SubagentResponse(
                status="completed",
                content=f"got: {message}",
                session_id="ctx-x",
            )

        sub = AsyncSubagent("worker", "Local worker", handler)
        response = await sub.consult("hello")
        assert response.status == "completed"
        assert response.content == "got: hello"
        assert response.session_id == "ctx-x"

    @pytest.mark.asyncio
    async def test_progress_callback_propagates(self) -> None:
        async def handler(message, session_id, on_progress):
            if on_progress is not None:
                await on_progress("step 1… ")
                await on_progress("step 2… ")
            return SubagentResponse(status="completed", content="done")

        captured: list[str] = []

        async def on_progress(delta: str) -> None:
            captured.append(delta)

        sub = AsyncSubagent("worker", "Local worker", handler)
        await sub.consult("go", on_progress=on_progress)
        assert captured == ["step 1… ", "step 2… "]

    @pytest.mark.asyncio
    async def test_timeout_raises_subagent_timeout(self) -> None:
        async def slow_handler(message, session_id, on_progress):
            await asyncio.sleep(5)
            return SubagentResponse(status="completed", content="late")  # pragma: no cover

        sub = AsyncSubagent("slow", "Slow worker", slow_handler)
        with pytest.raises(SubagentTimeout):
            await sub.consult("hi", timeout=0.1)


# --------------------------------------------------------------------------- #
# InMemoryMessageBus + MessagingSubagent + Worker                              #
# --------------------------------------------------------------------------- #

class TestInMemoryMessageBus:
    @pytest.mark.asyncio
    async def test_request_routes_to_subscriber(self) -> None:
        bus = InMemoryMessageBus()

        async def handler(payload: dict) -> dict:
            return {"echo": payload["x"]}

        bus.subscribe("topic.echo", handler)
        reply = await bus.request("topic.echo", {"x": 42})
        assert reply == {"echo": 42}

    @pytest.mark.asyncio
    async def test_no_subscriber_raises(self) -> None:
        bus = InMemoryMessageBus()
        with pytest.raises(NoSubscriberError):
            await bus.request("nope", {})

    @pytest.mark.asyncio
    async def test_correlation_id_added_to_payload(self) -> None:
        bus = InMemoryMessageBus()
        received: list[dict] = []

        async def handler(payload: dict) -> dict:
            received.append(payload)
            return {}

        bus.subscribe("t", handler)
        await bus.request("t", {"k": "v"})
        assert "__correlation_id__" in received[0]
        assert received[0]["k"] == "v"

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        bus = InMemoryMessageBus()

        async def slow_handler(payload: dict) -> dict:
            await asyncio.sleep(5)
            return {}  # pragma: no cover

        bus.subscribe("slow", slow_handler)
        with pytest.raises(SubagentTimeout):
            await bus.request("slow", {}, timeout=0.1)

    def test_duplicate_subscription_raises(self) -> None:
        bus = InMemoryMessageBus()

        async def h(_):
            return {}  # pragma: no cover

        bus.subscribe("dup", h)
        with pytest.raises(ValueError, match="already has a subscriber"):
            bus.subscribe("dup", h)

    def test_unsubscribe_clears_handler(self) -> None:
        bus = InMemoryMessageBus()

        async def h(_):
            return {}  # pragma: no cover

        sub = bus.subscribe("t", h)
        sub.unsubscribe()
        # Re-subscribing now works.
        bus.subscribe("t", h)


class TestMessagingSubagent:
    @pytest.mark.asyncio
    async def test_round_trip_through_worker(self) -> None:
        bus = InMemoryMessageBus()

        async def serve(message: str, session_id: str | None) -> SubagentResponse:
            return SubagentResponse(
                status="completed",
                content=f"echo:{message}",
                session_id=session_id or "fresh",
            )

        worker = MessagingSubagentWorker(bus, "classify.requests", serve)
        sub = MessagingSubagent(
            name="classifier",
            description="A classifier",
            bus=bus,
            topic="classify.requests",
        )

        response = await sub.consult("hello", session_id=None)
        assert response.status == "completed"
        assert response.content == "echo:hello"
        assert response.session_id == "fresh"
        worker.stop()

    @pytest.mark.asyncio
    async def test_artefacts_round_trip(self) -> None:
        bus = InMemoryMessageBus()

        async def serve(message, session_id):
            return SubagentResponse(
                status="completed",
                content="here is your file",
                artefacts=[Artefact(
                    name="report.pdf",
                    content=b"%PDF-1.7" + b"x" * 100,
                    mime="application/pdf",
                    description="Summary report",
                )],
            )

        worker = MessagingSubagentWorker(bus, "t", serve)
        sub = MessagingSubagent("svc", "svc", bus=bus, topic="t")
        response = await sub.consult("go")
        assert len(response.artefacts) == 1
        art = response.artefacts[0]
        assert art.name == "report.pdf"
        assert art.content.startswith(b"%PDF-1.7")
        assert art.mime == "application/pdf"
        assert art.description == "Summary report"
        worker.stop()


# --------------------------------------------------------------------------- #
# Runtime: artefact persistence                                                #
# --------------------------------------------------------------------------- #

class TestRuntimeArtefactPersistence:
    @pytest.mark.asyncio
    async def test_artefact_written_to_namespace(self) -> None:
        pdf_bytes = b"%PDF-1.7" + b"\x00" * 500

        async def handler(message, session_id, on_progress):
            return SubagentResponse(
                status="completed",
                content="here is the brief",
                artefacts=[Artefact(
                    name="legal_brief.pdf",
                    content=pdf_bytes,
                    mime="application/pdf",
                    description="Draft legal brief",
                )],
            )

        namespace = InMemoryNamespace()
        eventlog = InMemoryEventLog()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_drafter",
                    arguments={"message": "draft a brief"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=namespace,
            eventlog=eventlog,
            llm=llm,
        )
        runtime.subagents.register(AsyncSubagent("drafter", "drafts docs", handler))
        await runtime.handle(_chat("draft please"))

        # The artefact should be at demo/artefacts/<sha[:8]>__legal_brief.pdf
        paths = namespace.list_docs("demo/artefacts/")
        assert len(paths) == 1
        path = paths[0]
        assert path.startswith("demo/artefacts/")
        assert path.endswith("__legal_brief.pdf")
        # Content matches
        assert namespace.read_binary(path) == pdf_bytes
        # Hash prefix matches
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        assert sha[:8] in path

    @pytest.mark.asyncio
    async def test_event_records_artefact_path(self) -> None:
        async def handler(message, session_id, on_progress):
            return SubagentResponse(
                status="completed",
                content="see attached",
                artefacts=[Artefact(name="foo.pdf", content=b"%PDF" + b"x" * 50)],
            )

        namespace = InMemoryNamespace()
        eventlog = InMemoryEventLog()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_drafter",
                    arguments={"message": "go"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=namespace,
            eventlog=eventlog,
            llm=llm,
        )
        runtime.subagents.register(AsyncSubagent("drafter", "drafts", handler))
        await runtime.handle(_chat("go"))

        events = await eventlog.read_events("demo")
        consult_events = [e for e in events if e.action_name.startswith("consult_")]
        assert len(consult_events) == 1
        assert "artefacts=" in (consult_events[0].result_summary or "")
        assert "demo/artefacts/" in (consult_events[0].result_summary or "")

    @pytest.mark.asyncio
    async def test_tool_message_lists_artefact_paths_for_llm(self) -> None:
        async def handler(message, session_id, on_progress):
            return SubagentResponse(
                status="completed",
                content="see attached",
                artefacts=[Artefact(
                    name="foo.pdf",
                    content=b"%PDF" + b"x" * 50,
                    mime="application/pdf",
                    description="A draft",
                )],
            )

        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_drafter",
                    arguments={"message": "go"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        runtime.subagents.register(AsyncSubagent("drafter", "drafts", handler))
        await runtime.handle(_chat("go"))

        # Second LLM call sees the TOOL message with artefact paths in the body.
        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "Artefacts written" in body
        assert "demo/artefacts/" in body
        assert "Use `read`" in body
        assert "A draft" in body  # description rendered

    @pytest.mark.asyncio
    async def test_sanitisation_handles_path_unsafe_names(self) -> None:
        async def handler(message, session_id, on_progress):
            return SubagentResponse(
                status="completed",
                content="ok",
                artefacts=[Artefact(name="../../etc/passwd", content=b"x")],
            )

        namespace = InMemoryNamespace()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_evil",
                    arguments={"message": "go"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=namespace,
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        runtime.subagents.register(AsyncSubagent("evil", "evil", handler))
        await runtime.handle(_chat("attack"))

        paths = namespace.list_docs()
        # Nothing escaped outside demo/artefacts/
        assert all(p.startswith("demo/artefacts/") for p in paths if p.startswith("demo"))
        # None of the stored paths contain "../"
        assert not any("../" in p for p in paths)

    @pytest.mark.asyncio
    async def test_subagent_hooks_still_fire(self) -> None:
        hooks = HookRegistry()
        seen_pre: list[SubagentContext] = []
        seen_post: list[Any] = []

        @hooks.on_pre_subagent_call("drafter")
        async def pre(ctx):
            seen_pre.append(ctx)

        @hooks.on_post_subagent_call("drafter")
        async def post(ctx, response):
            seen_post.append((ctx, response))

        async def handler(message, session_id, on_progress):
            return SubagentResponse(status="completed", content="ok")

        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_drafter",
                    arguments={"message": "hi"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
            llm=llm,
            hooks=hooks,
        )
        runtime.subagents.register(AsyncSubagent("drafter", "drafts", handler))
        await runtime.handle(_chat("hi"))

        assert len(seen_pre) == 1
        assert len(seen_post) == 1


# --------------------------------------------------------------------------- #
# Full loop: MessagingSubagent registered on runtime with worker               #
# --------------------------------------------------------------------------- #

class TestMessagingSubagentInRuntime:
    @pytest.mark.asyncio
    async def test_full_loop_through_runtime(self) -> None:
        bus = InMemoryMessageBus()

        async def serve(message, session_id):
            return SubagentResponse(
                status="completed",
                content=f"reply to {message}",
                session_id="ctx-m",
            )

        worker = MessagingSubagentWorker(bus, "svc.requests", serve)
        sub = MessagingSubagent(
            name="svc", description="messaging svc",
            bus=bus, topic="svc.requests",
        )

        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="consult_svc",
                    arguments={"message": "ping"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        runtime.subagents.register(sub)

        await runtime.handle(_chat("ping"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "reply to ping" in body
        assert "ctx-m" in body
        worker.stop()
