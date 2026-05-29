"""
Runtime-level tests for the three always-on namespace meta-tools (`ls`,
`read`, `grep`) and the reserved-name collision check.
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
from distributed_agent_harness.runtime import AgentRuntime
from distributed_agent_harness.transport import (
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


def _runtime_with_seeded_namespace() -> tuple[AgentRuntime, FakeLLM, InMemoryNamespace]:
    """Build a runtime over a namespace pre-seeded with a few docs."""
    namespace = InMemoryNamespace()
    namespace.write_doc("demo/state.json", '{"counter": 7}')
    namespace.write_doc("demo/summary.md", "# Demo\n\nAcme summary here.")
    namespace.write_doc(
        "demo/event_log.md",
        "# Event Log\n\n- offset 0: did a thing\n- offset 1: did another thing\n",
    )
    namespace.write_doc("demo/artefacts/legal_brief.md", "Brief about Acme Corp.")
    return AgentRuntime, FakeLLM, namespace


def _chat(text: str) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="demo",
    )


# --------------------------------------------------------------------------- #
# Tool schemas                                                                 #
# --------------------------------------------------------------------------- #

class TestMetaToolSchemas:
    @pytest.mark.asyncio
    async def test_ls_read_grep_present_at_all_times(self) -> None:
        llm = FakeLLM([Message(role=Role.ASSISTANT, content="hi")])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        await runtime.handle(_chat("hi"))

        _, tools = llm.calls[0]
        names = {t.name for t in tools or []}
        assert {"ls", "read", "grep", "search_event_log"} <= names


# --------------------------------------------------------------------------- #
# ls / read / grep dispatch                                                    #
# --------------------------------------------------------------------------- #

class TestLsTool:
    @pytest.mark.asyncio
    async def test_ls_returns_directory_listing(self) -> None:
        _, _, namespace = _runtime_with_seeded_namespace()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="ls", arguments={"path": "demo/"},
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
        await runtime.handle(_chat("ls"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "artefacts/" in body
        assert "event_log.md" in body
        assert "summary.md" in body


class TestReadTool:
    @pytest.mark.asyncio
    async def test_read_returns_text(self) -> None:
        _, _, namespace = _runtime_with_seeded_namespace()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="read",
                    arguments={"path": "demo/summary.md"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=namespace,
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        await runtime.handle(_chat("read"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "Acme summary" in body

    @pytest.mark.asyncio
    async def test_read_offset_and_limit(self) -> None:
        _, _, namespace = _runtime_with_seeded_namespace()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="read",
                    arguments={
                        "path": "demo/event_log.md",
                        "offset": 3,
                        "limit": 1,
                    },
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=namespace,
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        await runtime.handle(_chat("read"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        # Line 3 of event_log.md is the first event entry
        assert "did a thing" in body

    @pytest.mark.asyncio
    async def test_read_missing_path_errors_cleanly(self) -> None:
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="read", arguments={},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        await runtime.handle(_chat("bad read"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        assert any("'path' is required" in (m.content or "") for m in tool_msgs)


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_grep_finds_across_docs(self) -> None:
        _, _, namespace = _runtime_with_seeded_namespace()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="grep",
                    arguments={"pattern": "Acme", "path": "demo/"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=namespace,
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        await runtime.handle(_chat("grep"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "demo/summary.md" in body
        assert "demo/artefacts/legal_brief.md" in body

    @pytest.mark.asyncio
    async def test_grep_missing_pattern_errors_cleanly(self) -> None:
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="grep", arguments={"path": "demo/"},
                )],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
            llm=llm,
        )
        await runtime.handle(_chat("bad grep"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        assert any("'pattern' is required" in (m.content or "") for m in tool_msgs)


# --------------------------------------------------------------------------- #
# Meta-tools never append to the event log                                     #
# --------------------------------------------------------------------------- #

class TestMetaToolsAreReadOnly:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name,args",
        [
            ("ls", {"path": ""}),
            ("read", {"path": "demo/summary.md"}),
            ("grep", {"pattern": "anything"}),
        ],
    )
    async def test_no_event_appended(self, tool_name: str, args: dict) -> None:
        _, _, namespace = _runtime_with_seeded_namespace()
        eventlog = InMemoryEventLog()
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name=tool_name, arguments=args)],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ])
        runtime = AgentRuntime(
            world_class=TodoWorld,
            namespace=namespace,
            eventlog=eventlog,
            llm=llm,
        )
        before = await eventlog.current_offset("demo")
        await runtime.handle(_chat("read-only"))
        after = await eventlog.current_offset("demo")
        assert before == after


# --------------------------------------------------------------------------- #
# Reserved-name collision check                                                #
# --------------------------------------------------------------------------- #

class TestReservedNameCollision:
    @pytest.mark.parametrize("reserved", ["ls", "read", "grep", "search_event_log"])
    def test_action_named_after_meta_tool_raises(self, reserved: str) -> None:
        # Build a class that defines an @action whose name clashes.
        ns: dict[str, Any] = {
            "State": TodoState,
            "__annotations__": {},
        }

        @action
        def colliding(self, x: str = "") -> str:  # pragma: no cover
            return x

        # Rename to the reserved name and attach.
        colliding.__name__ = reserved
        ns[reserved] = colliding

        BadWorld = type("BadWorld", (BaseWorldEnvironment,), ns)

        with pytest.raises(ValueError, match="collide with built-in meta-tools"):
            AgentRuntime(
                world_class=BadWorld,
                namespace=InMemoryNamespace(),
                eventlog=InMemoryEventLog(),
                llm=FakeLLM([]),
            )

    def test_action_with_consult_prefix_raises(self) -> None:
        ns: dict[str, Any] = {
            "State": TodoState,
            "__annotations__": {},
        }

        @action
        def consult_local(self, message: str = "") -> str:  # pragma: no cover
            return message

        ns["consult_local"] = consult_local
        BadWorld = type("BadWorld", (BaseWorldEnvironment,), ns)

        with pytest.raises(ValueError, match="reserved subagent prefix"):
            AgentRuntime(
                world_class=BadWorld,
                namespace=InMemoryNamespace(),
                eventlog=InMemoryEventLog(),
                llm=FakeLLM([]),
            )
