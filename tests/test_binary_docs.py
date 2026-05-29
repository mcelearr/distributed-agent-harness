"""
Tests for binary document support: NamespaceAdapter.read_binary / write_binary
/ doc_info, plus the read meta-tool's auto-detection of binary docs.
"""
from __future__ import annotations

import hashlib
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
from distributed_agent_harness.namespace import DocInfo, NamespaceAdapter
from distributed_agent_harness.namespace_browse import (
    list_dir,
    read_doc,
    render_read,
)
from distributed_agent_harness.runtime import AgentRuntime
from distributed_agent_harness.transport import TriggerEvent, TriggerKind
from distributed_agent_harness.world import BaseWorldEnvironment, action


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

PDF_BYTES = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + b"x" * 200  # Fake PDF header + filler
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"y" * 1024              # Fake PNG header + filler


class TodoState(BaseModel):
    items: list[str] = []


class TodoWorld(BaseWorldEnvironment):
    State = TodoState

    @action
    def add_item(self, text: str) -> str:
        """Add an item."""
        self.state.items.append(text)
        return text


# --------------------------------------------------------------------------- #
# InMemoryNamespace binary round-trip                                          #
# --------------------------------------------------------------------------- #

class TestInMemoryNamespaceBinary:
    def test_write_and_read_binary(self) -> None:
        ns = InMemoryNamespace()
        ns.write_binary("demo/artefacts/foo.pdf", PDF_BYTES)
        assert ns.read_binary("demo/artefacts/foo.pdf") == PDF_BYTES

    def test_read_doc_on_binary_path_returns_none(self) -> None:
        ns = InMemoryNamespace()
        ns.write_binary("demo/artefacts/foo.pdf", PDF_BYTES)
        assert ns.read_doc("demo/artefacts/foo.pdf") is None

    def test_read_binary_on_text_path_returns_none(self) -> None:
        ns = InMemoryNamespace()
        ns.write_doc("demo/summary.md", "# Hello")
        assert ns.read_binary("demo/summary.md") is None

    def test_text_paths_unchanged_by_binary_storage(self) -> None:
        ns = InMemoryNamespace()
        ns.write_doc("demo/summary.md", "# Hello")
        ns.write_binary("demo/artefacts/foo.pdf", PDF_BYTES)
        assert ns.read_doc("demo/summary.md") == "# Hello"

    def test_write_binary_rejects_non_bytes(self) -> None:
        ns = InMemoryNamespace()
        with pytest.raises(TypeError, match="expects bytes"):
            ns.write_binary("demo/a.pdf", "not bytes")  # type: ignore[arg-type]

    def test_list_docs_returns_paths_regardless_of_kind(self) -> None:
        ns = InMemoryNamespace()
        ns.write_doc("demo/summary.md", "# X")
        ns.write_binary("demo/foo.pdf", PDF_BYTES)
        assert sorted(ns.list_docs("demo/")) == ["demo/foo.pdf", "demo/summary.md"]

    def test_delete_works_for_both(self) -> None:
        ns = InMemoryNamespace()
        ns.write_doc("a.md", "x")
        ns.write_binary("b.pdf", b"y")
        ns.delete_doc("a.md")
        ns.delete_doc("b.pdf")
        assert ns.list_docs() == []


# --------------------------------------------------------------------------- #
# doc_info                                                                     #
# --------------------------------------------------------------------------- #

class TestDocInfo:
    def test_doc_info_for_text(self) -> None:
        ns = InMemoryNamespace()
        ns.write_doc("demo/state.json", '{"counter": 7}')
        info = ns.doc_info("demo/state.json")
        assert info is not None
        assert info.kind == "text"
        assert info.size == len('{"counter": 7}'.encode("utf-8"))
        assert info.mime is not None  # JSON gets a mime guess
        assert info.sha256 is not None
        assert len(info.sha256) == 64

    def test_doc_info_for_binary(self) -> None:
        ns = InMemoryNamespace()
        ns.write_binary("demo/foo.pdf", PDF_BYTES)
        info = ns.doc_info("demo/foo.pdf")
        assert info is not None
        assert info.kind == "binary"
        assert info.size == len(PDF_BYTES)
        assert info.mime == "application/pdf"
        assert info.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()

    def test_doc_info_missing(self) -> None:
        ns = InMemoryNamespace()
        assert ns.doc_info("nope") is None


# --------------------------------------------------------------------------- #
# Default doc_info on a non-overriding adapter                                 #
# --------------------------------------------------------------------------- #

class _TextOnlyAdapter(NamespaceAdapter):
    """Adapter that only knows about text — used to verify defaults work."""

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    def read_doc(self, path: str) -> str | None:
        return self._docs.get(path)

    def write_doc(self, path: str, content: str) -> None:
        self._docs[path] = content

    def list_docs(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self._docs if p.startswith(prefix))


class TestAdapterDefaults:
    def test_text_only_adapter_can_compute_doc_info(self) -> None:
        a = _TextOnlyAdapter()
        a.write_doc("a", "hello")
        info = a.doc_info("a")
        assert info is not None
        assert info.kind == "text"
        assert info.size == 5

    def test_text_only_adapter_read_binary_raises_not_implemented(self) -> None:
        a = _TextOnlyAdapter()
        with pytest.raises(NotImplementedError):
            a.read_binary("a")

    def test_text_only_adapter_write_binary_raises_not_implemented(self) -> None:
        a = _TextOnlyAdapter()
        with pytest.raises(NotImplementedError):
            a.write_binary("a", b"x")


# --------------------------------------------------------------------------- #
# namespace_browse handles binary                                              #
# --------------------------------------------------------------------------- #

class TestBrowseBinary:
    def test_list_dir_reports_binary_size_correctly(self) -> None:
        ns = InMemoryNamespace()
        ns.write_binary("demo/foo.pdf", PDF_BYTES)
        ns.write_doc("demo/x.md", "hi")
        entries = list_dir(ns, "demo/")
        sizes = {e.name: e.size for e in entries if e.kind == "file"}
        assert sizes["foo.pdf"] == len(PDF_BYTES)
        assert sizes["x.md"] == 2

    def test_read_doc_returns_binary_meta(self) -> None:
        ns = InMemoryNamespace()
        ns.write_binary("demo/foo.pdf", PDF_BYTES)
        content, meta = read_doc(ns, "demo/foo.pdf")
        assert content is None
        assert meta["binary"] is True
        assert meta["size"] == len(PDF_BYTES)
        assert meta["mime"] == "application/pdf"
        assert meta["sha256"] == hashlib.sha256(PDF_BYTES).hexdigest()

    def test_render_read_binary_descriptor(self) -> None:
        ns = InMemoryNamespace()
        ns.write_binary("demo/foo.pdf", PDF_BYTES)
        content, meta = read_doc(ns, "demo/foo.pdf")
        rendered = render_read(content, "demo/foo.pdf", meta)
        assert "binary doc" in rendered
        assert "application/pdf" in rendered
        assert "sha256:" in rendered

    def test_read_doc_missing_still_returns_none(self) -> None:
        ns = InMemoryNamespace()
        content, meta = read_doc(ns, "nope")
        assert content is None
        assert meta == {}

    def test_read_doc_on_text_only_adapter_handles_missing(self) -> None:
        # When the adapter doesn't implement read_binary, a missing path
        # still resolves to (None, {}), not an exception.
        a = _TextOnlyAdapter()
        content, meta = read_doc(a, "nope")
        assert content is None
        assert meta == {}


# --------------------------------------------------------------------------- #
# read meta-tool dispatches to binary descriptor                               #
# --------------------------------------------------------------------------- #

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


def _chat(text: str) -> TriggerEvent:
    return TriggerEvent(
        source="test",
        kind=TriggerKind.CHAT_MESSAGE,
        payload={"text": text},
        project_id="demo",
    )


class TestReadMetaToolBinary:
    @pytest.mark.asyncio
    async def test_read_pdf_returns_descriptor(self) -> None:
        namespace = InMemoryNamespace()
        namespace.write_binary("demo/artefacts/foo.pdf", PDF_BYTES)
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="read",
                    arguments={"path": "demo/artefacts/foo.pdf"},
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
        await runtime.handle(_chat("read pdf"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "binary doc" in body
        assert "application/pdf" in body
        assert "sha256:" in body
        # Size human-readable
        assert "B" in body or "KB" in body

    @pytest.mark.asyncio
    async def test_read_png_returns_image_descriptor(self) -> None:
        namespace = InMemoryNamespace()
        namespace.write_binary("demo/artefacts/bar.png", PNG_BYTES)
        llm = FakeLLM([
            Message(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(
                    id="c1", name="read",
                    arguments={"path": "demo/artefacts/bar.png"},
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
        await runtime.handle(_chat("read png"))

        tool_msgs = [m for m in llm.calls[1][0] if m.role == Role.TOOL]
        body = tool_msgs[-1].content or ""
        assert "image/png" in body
