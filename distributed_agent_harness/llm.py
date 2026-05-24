"""
LLM provider abstraction.

A thin, OpenAI-shaped interface that maps cleanly to Mistral, OpenAI, Anthropic,
and most other modern chat-completion APIs. We avoid LangChain and similar
heavy abstractions — most providers speak almost identical wire formats, and
a tiny ABC is enough to adapt them.

Adding a new provider means writing one subclass of ``LLMProvider``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool/function call request emitted by the model."""
    id: str
    name: str
    arguments: dict[str, Any]   # parsed from JSON


@dataclass
class Message:
    """A single chat message in the conversation."""
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For role == TOOL: the id of the call this is responding to, and the tool name.
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolSchema:
    """A function/tool exposed to the model."""
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema for the parameters object


@dataclass
class CompletionChunk:
    """One incremental chunk from a streaming completion."""
    delta_content: str | None = None
    delta_tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None   # "stop" | "tool_calls" | "length" | None


class LLMProvider(ABC):
    """
    Provider-agnostic chat-completion interface.

    Implementations should:
    - Accept the typed ``Message`` / ``ToolSchema`` objects above
    - Convert to/from the provider's wire format internally
    - Return a single ``Message`` (with ``content`` and/or ``tool_calls``) for
      non-streaming calls, or yield ``CompletionChunk``s for streaming calls.

    No global state, no LangChain. Just HTTP + serialisation.
    """

    @abstractmethod
    async def chat_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> Message:
        """Run a non-streaming completion and return the assistant's response."""

    @abstractmethod
    def chat_complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        """Run a streaming completion, yielding chunks as they arrive."""
