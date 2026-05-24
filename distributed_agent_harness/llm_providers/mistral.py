"""
MistralProvider — Mistral La Plateforme chat completion adapter.

Targets the standard ``/v1/chat/completions`` endpoint, which is OpenAI-compatible.
Defaults to ``mistral-medium-latest`` (currently Mistral Medium 3.5: 128B dense,
256K context, open weights, designed for agentic multi-tool workflows).

Set ``MISTRAL_API_KEY`` in the environment, or pass ``api_key=...`` to the
constructor. The ``base_url`` is overridable so the same provider works against
self-hosted deployments of the open-weights model.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import httpx

from ..llm import (
    CompletionChunk,
    LLMProvider,
    Message,
    Role,
    ToolCall,
    ToolSchema,
)


class MistralProvider(LLMProvider):
    """
    Mistral chat-completion provider.

    Example::

        provider = MistralProvider(model="mistral-medium-latest")
        msg = await provider.chat_complete([
            Message(role=Role.USER, content="Hello")
        ])

    For self-hosted deployments::

        provider = MistralProvider(
            model="mistral-medium-3.5",
            base_url="http://my-vllm-host:8000/v1",
            api_key="not-required-but-set-something",
        )
    """

    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
    DEFAULT_MODEL = "mistral-medium-latest"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY not set. Either export it or pass api_key=... "
                "to MistralProvider."
            )

    # ----------------------------------------------------------------------- #
    # Public API                                                               #
    # ----------------------------------------------------------------------- #

    async def chat_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> Message:
        payload = self._build_payload(messages, tools, stream=False, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return self._parse_message(data["choices"][0]["message"])

    async def chat_complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        payload = self._build_payload(messages, tools, stream=True, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self._parse_sse_line(line)
                    if chunk is not None:
                        yield chunk

    # ----------------------------------------------------------------------- #
    # Internals                                                                #
    # ----------------------------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._encode_message(m) for m in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [self._encode_tool(t) for t in tools]
            payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
        # Allow caller to pass through any provider-specific kwargs
        payload.update(kwargs)
        return payload

    @staticmethod
    def _encode_message(msg: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": msg.role.value}
        if msg.content is not None:
            out["content"] = msg.content
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id is not None:
            out["tool_call_id"] = msg.tool_call_id
        if msg.name is not None:
            out["name"] = msg.name
        return out

    @staticmethod
    def _encode_tool(tool: ToolSchema) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _parse_message(raw: dict[str, Any]) -> Message:
        tool_calls: list[ToolCall] = []
        for tc in raw.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "{}")
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )
        return Message(
            role=Role(raw.get("role", "assistant")),
            content=raw.get("content"),
            tool_calls=tool_calls,
        )

    @staticmethod
    def _parse_sse_line(line: str) -> CompletionChunk | None:
        """Parse one SSE line, returning None for keep-alives and [DONE]."""
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return None
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None

        choice = (obj.get("choices") or [{}])[0]
        delta = choice.get("delta", {})

        tool_calls: list[ToolCall] = []
        for tc in delta.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "")
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        return CompletionChunk(
            delta_content=delta.get("content"),
            delta_tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
        )
