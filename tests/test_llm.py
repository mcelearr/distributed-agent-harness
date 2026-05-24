"""
Tests for the LLM provider abstraction and the Mistral adapter.

The MistralProvider tests use ``respx`` would be nice but we keep dependencies
minimal — instead we substitute httpx with a transport mock via httpx's own
``MockTransport``.
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

from distributed_agent_harness.llm import Message, Role, ToolCall, ToolSchema
from distributed_agent_harness.llm_providers.mistral import MistralProvider


# --------------------------------------------------------------------------- #
# Encoding                                                                     #
# --------------------------------------------------------------------------- #

class TestMessageEncoding:
    def test_user_message(self) -> None:
        m = Message(role=Role.USER, content="hello")
        encoded = MistralProvider._encode_message(m)
        assert encoded == {"role": "user", "content": "hello"}

    def test_assistant_with_tool_calls(self) -> None:
        m = Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="c1", name="do_thing", arguments={"x": 1})],
        )
        encoded = MistralProvider._encode_message(m)
        assert encoded["role"] == "assistant"
        assert encoded["tool_calls"][0]["function"]["name"] == "do_thing"
        # Arguments are serialised as a JSON string on the wire
        assert json.loads(encoded["tool_calls"][0]["function"]["arguments"]) == {"x": 1}

    def test_tool_result_message(self) -> None:
        m = Message(
            role=Role.TOOL,
            content="result-text",
            tool_call_id="c1",
            name="do_thing",
        )
        encoded = MistralProvider._encode_message(m)
        assert encoded == {
            "role": "tool",
            "content": "result-text",
            "tool_call_id": "c1",
            "name": "do_thing",
        }


class TestToolEncoding:
    def test_tool_schema_encoding(self) -> None:
        t = ToolSchema(
            name="foo",
            description="Do foo.",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        encoded = MistralProvider._encode_tool(t)
        assert encoded["type"] == "function"
        assert encoded["function"]["name"] == "foo"
        assert encoded["function"]["parameters"]["properties"]["x"]["type"] == "integer"


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #

class TestMessageParsing:
    def test_plain_assistant(self) -> None:
        raw = {"role": "assistant", "content": "hi back"}
        msg = MistralProvider._parse_message(raw)
        assert msg.role == Role.ASSISTANT
        assert msg.content == "hi back"
        assert msg.tool_calls == []

    def test_tool_call(self) -> None:
        raw = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_42",
                "type": "function",
                "function": {
                    "name": "submit_pitch",
                    "arguments": '{"client_name": "Acme"}',
                },
            }],
        }
        msg = MistralProvider._parse_message(raw)
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc.id == "call_42"
        assert tc.name == "submit_pitch"
        assert tc.arguments == {"client_name": "Acme"}


class TestSseParsing:
    def test_content_chunk(self) -> None:
        line = 'data: {"choices":[{"delta":{"content":"hel"}}]}'
        chunk = MistralProvider._parse_sse_line(line)
        assert chunk is not None
        assert chunk.delta_content == "hel"

    def test_done_marker(self) -> None:
        assert MistralProvider._parse_sse_line("data: [DONE]") is None

    def test_empty_line(self) -> None:
        assert MistralProvider._parse_sse_line("") is None
        assert MistralProvider._parse_sse_line("  ") is None

    def test_finish_reason(self) -> None:
        line = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        chunk = MistralProvider._parse_sse_line(line)
        assert chunk is not None
        assert chunk.finish_reason == "stop"


# --------------------------------------------------------------------------- #
# End-to-end with mock transport                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


class TestChatComplete:
    @pytest.mark.asyncio
    async def test_calls_correct_endpoint_and_parses_response(
        self, api_key: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )

        mock_transport = httpx.MockTransport(handler)

        # Patch AsyncClient to use our mock transport
        original = httpx.AsyncClient

        def mock_client_factory(*args, **kwargs):
            kwargs["transport"] = mock_transport
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", mock_client_factory)

        provider = MistralProvider(model="mistral-medium-latest")
        result = await provider.chat_complete([Message(role=Role.USER, content="hi")])

        assert result.content == "ok"
        assert captured["url"].endswith("/v1/chat/completions")
        assert captured["headers"]["authorization"] == "Bearer test-key"
        assert captured["payload"]["model"] == "mistral-medium-latest"
        assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
        assert captured["payload"]["stream"] is False

    @pytest.mark.asyncio
    async def test_includes_tools_when_provided(
        self, api_key: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "k"}}]},
            )

        original = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda *a, **kw: original(*a, **{**kw, "transport": httpx.MockTransport(handler)}),
        )

        provider = MistralProvider()
        await provider.chat_complete(
            [Message(role=Role.USER, content="hi")],
            tools=[ToolSchema(name="t", description="d", parameters={"type": "object"})],
        )

        assert "tools" in captured["payload"]
        assert captured["payload"]["tools"][0]["function"]["name"] == "t"
        assert captured["payload"]["tool_choice"] == "auto"


class TestApiKeyHandling:
    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            MistralProvider()

    def test_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        provider = MistralProvider(api_key="explicit")
        assert provider._api_key == "explicit"
