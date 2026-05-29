"""
A2ASubagent — speaks the A2A protocol over HTTP + SSE.

Wire shape we implement (subset of the A2A spec sufficient for v1):

1. POST ``<card.url>/tasks/send`` with
   ``{"message": {"role": "user", "parts": [{"text": "..."}]}, "contextId": <session_id or null>}``.
2. The server replies ``{"taskId": "...", "contextId": "..."}``.
3. GET ``<card.url>/tasks/{taskId}/events`` opens an SSE stream.
4. ``working`` events carry an incremental text delta in ``data``.
5. The terminal event is one of ``completed`` / ``input-required`` / ``failed``
   with ``{"text": "...", "contextId": "..."}``.

We use ``httpx.AsyncClient`` directly; SSE parsing is a small line buffer.
No external SSE library — the protocol is text, three rules, easy.

The 60 s watchdog resets on **any** inbound bytes (event, comment, ping),
so a polite server emitting `: keep-alive` every 30 s stays alive
indefinitely. Silence past the window cancels the stream and raises
``SubagentTimeout``.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx

from .base import (
    AgentCard,
    ProgressCallback,
    SubagentClient,
    SubagentResponse,
    SubagentTimeout,
)

if TYPE_CHECKING:
    from ..world import Predicate


# Type alias for the auth knob — either static headers, or a callable that
# computes fresh headers per call (for token refresh, signed requests, ...).
AuthHeaders = dict[str, str] | Callable[[], dict[str, str]] | None


class A2ASubagent(SubagentClient):
    """Subagent client speaking A2A over HTTP + SSE."""

    def __init__(
        self,
        card: AgentCard,
        auth: AuthHeaders = None,
        show_when: "Predicate | None" = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.card = card
        self.name = card.name
        self.description = card.description
        self.show_when = show_when
        self._auth = auth
        # Caller may inject a pre-configured client (useful for tests with
        # MockTransport and for sharing a connection pool across subagents).
        self._client = client
        self._owns_client = client is None

    async def consult(
        self,
        message: str,
        session_id: str | None = None,
        timeout: float = 60.0,
        on_progress: ProgressCallback | None = None,
    ) -> SubagentResponse:
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(None))
        try:
            return await self._consult_with_client(
                client, message, session_id, timeout, on_progress,
            )
        finally:
            if self._owns_client:
                await client.aclose()

    # ----------------------------------------------------------------------- #
    # Internals                                                                #
    # ----------------------------------------------------------------------- #

    async def _consult_with_client(
        self,
        client: httpx.AsyncClient,
        message: str,
        session_id: str | None,
        timeout: float,
        on_progress: ProgressCallback | None,
    ) -> SubagentResponse:
        headers = self._resolve_auth()
        # ----- step 1: submit the task
        submit_resp = await client.post(
            f"{self.card.url.rstrip('/')}/tasks/send",
            json={
                "message": {
                    "role": "user",
                    "parts": [{"text": message}],
                },
                "contextId": session_id,
            },
            headers=headers,
            timeout=timeout,
        )
        submit_resp.raise_for_status()
        submit_body = submit_resp.json()
        task_id = submit_body["taskId"]
        context_id = submit_body.get("contextId", session_id)

        # ----- step 2: stream task events
        return await self._stream_task(
            client=client,
            task_id=task_id,
            context_id=context_id,
            timeout=timeout,
            on_progress=on_progress,
            headers=headers,
        )

    async def _stream_task(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        context_id: str | None,
        timeout: float,
        on_progress: ProgressCallback | None,
        headers: dict[str, str],
    ) -> SubagentResponse:
        url = f"{self.card.url.rstrip('/')}/tasks/{task_id}/events"
        sse_headers = {"Accept": "text/event-stream", **headers}

        accumulated_working = ""
        async with client.stream("GET", url, headers=sse_headers, timeout=None) as stream:
            stream.raise_for_status()
            async for event_type, data in _aiter_sse_events(stream, timeout=timeout):
                if event_type == "working":
                    delta = _extract_text(data)
                    if delta:
                        accumulated_working += delta
                        if on_progress is not None:
                            await on_progress(delta)
                    continue
                if event_type in ("completed", "input-required", "failed"):
                    text = _extract_text(data) or accumulated_working
                    final_context = (
                        data.get("contextId") if isinstance(data, dict) else None
                    ) or context_id
                    return SubagentResponse(
                        status=event_type,
                        content=text,
                        session_id=final_context,
                        metadata=data if isinstance(data, dict) else {},
                    )
                # Unknown event types are ignored; the watchdog still
                # treats them as activity.

        # Stream ended without a terminal event — treat as failure.
        return SubagentResponse(
            status="failed",
            content="A2A stream closed before a terminal event arrived",
            session_id=context_id,
            metadata={},
        )

    def _resolve_auth(self) -> dict[str, str]:
        if self._auth is None:
            return {}
        if callable(self._auth):
            return dict(self._auth())
        return dict(self._auth)


# --------------------------------------------------------------------------- #
# SSE parsing                                                                  #
# --------------------------------------------------------------------------- #

async def _aiter_sse_events(
    response: httpx.Response,
    timeout: float,
):
    """Yield ``(event_type, payload_dict_or_text)`` per SSE event.

    Implements only what we need: ``event:`` and ``data:`` fields, blank
    line as event terminator, ``:`` comments treated as activity. The
    watchdog resets on **any** inbound line — keep-alives keep us alive.
    """
    event_type: str | None = None
    data_lines: list[str] = []

    line_iter = response.aiter_lines().__aiter__()
    while True:
        try:
            line = await asyncio.wait_for(line_iter.__anext__(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise SubagentTimeout(
                f"No bytes from subagent for {timeout}s"
            ) from exc
        except StopAsyncIteration:
            # Stream ended.
            if event_type is not None or data_lines:
                yield event_type or "message", _parse_data(data_lines)
            return

        if line == "":
            # Blank line — dispatch the buffered event.
            if event_type is not None or data_lines:
                yield event_type or "message", _parse_data(data_lines)
            event_type = None
            data_lines = []
            continue

        if line.startswith(":"):
            # Comment / keep-alive — counts as activity, no dispatch.
            continue

        field, _, value = line.partition(":")
        value = value.lstrip(" ")
        if field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)


def _parse_data(data_lines: list[str]) -> Any:
    """Try JSON, fall back to raw text."""
    raw = "\n".join(data_lines)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw}


def _extract_text(payload: Any) -> str:
    """Pull a human-readable text out of an A2A event payload."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    if "text" in payload and isinstance(payload["text"], str):
        return payload["text"]
    # A2A "message" shape: {"role": ..., "parts": [{"text": "..."}, ...]}
    message = payload.get("message")
    if isinstance(message, dict):
        parts = message.get("parts", [])
        return "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        )
    return ""
