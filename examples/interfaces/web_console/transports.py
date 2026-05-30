"""
HTTP + SSE adapters for the harness `TriggerSource` / `OutputChannel`
interfaces.

These two classes are the entire SDK-touching surface of the web console.
Everything else in this package (`server.py`, `registry.py`, the static
files in `web/`) is plain web plumbing on top of these adapters.

Pattern:

- `SseOutputChannel` — one instance per browser connection. Buffers
  `OutputEvent`s in an `asyncio.Queue` and yields them to the SSE stream.
- `HttpTriggerSource` — wraps the SDK's `TriggerSource` contract around a
  single `TriggerEvent` produced from a POST body. The web server creates
  one per incoming chat message and runs the runtime against it.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterator

from distributed_agent_harness.transport import (
    OutputChannel,
    OutputEvent,
    TriggerEvent,
    TriggerKind,
    TriggerSource,
)


_SENTINEL: Any = object()


class SseOutputChannel(OutputChannel):
    """An `OutputChannel` that pushes events onto an `asyncio.Queue`.

    The web server consumes the queue and writes each event to a
    Server-Sent Events stream. A single channel is reused for the lifetime
    of one chat turn; `close()` puts a sentinel that ends the stream.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def emit(self, event: OutputEvent) -> None:
        await self._queue.put(event)

    async def close(self) -> None:
        await self._queue.put(_SENTINEL)

    async def stream(self) -> AsyncIterator[str]:
        """Yield each queued event as an SSE-formatted string."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield _format_sse(item)


class HttpTriggerSource(TriggerSource):
    """Wraps a single inbound HTTP message as the harness `TriggerSource`.

    The runtime expects an async iterator of events. For HTTP we know
    exactly one event will fire per request, so we yield it and stop.
    """

    def __init__(
        self,
        *,
        project_id: str,
        text: str,
        reply_to: OutputChannel,
    ) -> None:
        self._event = TriggerEvent(
            source="web_console",
            kind=TriggerKind.CHAT_MESSAGE,
            payload={"text": text},
            project_id=project_id,
            reply_to=reply_to,
        )

    async def events(self) -> AsyncIterator[TriggerEvent]:
        yield self._event


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _format_sse(event: OutputEvent) -> str:
    """Encode an `OutputEvent` as a single SSE message."""
    body = {
        "kind": event.kind.value,
        "payload": _to_jsonable(event.payload),
    }
    return f"data: {json.dumps(body)}\n\n"


def _to_jsonable(value: Any) -> Any:
    """Best-effort JSON-friendly coercion for arbitrary payload dicts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    return str(value)
