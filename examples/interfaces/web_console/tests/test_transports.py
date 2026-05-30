"""Smoke tests for the web_console transports.

The console itself is not part of the SDK; these tests just confirm that
the bundled adapters honour the `TriggerSource` / `OutputChannel` contracts.
"""
from __future__ import annotations

import json

import pytest

from distributed_agent_harness.transport import (
    OutputEvent,
    OutputEventKind,
    TriggerKind,
)

from examples.interfaces.web_console.transports import (
    HttpTriggerSource,
    SseOutputChannel,
)


@pytest.mark.asyncio
async def test_http_trigger_yields_one_event() -> None:
    channel = SseOutputChannel()
    source = HttpTriggerSource(project_id="p1", text="hello", reply_to=channel)

    events = [e async for e in source.events()]

    assert len(events) == 1
    evt = events[0]
    assert evt.kind == TriggerKind.CHAT_MESSAGE
    assert evt.project_id == "p1"
    assert evt.payload == {"text": "hello"}
    assert evt.reply_to is channel
    assert evt.source == "web_console"


@pytest.mark.asyncio
async def test_sse_channel_streams_events_then_closes() -> None:
    channel = SseOutputChannel()
    await channel.emit(OutputEvent(
        kind=OutputEventKind.MESSAGE,
        payload={"content": "hi"},
    ))
    await channel.emit(OutputEvent(
        kind=OutputEventKind.ACTION_CALLED,
        payload={"name": "foo", "args": {"a": 1}},
    ))
    await channel.close()

    chunks = [c async for c in channel.stream()]

    assert len(chunks) == 2
    first = json.loads(chunks[0].removeprefix("data: ").strip())
    assert first == {"kind": "message", "payload": {"content": "hi"}}
    second = json.loads(chunks[1].removeprefix("data: ").strip())
    assert second["kind"] == "action_called"
    assert second["payload"]["name"] == "foo"
