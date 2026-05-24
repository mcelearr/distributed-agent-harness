"""
Tests for the transport layer — TriggerEvent / OutputEvent typing and CliChat.
"""
from __future__ import annotations

import asyncio
import io
from typing import AsyncIterator

import pytest

from distributed_agent_harness.interfaces.cli import CliChat, _strip_ansi, _fmt_args
from distributed_agent_harness.transport import (
    OutputChannel,
    OutputEvent,
    OutputEventKind,
    TriggerEvent,
    TriggerKind,
    TriggerSource,
)


class TestTriggerEvent:
    def test_construct_with_reply_to_none(self) -> None:
        event = TriggerEvent(
            source="webhook",
            kind=TriggerKind.WEBHOOK,
            payload={"foo": "bar"},
            project_id="p1",
        )
        assert event.reply_to is None
        assert event.kind == TriggerKind.WEBHOOK

    def test_chat_message_has_reply_to(self) -> None:
        channel = CliChat(project_id="p1")
        event = TriggerEvent(
            source="cli",
            kind=TriggerKind.CHAT_MESSAGE,
            payload={"text": "hi"},
            project_id="p1",
            reply_to=channel,
        )
        assert event.reply_to is channel


class TestOutputEvent:
    def test_default_payload_is_empty_dict(self) -> None:
        event = OutputEvent(kind=OutputEventKind.FINAL)
        assert event.payload == {}

    def test_with_payload(self) -> None:
        event = OutputEvent(
            kind=OutputEventKind.MESSAGE,
            payload={"content": "hello"},
        )
        assert event.payload["content"] == "hello"


# --------------------------------------------------------------------------- #
# CliChat                                                                      #
# --------------------------------------------------------------------------- #

class TestCliChatChannel:
    @pytest.mark.asyncio
    async def test_emit_message(self, capsys: pytest.CaptureFixture) -> None:
        cli = CliChat(project_id="p", colour=False)
        await cli.emit(OutputEvent(
            kind=OutputEventKind.MESSAGE,
            payload={"content": "hello user"},
        ))
        captured = capsys.readouterr()
        assert "hello user" in captured.out

    @pytest.mark.asyncio
    async def test_emit_action_called(self, capsys: pytest.CaptureFixture) -> None:
        cli = CliChat(project_id="p", colour=False)
        await cli.emit(OutputEvent(
            kind=OutputEventKind.ACTION_CALLED,
            payload={"name": "add_item", "args": {"text": "buy milk"}},
        ))
        captured = capsys.readouterr()
        assert "add_item" in captured.out
        assert "buy milk" in captured.out

    @pytest.mark.asyncio
    async def test_emit_action_result_error(self, capsys: pytest.CaptureFixture) -> None:
        cli = CliChat(project_id="p", colour=False)
        await cli.emit(OutputEvent(
            kind=OutputEventKind.ACTION_RESULT,
            payload={"name": "x", "error": "boom"},
        ))
        captured = capsys.readouterr()
        assert "boom" in captured.out

    @pytest.mark.asyncio
    async def test_show_actions_false_suppresses_tool_events(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        cli = CliChat(project_id="p", colour=False, show_actions=False)
        await cli.emit(OutputEvent(
            kind=OutputEventKind.ACTION_CALLED,
            payload={"name": "secret", "args": {}},
        ))
        captured = capsys.readouterr()
        assert "secret" not in captured.out


class TestCliChatTrigger:
    @pytest.mark.asyncio
    async def test_yields_events_for_input(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cli = CliChat(project_id="my-proj", colour=False)
        lines = iter(["hello", "world", "exit"])

        def fake_input() -> str:
            return next(lines)

        monkeypatch.setattr(cli, "_read_line", fake_input)

        collected: list[TriggerEvent] = []
        async for event in cli.events():
            collected.append(event)

        assert len(collected) == 2
        assert collected[0].payload["text"] == "hello"
        assert collected[0].project_id == "my-proj"
        assert collected[0].reply_to is cli
        assert collected[1].payload["text"] == "world"

    @pytest.mark.asyncio
    async def test_eof_terminates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cli = CliChat(project_id="p", colour=False)

        def raise_eof() -> str:
            raise EOFError()

        monkeypatch.setattr(cli, "_read_line", raise_eof)

        collected: list[TriggerEvent] = []
        async for event in cli.events():
            collected.append(event)
        assert collected == []

    @pytest.mark.asyncio
    async def test_skips_blank_lines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cli = CliChat(project_id="p", colour=False)
        lines = iter(["", "   ", "real input", "quit"])
        monkeypatch.setattr(cli, "_read_line", lambda: next(lines))

        collected: list[TriggerEvent] = []
        async for event in cli.events():
            collected.append(event)
        assert len(collected) == 1
        assert collected[0].payload["text"] == "real input"


class TestHelpers:
    def test_strip_ansi(self) -> None:
        assert _strip_ansi("\033[31mred\033[0m text") == "red text"

    def test_fmt_args_short(self) -> None:
        assert _fmt_args({"x": 1, "y": "hi"}) == "x=1, y='hi'"

    def test_fmt_args_truncates_long_values(self) -> None:
        long = "a" * 100
        out = _fmt_args({"x": long})
        assert "…" in out
        assert len(out) < 60
