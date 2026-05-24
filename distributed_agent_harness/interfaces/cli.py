"""
CliChat — terminal REPL interface.

Implements both ``TriggerSource`` (reads stdin lines) and ``OutputChannel``
(prints events to stdout). One instance handles a full interactive session
with a single ``project_id``.

Designed as the zero-dependency default for manual testing.
"""
from __future__ import annotations

import asyncio
import sys
from typing import AsyncIterator

from ..transport import (
    OutputChannel,
    OutputEvent,
    OutputEventKind,
    TriggerEvent,
    TriggerKind,
    TriggerSource,
)


# ANSI escape codes — bare minimum, no dependencies.
class _C:
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"


class CliChat(TriggerSource, OutputChannel):
    """
    Terminal chat session bound to one project.

    Usage::

        cli = CliChat(project_id="acme-gdpr-2025")
        async for event in cli.events():
            await runtime.handle(event)

    Type ``exit`` or ``quit`` (or send EOF) to end the session.
    """

    EXIT_COMMANDS = {"exit", "quit", ":q", ":wq"}

    def __init__(
        self,
        project_id: str,
        prompt: str = "you ▸ ",
        show_actions: bool = True,
        colour: bool = True,
    ) -> None:
        self.project_id = project_id
        self.prompt = prompt
        self.show_actions = show_actions
        self.colour = colour and sys.stdout.isatty()

    # ----------------------------------------------------------------------- #
    # TriggerSource                                                            #
    # ----------------------------------------------------------------------- #

    async def events(self) -> AsyncIterator[TriggerEvent]:
        loop = asyncio.get_running_loop()
        self._print_banner()
        while True:
            try:
                # input() is blocking — push to a thread to keep the loop free
                line = await loop.run_in_executor(None, self._read_line)
            except (EOFError, KeyboardInterrupt):
                self._println(self._style("\n[session ended]", _C.DIM))
                return

            text = line.strip()
            if not text:
                continue
            if text.lower() in self.EXIT_COMMANDS:
                self._println(self._style("[session ended]", _C.DIM))
                return

            yield TriggerEvent(
                source="cli",
                kind=TriggerKind.CHAT_MESSAGE,
                payload={"text": text},
                project_id=self.project_id,
                reply_to=self,
            )

    def _read_line(self) -> str:
        return input(self.prompt)

    def _print_banner(self) -> None:
        banner = (
            f"{_C.BOLD}Distributed Agent Harness — chat session{_C.RESET}\n"
            f"  project: {_C.CYAN}{self.project_id}{_C.RESET}\n"
            f"  type {_C.YELLOW}exit{_C.RESET} or send EOF to quit.\n"
        )
        self._println(banner if self.colour else _strip_ansi(banner))

    # ----------------------------------------------------------------------- #
    # OutputChannel                                                            #
    # ----------------------------------------------------------------------- #

    async def emit(self, event: OutputEvent) -> None:
        if event.kind == OutputEventKind.MESSAGE:
            content = event.payload.get("content", "")
            self._println(self._style("agent ▸ ", _C.GREEN) + content)
        elif event.kind == OutputEventKind.ACTION_CALLED and self.show_actions:
            name = event.payload.get("name", "?")
            args = event.payload.get("args", {})
            self._println(self._style(f"  ↪ {name}({_fmt_args(args)})", _C.DIM))
        elif event.kind == OutputEventKind.ACTION_RESULT and self.show_actions:
            name = event.payload.get("name", "?")
            if "error" in event.payload:
                msg = self._style(f"  ✗ {name} → {event.payload['error']}", _C.RED)
            else:
                result = event.payload.get("result", "")
                short = (result[:120] + "…") if len(result) > 120 else result
                msg = self._style(f"  ✓ {name} → {short}", _C.DIM)
            self._println(msg)
        elif event.kind == OutputEventKind.ERROR:
            self._println(self._style(
                f"[error] {event.payload.get('error', 'unknown')}", _C.RED
            ))
        elif event.kind == OutputEventKind.FINAL:
            self._println("")  # blank line between turns

    # ----------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ----------------------------------------------------------------------- #

    def _style(self, text: str, code: str) -> str:
        if not self.colour:
            return text
        return f"{code}{text}{_C.RESET}"

    @staticmethod
    def _println(text: str) -> None:
        print(text, flush=True)


def _fmt_args(args: dict) -> str:
    """Render args inline, truncating long values."""
    parts = []
    for k, v in args.items():
        sv = repr(v)
        if len(sv) > 40:
            sv = sv[:37] + "…"
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)
