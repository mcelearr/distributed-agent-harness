"""
``python -m distributed_agent_harness.chat`` — default CLI entry point.

Spawns a chat REPL bound to a specific WorldEnvironment class and project.
Defaults to the data protection example so you can test end-to-end out of
the box::

    export MISTRAL_API_KEY=...
    python -m distributed_agent_harness.chat --project=acme-gdpr-2025

To use a different WorldEnvironment, pass ``--world dotted.path.to.Class``.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from typing import Type

from .adapters import InMemoryNamespace
from .eventlog import InMemoryEventLog
from .interfaces import CliChat
from .llm_providers import MistralProvider
from .runtime import AgentRuntime
from .world import BaseWorldEnvironment


def _import_class(dotted: str) -> Type[BaseWorldEnvironment]:
    """Import ``module.path:ClassName`` or ``module.path.ClassName``."""
    if ":" in dotted:
        module_path, class_name = dotted.split(":", 1)
    else:
        module_path, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseWorldEnvironment):
        raise TypeError(f"{dotted} is not a BaseWorldEnvironment subclass")
    return cls


async def _main_async(args: argparse.Namespace) -> int:
    world_class = _import_class(args.world)

    if not os.environ.get("MISTRAL_API_KEY"):
        print(
            "error: MISTRAL_API_KEY is not set.\n"
            "  Get one at https://console.mistral.ai → API keys, then either:\n"
            "    - put it in a .env file and run:\n"
            "        uv run --env-file .env python -m distributed_agent_harness.chat\n"
            "    - or export it directly: export MISTRAL_API_KEY=...",
            file=sys.stderr,
        )
        return 2

    llm = MistralProvider(model=args.model)
    runtime = AgentRuntime(
        world_class=world_class,
        namespace=InMemoryNamespace(),
        eventlog=InMemoryEventLog(),
        llm=llm,
        max_iterations=args.max_iterations,
    )

    cli = CliChat(project_id=args.project, show_actions=not args.quiet)
    async for event in cli.events():
        try:
            await runtime.handle(event)
        except Exception as exc:  # noqa: BLE001
            print(f"[runtime error] {exc}", file=sys.stderr)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m distributed_agent_harness.chat",
        description="Interactive chat REPL bound to a WorldEnvironment.",
    )
    parser.add_argument(
        "--project",
        default="demo-project",
        help="project_id (namespace key for the WorldEnvironment instance)",
    )
    parser.add_argument(
        "--world",
        default="examples.use_cases.data_protection.world:DataProtectionWorldEnvironment",
        help="Dotted path to a BaseWorldEnvironment subclass",
    )
    parser.add_argument(
        "--model",
        default=MistralProvider.DEFAULT_MODEL,
        help=f"Mistral model name (default: {MistralProvider.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="Maximum LLM loop iterations per user message",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide intermediate tool calls — only show final assistant messages",
    )

    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
