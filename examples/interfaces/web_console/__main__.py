"""``python -m examples.interfaces.web_console`` — launch the console.

Usage::

    export MISTRAL_API_KEY=...
    python -m examples.interfaces.web_console \\
        --worlds examples/interfaces/web_console/worlds.toml

Then visit http://localhost:8765 in a browser.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .registry import WorldRegistry
from .server import create_app, require_api_key


DEFAULT_WORLDS = Path(__file__).parent / "worlds.toml"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m examples.interfaces.web_console",
        description="Web testing console for the Distributed Agent Harness.",
    )
    parser.add_argument(
        "--worlds",
        type=Path,
        default=DEFAULT_WORLDS,
        help=f"Path to a worlds.toml registry (default: {DEFAULT_WORLDS})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model",
        default=None,
        help="Mistral model name (default: provider default)",
    )
    args = parser.parse_args()

    require_api_key()
    registry = WorldRegistry.from_toml(args.worlds)
    app = create_app(registry, model=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
