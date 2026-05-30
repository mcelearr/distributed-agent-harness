"""
FastAPI app exposing the harness over HTTP + SSE.

Endpoints:

- ``GET  /``                 — bundled chat UI (single HTML page)
- ``GET  /api/worlds``       — list of WorldEnvironments from `worlds.toml`
- ``POST /api/chat/{world}`` — submit one user message; returns an SSE
                                stream of OutputEvents for that one turn
- ``GET  /api/ls/{world}``   — list documents in the world's namespace
- ``GET  /api/read/{world}`` — read one document by path

Per-world state (an in-memory `NamespaceAdapter` and `InMemoryEventLog`)
is created lazily on first use and cached for the lifetime of the server,
so a demo session is conversation-continuous.

This module is intentionally short: anything novel lives in
`transports.py`. This file is just wiring.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.eventlog import InMemoryEventLog
from distributed_agent_harness.llm_providers import MistralProvider
from distributed_agent_harness.runtime import AgentRuntime

from .registry import WorldEntry, WorldRegistry
from .transports import HttpTriggerSource, SseOutputChannel


WEB_DIR = Path(__file__).parent / "web"


@dataclass
class _WorldState:
    """Per-world singletons reused across chat turns."""
    namespace: InMemoryNamespace
    eventlog: InMemoryEventLog
    runtime: AgentRuntime


class ChatRequest(BaseModel):
    text: str


def create_app(registry: WorldRegistry, *, model: str | None = None) -> FastAPI:
    """Build the FastAPI app bound to a `WorldRegistry`."""
    app = FastAPI(title="Distributed Agent Harness — Web Console")
    state: dict[str, _WorldState] = {}

    def _world_state(entry: WorldEntry) -> _WorldState:
        if entry.id not in state:
            namespace = InMemoryNamespace()
            eventlog = InMemoryEventLog()
            llm = MistralProvider(model=model) if model else MistralProvider()
            runtime = AgentRuntime(
                world_class=entry.world_class,
                namespace=namespace,
                eventlog=eventlog,
                llm=llm,
            )
            state[entry.id] = _WorldState(namespace=namespace, eventlog=eventlog, runtime=runtime)
        return state[entry.id]

    # --------------------------------------------------------------- routes

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/app.js")
    async def appjs() -> FileResponse:
        return FileResponse(WEB_DIR / "app.js")

    @app.get("/api/worlds")
    async def list_worlds() -> list[dict[str, str]]:
        return [
            {"id": e.id, "name": e.name, "project_id": e.project_id}
            for e in registry.list()
        ]

    @app.post("/api/chat/{world_id}")
    async def chat(world_id: str, body: ChatRequest) -> StreamingResponse:
        try:
            entry = registry.get(world_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        ws = _world_state(entry)
        channel = SseOutputChannel()
        trigger = HttpTriggerSource(
            project_id=entry.project_id,
            text=body.text,
            reply_to=channel,
        )

        async def runner() -> None:
            try:
                async for event in trigger.events():
                    await ws.runtime.handle(event)
            finally:
                await channel.close()

        import asyncio
        asyncio.create_task(runner())
        return StreamingResponse(channel.stream(), media_type="text/event-stream")

    @app.get("/api/ls/{world_id}")
    async def ls_docs(world_id: str) -> list[str]:
        entry = _entry_or_404(registry, world_id)
        ws = _world_state(entry)
        return ws.namespace.list_docs()

    @app.get("/api/read/{world_id}")
    async def read_doc(world_id: str, path: str) -> dict[str, Any]:
        entry = _entry_or_404(registry, world_id)
        ws = _world_state(entry)
        content = ws.namespace.read_doc(path)
        if content is None:
            raise HTTPException(status_code=404, detail=f"Not found: {path}")
        return {"path": path, "content": content}

    return app


def _entry_or_404(registry: WorldRegistry, world_id: str) -> WorldEntry:
    try:
        return registry.get(world_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def require_api_key() -> None:
    """Sanity-check that an LLM key is present before binding the port."""
    if not os.environ.get("MISTRAL_API_KEY"):
        raise SystemExit(
            "MISTRAL_API_KEY is not set.\n"
            "  - If you have a .env file at the repo root, run with:\n"
            "      uv run --env-file .env python -m examples.interfaces.web_console\n"
            "  - Otherwise create one: cp .env.example .env, paste in your key.\n"
            "  - Or export the variable directly: export MISTRAL_API_KEY=..."
        )
