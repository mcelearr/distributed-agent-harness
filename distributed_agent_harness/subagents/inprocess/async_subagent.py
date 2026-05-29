"""
InProcessSubagent marker class + AsyncSubagent.

``InProcessSubagent`` exists purely as a typing marker for "this subagent
runs inside the harness process" — there's no protocol it imposes beyond
``SubagentClient`` itself. ``isinstance(sub, InProcessSubagent)`` lets
operators tell at a glance which subagents are local-only (relevant for
policies like "only allow A2A subagents in production").

``AsyncSubagent`` wraps an async handler. Each ``consult()`` invokes the
handler on the current event loop, spawn-per-call: one task per call,
nothing pooled. The per-call timeout is enforced with
``asyncio.timeout`` (Python 3.11+, which we already require).
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable

from ..base import (
    ProgressCallback,
    SubagentClient,
    SubagentResponse,
    SubagentTimeout,
)

if TYPE_CHECKING:
    from ...world import Predicate


# --------------------------------------------------------------------------- #
# Marker base class                                                            #
# --------------------------------------------------------------------------- #

class InProcessSubagent(SubagentClient):
    """Marker for subagents that run inside the harness process.

    Subclasses implement ``consult`` normally; this base adds no behaviour.
    The marker is used by operator policy code (hooks, allowlists) to tell
    in-process subagents apart from A2A ones.
    """


# --------------------------------------------------------------------------- #
# AsyncSubagent                                                                #
# --------------------------------------------------------------------------- #

#: A handler that does the actual work. Receives the user message, the
#: optional session_id, and an optional progress callback the handler can
#: invoke with text deltas as it makes progress (mirrors A2A's `working`
#: events, surfaced as THINKING events on the runtime's OutputChannel).
AsyncHandler = Callable[
    [str, "str | None", "ProgressCallback | None"],
    Awaitable[SubagentResponse],
]


class AsyncSubagent(InProcessSubagent):
    """Subagent backed by an async function on the same event loop.

    Spawn-per-call: each ``consult()`` invokes the handler fresh; there
    is no pool, no queue, no concurrency cap. The runtime's per-call
    timeout wraps the handler with ``asyncio.timeout``; a slow handler
    raises ``SubagentTimeout``.

    Usage::

        async def classify(message: str, session_id: str | None, on_progress):
            # ... do work ...
            return SubagentResponse(status="completed", content="...")

        runtime.subagents.register(AsyncSubagent(
            name="classifier",
            description="Classifies incoming docs by type.",
            handler=classify,
        ))
    """

    def __init__(
        self,
        name: str,
        description: str,
        handler: AsyncHandler,
        show_when: "Predicate | None" = None,
    ) -> None:
        self.name = name
        self.description = description
        self.show_when = show_when
        self._handler = handler

    async def consult(
        self,
        message: str,
        session_id: str | None = None,
        timeout: float = 60.0,
        on_progress: "ProgressCallback | None" = None,
    ) -> SubagentResponse:
        try:
            async with asyncio.timeout(timeout):
                return await self._handler(message, session_id, on_progress)
        except asyncio.TimeoutError as exc:
            raise SubagentTimeout(
                f"AsyncSubagent {self.name!r} did not return within {timeout}s"
            ) from exc
