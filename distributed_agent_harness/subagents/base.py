"""
SubagentClient ABC + the dataclasses every subagent backend shares.

A subagent is an external specialist surfaced to the LLM as
``consult_<name>(message, session_id=None)``. The harness is agnostic to
how the subagent runs — over A2A, in a subprocess, on a message bus — as
long as it implements ``SubagentClient.consult``.

Session continuity is opaque: the subagent returns a ``session_id`` (A2A
``contextId`` in practice); the LLM round-trips that id through its
conversation history or via ``search_event_log`` to continue the same
context on a follow-up consult. The harness stores no session state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:
    from ..world import Predicate


# --------------------------------------------------------------------------- #
# AgentCard — A2A discovery descriptor                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Skill:
    """One declared capability on an ``AgentCard``."""
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """The A2A unit of discovery and registration.

    A subset of the full A2A spec — we model the fields we actually use
    (name, description, url, skills, provider). Extra fields a registry
    returns can be carried in ``metadata`` without subclassing.
    """
    name: str
    description: str
    url: str
    skills: list[Skill] = field(default_factory=list)
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Response shapes                                                              #
# --------------------------------------------------------------------------- #

SubagentStatus = Literal["completed", "input-required", "failed"]


@dataclass
class SubagentResponse:
    """Outcome of one ``consult``.

    Three statuses match A2A's terminal task states verbatim:

    - ``completed`` — the subagent finished; ``content`` is the answer.
    - ``input-required`` — the subagent paused mid-task and needs more
      info; ``content`` carries the clarification question. The caller
      (typically the LLM) decides whether to follow up with another
      ``consult`` using the same ``session_id`` or abandon.
    - ``failed`` — the subagent errored; ``content`` is the reason.
    """
    status: SubagentStatus
    content: str
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Exceptions                                                                   #
# --------------------------------------------------------------------------- #

class SubagentTimeout(Exception):
    """Raised when the SSE stream stalls past the per-call timeout.

    A subagent must emit at least one byte (a keep-alive comment, a
    ``working`` event, or a terminal event) within ``timeout`` seconds.
    Otherwise the harness cancels the stream and raises this — surfaced to
    the LLM as a TOOL error.
    """


# --------------------------------------------------------------------------- #
# Client interface                                                             #
# --------------------------------------------------------------------------- #

# A progress callback the runtime injects to fan ``working`` text deltas
# out as ``OutputEventKind.THINKING`` events on the active OutputChannel.
ProgressCallback = Callable[[str], Awaitable[None]]


class SubagentClient(ABC):
    """Pluggable subagent transport.

    Subclasses implement ``consult``; everything else (prompt surfacing,
    audit logging, hook firing, event-log writes) is handled by the
    runtime.
    """

    name: str
    description: str
    show_when: "Predicate | None"

    @abstractmethod
    async def consult(
        self,
        message: str,
        session_id: str | None = None,
        timeout: float = 60.0,
        on_progress: ProgressCallback | None = None,
    ) -> SubagentResponse:
        """Send ``message`` to the subagent and resolve to a terminal response.

        Parameters
        ----------
        message:
            The user-facing prompt to send.
        session_id:
            Optional A2A ``contextId`` — when set, the subagent should
            resume the same conversation context.
        timeout:
            Per-call hard timeout in seconds. The connection must show
            inbound activity within this window or ``SubagentTimeout`` is
            raised. Defaults to 60 s.
        on_progress:
            Optional async callback invoked with each incremental text
            delta the subagent emits while ``working``. Used by the
            runtime to forward thinking to the OutputChannel.
        """


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

class SubagentRegistry:
    """In-process registry of subagents available to one ``AgentRuntime``.

    Subagents are looked up by ``name``. The runtime exposes one tool per
    registered subagent as ``consult_<name>``.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, SubagentClient] = {}

    def register(self, client: SubagentClient) -> None:
        if client.name in self._by_name:
            raise ValueError(f"Subagent {client.name!r} is already registered")
        self._by_name[client.name] = client

    def unregister(self, name: str) -> None:
        self._by_name.pop(name, None)

    def get(self, name: str) -> SubagentClient | None:
        return self._by_name.get(name)

    def list(self) -> list[SubagentClient]:
        return list(self._by_name.values())

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)
