"""
Transport layer — pluggable trigger sources and output channels.

The harness separates *what triggers an agent run* from *where the agent's
output goes*. This lets us support:

- **Interactive use cases** (chat) where one adapter implements both
  ``TriggerSource`` and ``OutputChannel`` so input and output share a session.
- **Event-driven use cases** (webhook, cron) where a pure ``TriggerSource``
  fires events with no streaming back-channel.

A ``TriggerEvent`` may carry an optional ``reply_to`` ``OutputChannel`` —
present for interactive triggers, absent for pure event-driven ones.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from .identity import AgentIdentity


# --------------------------------------------------------------------------- #
# Triggers                                                                     #
# --------------------------------------------------------------------------- #

class TriggerKind(str, Enum):
    CHAT_MESSAGE = "chat_message"   # user sent a message in a chat UI
    WEBHOOK = "webhook"             # external system fired a webhook
    SCHEDULED = "scheduled"         # cron-like timer fired
    SYSTEM = "system"               # internal trigger (e.g. self-scheduled)


@dataclass
class TriggerEvent:
    """An event that may cause an agent run to start.

    ``id`` is the request-id every downstream ``Event`` references via
    ``Event.trigger_id`` — letting an investigator filter ``audit.jsonl``
    down to a single triggering request chain. Defaults to a fresh uuid
    hex so callers don't have to set it explicitly; pass an explicit value
    when correlating with an upstream system's own trace id.

    ``identity`` is the principal that fired this trigger. When ``None``
    the runtime substitutes ``AgentIdentity.anonymous_agent()`` so the
    no-config use case stays one-liner.
    """
    source: str                           # e.g. "cli", "http", "sharepoint"
    kind: TriggerKind
    payload: dict[str, Any]                # free-form per-source data
    project_id: str                        # which WorldEnvironment to act on
    reply_to: "OutputChannel | None" = None  # optional streaming back-channel
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    identity: "AgentIdentity | None" = None


class TriggerSource(ABC):
    """
    Source of TriggerEvents.

    Implementations yield events as an async iterator. The runtime consumes
    this iterator and dispatches each event to the AgentRuntime.

    Examples:
    - ``CliChat``: yields one event per line of stdin
    - ``HttpServer``: yields one event per POST request
    - ``WebhookListener``: yields one event per inbound webhook
    """

    @abstractmethod
    def events(self) -> AsyncIterator[TriggerEvent]:
        """Async iterator of trigger events."""

    async def close(self) -> None:
        """Optional cleanup. Default: no-op."""
        return None


# --------------------------------------------------------------------------- #
# Output channels                                                              #
# --------------------------------------------------------------------------- #

class OutputEventKind(str, Enum):
    THINKING = "thinking"             # LLM reasoning / non-final text
    ACTION_CALLED = "action_called"   # a WorldEnvironment @action was invoked
    ACTION_RESULT = "action_result"   # the @action returned (or errored)
    MESSAGE = "message"               # assistant message (final output)
    FINAL = "final"                   # run complete; no more events
    ERROR = "error"                   # something failed


@dataclass
class OutputEvent:
    """An event emitted during an agent run."""
    kind: OutputEventKind
    payload: dict[str, Any] = field(default_factory=dict)


class OutputChannel(ABC):
    """
    Destination for OutputEvents emitted during an agent run.

    Examples:
    - ``CliChat``: prints each event to stdout
    - ``SseChannel``: pushes each event as an SSE message
    - ``LogChannel``: appends each event to a logfile
    """

    @abstractmethod
    async def emit(self, event: OutputEvent) -> None:
        """Send one event to wherever this channel writes."""

    async def close(self) -> None:
        """Optional cleanup. Default: no-op."""
        return None
