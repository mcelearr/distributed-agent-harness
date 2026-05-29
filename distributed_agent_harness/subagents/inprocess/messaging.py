"""
MessageBus + InMemoryMessageBus + MessagingSubagent + MessagingSubagentWorker.

The ``MessageBus`` ABC separates the subagent layer from any concrete
broker so a future ``KafkaMessageBus`` can drop in without code changes.
The shipped ``InMemoryMessageBus`` runs entirely on asyncio — useful for
tests, local dev, and single-process deployments. It implements
**request/reply** semantics with one subscriber per topic; that's the
shape the subagent protocol needs.

``MessagingSubagent`` is the client side: publishes a payload on its
topic, awaits the response. ``MessagingSubagentWorker`` is a thin
adapter that exposes an async handler as a subscriber on a topic — so a
complete subagent loop (client + worker) runs in-process without any
external infrastructure during tests.
"""
from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..base import (
    Artefact,
    ProgressCallback,
    SubagentResponse,
    SubagentStatus,
    SubagentTimeout,
)
from .async_subagent import InProcessSubagent

if TYPE_CHECKING:
    from ...world import Predicate


# --------------------------------------------------------------------------- #
# Exceptions                                                                   #
# --------------------------------------------------------------------------- #

class NoSubscriberError(Exception):
    """Raised when a ``request`` is sent to a topic with no subscriber."""


# --------------------------------------------------------------------------- #
# Subscription handle                                                          #
# --------------------------------------------------------------------------- #

class Subscription:
    """Cancellation handle for a ``MessageBus.subscribe`` call.

    Usable as a context manager so workers can be wired up cleanly in
    tests::

        async with bus.subscribe(topic, handler) as sub:
            ...
        # sub.unsubscribe() called on exit
    """

    def __init__(self, bus: "MessageBus", topic: str) -> None:
        self._bus = bus
        self._topic = topic
        self._unsubscribed = False

    def unsubscribe(self) -> None:
        if self._unsubscribed:
            return
        self._unsubscribed = True
        self._bus._on_unsubscribe(self._topic)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.unsubscribe()


# --------------------------------------------------------------------------- #
# MessageBus ABC                                                               #
# --------------------------------------------------------------------------- #

#: Handler signature for ``subscribe``. Receives the request payload,
#: returns the reply payload. ``None`` is treated as an empty reply.
BusHandler = Callable[[dict], Awaitable["dict | None"]]


class MessageBus(ABC):
    """Pluggable request/reply transport.

    The subagent protocol uses one subscriber per topic — the worker that
    services that topic — and the client side blocks on ``request``
    until a reply arrives or the timeout expires. Fan-out and pub/sub
    aren't needed for v1.
    """

    @abstractmethod
    async def request(
        self,
        topic: str,
        payload: dict,
        timeout: float = 60.0,
    ) -> dict:
        """Send ``payload`` on ``topic`` and await the subscriber's reply.

        Raises ``NoSubscriberError`` if no one is subscribed and
        ``SubagentTimeout`` if the subscriber doesn't reply within
        ``timeout`` seconds.
        """

    @abstractmethod
    def subscribe(
        self,
        topic: str,
        handler: BusHandler,
    ) -> Subscription:
        """Register ``handler`` as the subscriber on ``topic``.

        At most one subscriber per topic in v1.
        """

    # Internal hook used by Subscription.unsubscribe — concrete buses
    # override to remove the registration.
    def _on_unsubscribe(self, topic: str) -> None: ...


# --------------------------------------------------------------------------- #
# In-memory implementation                                                     #
# --------------------------------------------------------------------------- #

class InMemoryMessageBus(MessageBus):
    """Asyncio-backed bus for tests and single-process deployments.

    One handler per topic; ``request`` invokes the handler directly and
    awaits its return value. No queues, no pub/sub fan-out — just clean
    in-process RPC with the ``MessageBus`` interface so the messaging
    subagent path can swap in Kafka later without touching subagent code.

    A correlation id is generated on every ``request`` and attached to
    the payload as ``__correlation_id__`` so workers can echo it back in
    diagnostics (the in-memory bus does not use it for routing — there's
    nothing to route between).
    """

    def __init__(self) -> None:
        self._handlers: dict[str, BusHandler] = {}

    async def request(
        self,
        topic: str,
        payload: dict,
        timeout: float = 60.0,
    ) -> dict:
        handler = self._handlers.get(topic)
        if handler is None:
            raise NoSubscriberError(f"No subscriber on topic {topic!r}")
        correlated = dict(payload)
        correlated.setdefault("__correlation_id__", uuid.uuid4().hex)
        try:
            async with asyncio.timeout(timeout):
                reply = await handler(correlated)
        except asyncio.TimeoutError as exc:
            raise SubagentTimeout(
                f"No reply on topic {topic!r} within {timeout}s"
            ) from exc
        return reply or {}

    def subscribe(self, topic: str, handler: BusHandler) -> Subscription:
        if topic in self._handlers:
            raise ValueError(
                f"Topic {topic!r} already has a subscriber on this bus"
            )
        self._handlers[topic] = handler
        return Subscription(self, topic)

    def _on_unsubscribe(self, topic: str) -> None:
        self._handlers.pop(topic, None)


# --------------------------------------------------------------------------- #
# MessagingSubagent                                                            #
# --------------------------------------------------------------------------- #

class MessagingSubagent(InProcessSubagent):
    """Subagent that consults via a ``MessageBus`` request/reply cycle.

    Payload sent on the request topic::

        {
            "message": <str>,
            "session_id": <str | None>,
        }

    Expected reply shape::

        {
            "status": "completed" | "input-required" | "failed",
            "content": <str>,
            "session_id": <str | None>,
            "metadata": <dict>,
            "artefacts": [
                {"name": <str>, "content": <bytes>, "mime": <str | None>,
                 "description": <str>},
                ...
            ],
        }

    Missing fields fall back to sensible defaults (``status='completed'``,
    empty ``content`` / ``metadata`` / ``artefacts``).
    """

    def __init__(
        self,
        name: str,
        description: str,
        bus: MessageBus,
        topic: str,
        show_when: "Predicate | None" = None,
    ) -> None:
        self.name = name
        self.description = description
        self.show_when = show_when
        self._bus = bus
        self._topic = topic

    async def consult(
        self,
        message: str,
        session_id: str | None = None,
        timeout: float = 60.0,
        on_progress: "ProgressCallback | None" = None,
    ) -> SubagentResponse:
        # on_progress is not supported over the bus in v1: the request/
        # reply contract has no streaming channel. A future enhancement
        # could carry progress events on a dedicated topic.
        payload = {
            "message": message,
            "session_id": session_id,
        }
        reply = await self._bus.request(self._topic, payload, timeout=timeout)
        return _response_from_reply(reply)


def _response_from_reply(reply: dict) -> SubagentResponse:
    status_raw = reply.get("status", "completed")
    status: SubagentStatus = (
        status_raw if status_raw in ("completed", "input-required", "failed")
        else "completed"
    )
    artefacts = []
    for a in reply.get("artefacts", []) or []:
        if not isinstance(a, dict):
            continue
        content = a.get("content", b"")
        if isinstance(content, bytearray):
            content = bytes(content)
        if not isinstance(content, bytes):
            continue
        artefacts.append(Artefact(
            name=str(a.get("name", "artefact.bin")),
            content=content,
            mime=a.get("mime") if isinstance(a.get("mime"), str) else None,
            description=str(a.get("description", "")),
        ))
    return SubagentResponse(
        status=status,
        content=str(reply.get("content", "")),
        session_id=(
            reply.get("session_id")
            if isinstance(reply.get("session_id"), str) else None
        ),
        metadata=reply.get("metadata", {}) if isinstance(reply.get("metadata"), dict) else {},
        artefacts=artefacts,
    )


# --------------------------------------------------------------------------- #
# MessagingSubagentWorker                                                      #
# --------------------------------------------------------------------------- #

#: Worker handler signature — mirrors the subagent's handler shape but
#: returns the structured ``SubagentResponse`` directly. The worker
#: serialises that into the reply dict shape expected by the bus.
WorkerHandler = Callable[
    [str, "str | None"],
    Awaitable[SubagentResponse],
]


class MessagingSubagentWorker:
    """Adapter that exposes an async handler as a subscriber on a topic.

    Usage::

        async def serve(message, session_id):
            return SubagentResponse(status="completed", content="...")

        worker = MessagingSubagentWorker(bus, topic="classifier.requests",
                                          handler=serve)
        # ... runtime hands LLM the subagent; worker handles requests ...
        worker.stop()  # idempotent
    """

    def __init__(
        self,
        bus: MessageBus,
        topic: str,
        handler: WorkerHandler,
    ) -> None:
        self._handler = handler
        self._subscription = bus.subscribe(topic, self._on_message)

    async def _on_message(self, payload: dict) -> dict:
        message = payload.get("message", "")
        session_id = payload.get("session_id")
        response = await self._handler(
            str(message),
            session_id if isinstance(session_id, str) else None,
        )
        return _reply_from_response(response)

    def stop(self) -> None:
        """Unsubscribe — safe to call more than once."""
        self._subscription.unsubscribe()


def _reply_from_response(response: SubagentResponse) -> dict:
    return {
        "status": response.status,
        "content": response.content,
        "session_id": response.session_id,
        "metadata": response.metadata,
        "artefacts": [
            {
                "name": a.name,
                "content": a.content,
                "mime": a.mime,
                "description": a.description,
            }
            for a in response.artefacts
        ],
    }
