"""
Event sourcing primitives — Event, EventLog ABC, InMemoryEventLog.

The event log is the source of truth for an event-sourced project. Every
``@action`` invocation produces one ``Event``. Appends are optimistic:
each append carries ``expected_offset``; if another writer has already
appended since the caller's last read, the append is rejected with a
``Conflict`` containing the intervening events. The caller (typically
the harness runtime) then decides whether to retry, re-plan, or abandon.

The in-memory implementation is the reference and test backend. A Kafka
implementation is planned (one topic; partition key = ``project_id``).

Tamper-evident hash chain
-------------------------
Each ``Event`` carries ``prev_hash`` (the previous event's ``hash``) and
its own ``hash``, computed over a canonical JSON form of every field
except ``hash`` itself. The log fills both fields during ``append`` so
callers cannot construct an event that lies about its predecessor.
``EventLog.verify_chain(project_id)`` re-derives every hash and confirms
the chain — any post-hoc edit to the stored log shows up as a
``ChainBreak`` with the bad offset. The chain is content-addressable and
deployment-agnostic: in-memory, Kafka, or any future backend gets
integrity verification "for free" via the same helper.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from .identity import AgentIdentity


# --------------------------------------------------------------------------- #
# Event                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Event:
    """One auditable mutation against a project's state.

    Events are totally ordered per project. The ``offset`` field is assigned
    by the log on successful append and is unique within a project.

    Identity / traceability
    -----------------------
    ``actor`` is the legacy free-form string (still grep-friendly). When the
    runtime constructs the event from an ``AgentIdentity`` the identity is
    *also* recorded on ``identity`` and ``actor`` is set to
    ``identity.label`` for backward compatibility.

    ``trigger_id`` links every event back to the originating
    ``TriggerEvent.id`` — letting an investigator filter ``audit.jsonl``
    down to "every action taken in response to webhook X."

    Integrity
    ---------
    ``prev_hash`` / ``hash`` form a tamper-evident chain. The log assigns
    both during ``append``; callers should leave them ``None`` on
    construction.
    """
    project_id: str
    action_name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    actor: str = "agent"
    result_summary: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    offset: int | None = None  # Assigned by the log on append
    identity: "AgentIdentity | None" = None
    trigger_id: str | None = None
    prev_hash: str | None = None  # Assigned by the log on append
    hash: str | None = None       # Assigned by the log on append

    def __post_init__(self) -> None:
        # When an identity is supplied alongside the default actor string,
        # derive actor from identity.label so downstream consumers (event
        # log search, audit.jsonl) report the structured form consistently.
        # Explicit non-default ``actor=`` overrides this.
        if self.identity is not None and self.actor == "agent":
            self.actor = self.identity.label


# --------------------------------------------------------------------------- #
# Hash chain                                                                   #
# --------------------------------------------------------------------------- #

def compute_event_hash(event: Event) -> str:
    """Return the hex SHA-256 of a canonical encoding of ``event``.

    Canonical form: every field except ``hash`` itself, sorted, with
    deterministic encodings for datetimes (ISO 8601) and identity
    (principal + instance_id + pubkey_fingerprint + sorted roles). The
    encoding is JSON with ``sort_keys=True`` and ``separators=(",", ":")``
    so two runs over identical content produce identical bytes.
    """
    payload: dict[str, Any] = {
        "project_id": event.project_id,
        "action_name": event.action_name,
        "args": _canonicalise(event.args),
        "kwargs": _canonicalise(event.kwargs),
        "actor": event.actor,
        "result_summary": event.result_summary,
        "id": event.id,
        "timestamp": event.timestamp.isoformat(),
        "offset": event.offset,
        "trigger_id": event.trigger_id,
        "prev_hash": event.prev_hash,
        "identity": _identity_canonical(event.identity),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _identity_canonical(identity: "AgentIdentity | None") -> dict[str, Any] | None:
    """Stable dict form of an AgentIdentity for hash inclusion."""
    if identity is None:
        return None
    return {
        "principal": identity.principal,
        "instance_id": identity.instance_id,
        "pubkey_fingerprint": identity.pubkey_fingerprint,
        "roles": sorted(identity.roles),
    }


def _canonicalise(value: Any) -> Any:
    """Recursively coerce a value into JSON-friendly, deterministic shape."""
    if isinstance(value, dict):
        return {str(k): _canonicalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class ChainBreak(Exception):
    """Raised by ``verify_chain`` when an event's hash or link is invalid.

    Attributes
    ----------
    offset:
        Position of the first event whose chain link is wrong.
    reason:
        Human-readable cause: ``"prev_hash mismatch"``, ``"hash mismatch"``,
        ``"missing hash"``.
    """

    def __init__(self, offset: int, reason: str) -> None:
        self.offset = offset
        self.reason = reason
        super().__init__(f"chain broken at offset {offset}: {reason}")


# --------------------------------------------------------------------------- #
# Append results                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Appended:
    """Returned on successful append. ``new_offset`` is the post-append count."""
    new_offset: int


@dataclass
class Conflict:
    """Returned when the supplied ``expected_offset`` is stale.

    ``current_offset`` is the log's offset after the intervening appends.
    ``intervening_events`` are the events appended by other actors between
    the caller's last seen offset and ``current_offset``.
    """
    current_offset: int
    intervening_events: list[Event]


AppendResult = Union[Appended, Conflict]


# --------------------------------------------------------------------------- #
# Abstract event log                                                           #
# --------------------------------------------------------------------------- #

class EventLog(ABC):
    """Append-only, per-project, optimistic-CAS event log.

    Implementations must guarantee:

    - Events within a project are totally ordered by ``offset`` (0-indexed).
    - ``append`` is atomic: either the supplied event becomes the next event
      in the log, or it is rejected without side effect.
    - ``append`` rejects when ``expected_offset != current_offset`` and
      returns the intervening events so the caller can resolve the conflict.
    - On successful append, ``event.prev_hash`` is set to the previous
      event's ``hash`` (or ``None`` at offset 0) and ``event.hash`` is set
      to ``compute_event_hash(event)``.
    """

    @abstractmethod
    async def current_offset(self, project_id: str) -> int:
        """Return the count of events currently logged for ``project_id``."""

    @abstractmethod
    async def append(
        self,
        project_id: str,
        event: Event,
        expected_offset: int,
    ) -> AppendResult:
        """Append ``event`` if ``expected_offset`` matches the current offset.

        On success, returns ``Appended(new_offset)``. The event's ``offset``
        attribute is set to ``expected_offset`` (its position in the log),
        and ``prev_hash`` + ``hash`` are filled in.

        On conflict, returns ``Conflict(current_offset, intervening_events)``
        without appending.
        """

    @abstractmethod
    async def read_events(
        self,
        project_id: str,
        from_offset: int = 0,
    ) -> list[Event]:
        """Return events at offsets ``[from_offset, current_offset)``."""

    async def verify_chain(self, project_id: str) -> None:
        """Re-derive every event hash and confirm the chain.

        Default implementation reads the full project log and validates
        each event's ``prev_hash`` link plus the integrity of its own
        ``hash``. Raises ``ChainBreak`` at the first inconsistency. Returns
        normally when the chain is intact (empty logs trivially pass).

        Backends with native verification (e.g. Merkle trees) may override
        — but they must surface a ``ChainBreak`` on any inconsistency so
        callers can rely on the contract.
        """
        events = await self.read_events(project_id, from_offset=0)
        expected_prev: str | None = None
        for i, event in enumerate(events):
            if event.hash is None:
                raise ChainBreak(i, "missing hash")
            if event.prev_hash != expected_prev:
                raise ChainBreak(i, "prev_hash mismatch")
            recomputed = compute_event_hash(event)
            if recomputed != event.hash:
                raise ChainBreak(i, "hash mismatch")
            expected_prev = event.hash


# --------------------------------------------------------------------------- #
# In-memory implementation                                                     #
# --------------------------------------------------------------------------- #

class InMemoryEventLog(EventLog):
    """Thread-safe in-process event log for tests and local development.

    Backed by a dict keyed on ``project_id``. Each append takes an internal
    lock just long enough to validate ``expected_offset`` and append — the
    lock is not held during user code, so this is *not* equivalent to the
    old pessimistic-lock model. Concurrent writers will still collide on
    CAS; the loser sees ``Conflict``.
    """

    def __init__(self) -> None:
        self._logs: dict[str, list[Event]] = {}
        self._lock = threading.Lock()

    async def current_offset(self, project_id: str) -> int:
        with self._lock:
            return len(self._logs.get(project_id, ()))

    async def append(
        self,
        project_id: str,
        event: Event,
        expected_offset: int,
    ) -> AppendResult:
        with self._lock:
            log = self._logs.setdefault(project_id, [])
            if expected_offset != len(log):
                return Conflict(
                    current_offset=len(log),
                    intervening_events=list(log[expected_offset:]),
                )
            event.offset = len(log)
            event.project_id = project_id
            event.prev_hash = log[-1].hash if log else None
            event.hash = compute_event_hash(event)
            log.append(event)
            return Appended(new_offset=len(log))

    async def read_events(
        self,
        project_id: str,
        from_offset: int = 0,
    ) -> list[Event]:
        with self._lock:
            log = self._logs.get(project_id, [])
            return list(log[from_offset:])
