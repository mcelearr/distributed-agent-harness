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
"""
from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Union


# --------------------------------------------------------------------------- #
# Event                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Event:
    """One auditable mutation against a project's state.

    Events are totally ordered per project. The ``offset`` field is assigned
    by the log on successful append and is unique within a project.
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
        attribute is set to ``expected_offset`` (its position in the log).

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
