"""Abstract base class for the Concurrency Handler."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ConcurrencyHandler(ABC):
    """
    Pluggable distributed coordination layer.

    Ensures that concurrent method calls from multiple agents or humans
    do not corrupt the shared Project Namespace. Every state-mutating
    action must follow the cycle:

        acquire_lock → read latest state → execute → write state → release_lock

    The handler abstracts the locking mechanism so that teams can choose the
    right approach for their deployment topology without changing agent code.

    Interface contract
    ------------------
    acquire_lock(resource_id, timeout)  Block until exclusive lock is held.
    release_lock(resource_id)           Release the lock unconditionally.

    Planned implementations
    -----------------------
    - InProcessLock   built-in, threading.Lock, single-machine
    - RedisLock       planned, distributed lock across pods in a cluster
    - KafkaHandler    planned, full event-sourcing via message log
    """

    @abstractmethod
    def acquire_lock(self, resource_id: str, timeout: float = 30.0) -> None:
        """
        Acquire an exclusive lock on *resource_id*.

        Blocks until the lock is available or *timeout* seconds have elapsed.
        Raises ``TimeoutError`` if the lock cannot be acquired in time.
        """

    @abstractmethod
    def release_lock(self, resource_id: str) -> None:
        """Release the lock on *resource_id*. Safe to call even if already released."""
