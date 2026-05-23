"""InProcessLock — threading.Lock-based concurrency handler for single-machine use."""
from __future__ import annotations

import threading

from ..concurrency import ConcurrencyHandler


class InProcessLock(ConcurrencyHandler):
    """
    Single-machine concurrency handler using ``threading.Lock``.

    Safe for multiple threads within the same Python process. Each unique
    ``resource_id`` gets its own lock, so concurrent projects do not block
    each other.

    Not safe across multiple processes or machines — use ``RedisLock``
    (planned) for distributed deployments.

    Suitable for:
    - Development and testing
    - Single-machine deployments where agents run as threads
    """

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def _get_lock(self, resource_id: str) -> threading.Lock:
        with self._registry_lock:
            if resource_id not in self._locks:
                self._locks[resource_id] = threading.Lock()
            return self._locks[resource_id]

    def acquire_lock(self, resource_id: str, timeout: float = 30.0) -> None:
        lock = self._get_lock(resource_id)
        acquired = lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Could not acquire lock on '{resource_id}' within {timeout}s. "
                f"Another agent or thread may be holding it."
            )

    def release_lock(self, resource_id: str) -> None:
        lock = self._get_lock(resource_id)
        try:
            lock.release()
        except RuntimeError:
            pass  # Already released — safe to ignore
