"""
BaseWorldEnvironment — the core of the Distributed Agent Harness.

Subclass this to define a domain-specific world model. The base class
handles all persistence, locking, and audit logging transparently.
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Type

from pydantic import BaseModel

from .concurrency import ConcurrencyHandler
from .namespace import NamespaceAdapter

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# @action decorator                                                             #
# --------------------------------------------------------------------------- #

def action(method: Any) -> Any:
    """
    Decorator that marks a WorldEnvironment method as an auditable agent action.

    Every decorated method is automatically wrapped in the full transaction cycle:

        1. Acquire an exclusive lock on the project namespace
        2. Read the latest state from the namespace (never operates on stale data)
        3. Execute the method body
        4. Flush the updated state back to the namespace
        5. Append an entry to the append-only audit log
        6. Release the lock (even on failure)

    The original source code is preserved on the wrapper as ``._source`` so
    the PromptBuilder can surface it to the LLM.

    Usage::

        class MyWorld(BaseWorldEnvironment):
            State = MyState

            @action
            def do_something(self, value: str) -> str:
                \"\"\"A clear docstring describing what this action does.\"\"\"
                self.state.things.append(value)
                return value
    """
    # Capture source before wrapping — inspect cannot retrieve source of a closure
    try:
        source = inspect.getsource(method)
    except OSError:
        source = ""

    @functools.wraps(method)
    def wrapper(self: "BaseWorldEnvironment", *args: Any, **kwargs: Any) -> Any:
        resource_id = self._project_id
        self._concurrency.acquire_lock(resource_id)
        try:
            # Always read the latest persisted state before executing
            self._hydrate()
            result = method(self, *args, **kwargs)
            self._flush()
            self._write_audit(method.__name__, args, kwargs, result=result)
            return result
        except Exception as exc:
            self._write_audit(method.__name__, args, kwargs, error=str(exc))
            raise
        finally:
            self._concurrency.release_lock(resource_id)

    wrapper._is_action = True   # type: ignore[attr-defined]
    wrapper._source = source    # type: ignore[attr-defined]
    return wrapper


# --------------------------------------------------------------------------- #
# BaseWorldEnvironment                                                          #
# --------------------------------------------------------------------------- #

class BaseWorldEnvironment:
    """
    Abstract base class for all World Environments.

    A World Environment is a Python object that represents the complete,
    authoritative state of a project. Agents interact with the project by
    calling @action methods on this object; humans interact by reading and
    writing the underlying namespace documents directly.

    How to use
    ----------
    1. Define a Pydantic model for your domain state::

        class MyState(BaseModel):
            items: list[str] = []

    2. Subclass BaseWorldEnvironment, set ``State``, and write @action methods::

        class MyWorld(BaseWorldEnvironment):
            State = MyState

            @action
            def add_item(self, text: str) -> str:
                \"\"\"Add a new item.\"\"\"
                self.state.items.append(text)
                return text

    3. Instantiate with pluggable namespace and concurrency backends::

        world = MyWorld(
            project_id="my-project",
            namespace=InMemoryNamespace(),
            concurrency=InProcessLock(),
        )

    The base class handles:
    - Loading state from the namespace on construction
    - Wrapping every @action call in acquire / hydrate / execute / flush / release
    - Appending to an append-only audit log on every action
    """

    #: Subclasses MUST set this to a Pydantic BaseModel class.
    State: ClassVar[Type[BaseModel]]

    _STATE_DOC: ClassVar[str] = "state.json"
    _AUDIT_DOC: ClassVar[str] = "audit.jsonl"

    def __init__(
        self,
        project_id: str,
        namespace: NamespaceAdapter,
        concurrency: ConcurrencyHandler,
    ) -> None:
        if not hasattr(self.__class__, "State"):
            raise TypeError(
                f"{type(self).__name__} must define a 'State' class attribute "
                f"pointing to a Pydantic BaseModel subclass."
            )
        self._project_id = project_id
        self._namespace = namespace
        self._concurrency = concurrency
        # Initialise with defaults; _hydrate will overwrite if a persisted state exists
        self.state: BaseModel = self.__class__.State()
        self._hydrate()

    # ----------------------------------------------------------------------- #
    # Lifecycle (called inside @action wrappers)                               #
    # ----------------------------------------------------------------------- #

    def _state_path(self) -> str:
        return f"{self._project_id}/{self._STATE_DOC}"

    def _audit_path(self) -> str:
        return f"{self._project_id}/{self._AUDIT_DOC}"

    def _hydrate(self) -> None:
        """Deserialise the latest persisted state from the namespace into ``self.state``."""
        raw = self._namespace.read_doc(self._state_path())
        if raw:
            self.state = self.__class__.State.model_validate_json(raw)

    def _flush(self) -> None:
        """Serialise ``self.state`` and write it to the namespace."""
        self._namespace.write_doc(
            self._state_path(),
            self.state.model_dump_json(indent=2),
        )

    def _write_audit(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Append one JSON-lines entry to the project's audit log."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": self._project_id,
            "method": method_name,
            "args": [repr(a) for a in args],
            "kwargs": {k: repr(v) for k, v in kwargs.items()},
        }
        if error is not None:
            entry["error"] = error
        else:
            entry["result_type"] = type(result).__name__

        existing = self._namespace.read_doc(self._audit_path()) or ""
        self._namespace.write_doc(
            self._audit_path(),
            existing + json.dumps(entry) + "\n",
        )

    # ----------------------------------------------------------------------- #
    # Introspection (used by PromptBuilder)                                    #
    # ----------------------------------------------------------------------- #

    @classmethod
    def get_actions(cls) -> dict[str, Any]:
        """Return all ``@action``-decorated callables defined on this class."""
        return {
            name: member
            for name, member in inspect.getmembers(cls, predicate=callable)
            if getattr(member, "_is_action", False)
        }
