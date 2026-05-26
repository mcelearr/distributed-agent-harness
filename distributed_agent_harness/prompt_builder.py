"""
PromptBuilder — lifts WorldEnvironment context into the LLM's system prompt.

The system prompt is composed in a fixed order:

    1. Project Summary    — the always-on "card view" from ``render_summary()``
    2. Available Actions  — partitioned into Active and Latent tiers based on
                             each action's optional ``precondition`` / ``relevance``
                             predicates evaluated against current state and trigger
    3. Recent Activity    — tail of the event log, to help the agent avoid loops
    4. Current State      — exact JSON state, for precise tool-call arguments

Action partitioning (per ``@action`` predicates):

- **Hidden**  — ``precondition`` is set and returns False. Dropped entirely.
- **Active**  — ``precondition`` (if set) is True AND ``relevance`` (if set or
                 absent) is True. Full signature + docstring + optional source.
- **Latent**  — ``precondition`` (if set) is True AND ``relevance`` is set but
                 returns False. Manifest line only (name + first-line docstring).
"""
from __future__ import annotations

import inspect
import logging
import textwrap
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from .transport import TriggerEvent
    from .world import BaseWorldEnvironment

log = logging.getLogger(__name__)


# Default number of event-log entries to surface in the system prompt.
DEFAULT_RECENT_ACTIVITY_COUNT = 12


class PromptBuilder:
    """
    Builds the agent's system prompt from a WorldEnvironment class.

    For each ``@action`` method it surfaces:

    - **Signature** — parameter names, types, and return type
    - **Docstring** — the human-readable description
    - **Source body** — the actual Python implementation (opt-in via
      ``include_source=True``) so the LLM can reason about second-order effects

    For each running ``world`` instance it also surfaces:

    - The contents of ``summary.md`` (rendered by ``world.render_summary()``)
    - The last N lines of ``event_log.md`` (configurable)
    - The exact JSON state

    Action partitioning by predicates: pass a ``world`` (and optionally the
    triggering ``event``) to ``build_actions_prompt`` / ``build_full_prompt``
    so each action's ``precondition`` and ``relevance`` can be evaluated.
    Without a world, predicates are not evaluated and every action is shown
    in full as Active (backward-compat mode for tests / introspection).
    """

    def __init__(
        self,
        world_class: Type["BaseWorldEnvironment"],
        include_source: bool = False,
        recent_activity_count: int = DEFAULT_RECENT_ACTIVITY_COUNT,
    ) -> None:
        self._world_class = world_class
        self._include_source = include_source
        self._recent_activity_count = recent_activity_count

    # ----------------------------------------------------------------------- #
    # Public sections                                                          #
    # ----------------------------------------------------------------------- #

    def build_summary_prompt(self, world: "BaseWorldEnvironment") -> str:
        """Return ``summary.md`` from the namespace, or fall back to a live render."""
        raw = world._namespace.read_doc(world._summary_path())
        if not raw:
            # No persisted summary yet (e.g. before the first @action) — render live.
            try:
                raw = world.render_summary()
            except Exception:  # noqa: BLE001
                raw = world._default_summary()
        return f"## Project Summary\n\n{raw.strip()}"

    def build_actions_prompt(
        self,
        world: "BaseWorldEnvironment | None" = None,
        event: "TriggerEvent | None" = None,
    ) -> str:
        """
        Return a Markdown-formatted description of all ``@action`` methods.

        When ``world`` is provided, actions are partitioned by their predicates
        into Active (full detail) and Latent (manifest line) sections; actions
        whose ``precondition`` returns False are hidden.

        When ``world`` is None, predicates are not evaluated and every action
        is shown in full (used by tests and introspection callers).
        """
        all_actions = self._world_class.get_actions()
        if not all_actions:
            return "## Available Actions\n\n_(none defined)_"

        if world is None:
            # Backward-compat: no world means we can't evaluate predicates.
            sections = ["## Available Actions\n"]
            for name, method in sorted(all_actions.items()):
                sections.append(self._format_action_full(name, method))
            return "\n\n".join(sections)

        active, latent = _partition_actions(all_actions, world.state, event)

        if not active and not latent:
            return "## Available Actions\n\n_(none currently available)_"

        sections = ["## Available Actions"]

        if active:
            sections.append(
                "\n### Active — relevant for the current state\n"
            )
            for name, method in active:
                sections.append(self._format_action_full(name, method))

        if latent:
            sections.append(
                "\n### Latent — defined but not currently in scope\n\n"
                "_These actions exist but are not relevant right now. "
                "Only call them if you have a specific reason._\n"
            )
            for name, method in latent:
                sections.append(self._format_action_manifest(name, method))

        return "\n\n".join(sections)

    def build_recent_activity_prompt(self, world: "BaseWorldEnvironment") -> str:
        """Return the last N entries from ``event_log.md`` as a markdown section."""
        raw = world._namespace.read_doc(world._event_log_path())
        bullets = _tail_bullets(raw, self._recent_activity_count) if raw else []

        body = "\n".join(bullets) if bullets else "_(no actions taken on this project yet)_"
        return (
            f"## Recent Activity (last {self._recent_activity_count})\n\n"
            f"_Review this before acting — do not repeat actions you have just taken._\n\n"
            f"{body}"
        )

    def build_state_prompt(self, world: "BaseWorldEnvironment") -> str:
        """Return a Markdown-formatted JSON snapshot of the current world state."""
        state_json = world.state.model_dump_json(indent=2)
        return f"## Current State (exact values)\n\n```json\n{state_json}\n```"

    def build_full_prompt(
        self,
        world: "BaseWorldEnvironment",
        event: "TriggerEvent | None" = None,
    ) -> str:
        """Return the complete system prompt in the standard section order."""
        return "\n\n---\n\n".join([
            self.build_summary_prompt(world),
            self.build_actions_prompt(world, event=event),
            self.build_recent_activity_prompt(world),
            self.build_state_prompt(world),
        ])

    # ----------------------------------------------------------------------- #
    # Internals — formatting one action                                        #
    # ----------------------------------------------------------------------- #

    def _format_action_full(self, name: str, method: object) -> str:
        """Full detail: signature header, docstring, optional source body."""
        display_sig = _format_signature(name, method)
        lines = [f"### `{display_sig}`"]

        doc = inspect.getdoc(method)  # type: ignore[arg-type]
        if doc:
            lines.append(doc)

        if self._include_source:
            source = getattr(method, "_source", None)
            if source:
                lines.append(f"```python\n{textwrap.dedent(source)}\n```")

        return "\n".join(lines)

    @staticmethod
    def _format_action_manifest(name: str, method: object) -> str:
        """Compact manifest line: name(params) — first line of docstring."""
        sig = inspect.signature(method)  # type: ignore[arg-type]
        param_names = [k for k in sig.parameters if k != "self"]
        doc = inspect.getdoc(method)  # type: ignore[arg-type]
        first_line = doc.split("\n", 1)[0] if doc else ""
        suffix = f" — {first_line}" if first_line else ""
        return f"- `{name}({', '.join(param_names)})`{suffix}"


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

def _format_signature(name: str, method: object) -> str:
    """Render a function signature for the prompt header."""
    sig = inspect.signature(method)  # type: ignore[arg-type]
    params = [p for k, p in sig.parameters.items() if k != "self"]
    param_str = ", ".join(str(p) for p in params)

    ret = sig.return_annotation
    if ret is inspect.Parameter.empty:
        ret_str = ""
    elif hasattr(ret, "__name__"):
        ret_str = f" -> {ret.__name__}"
    else:
        ret_str = f" -> {ret}"

    return f"{name}({param_str}){ret_str}"


def _evaluate_predicate(
    predicate: Any,
    state: Any,
    event: Any,
) -> bool:
    """
    Evaluate a predicate, treating exceptions as False (defensive — a buggy
    predicate should never crash the prompt builder).
    """
    if predicate is None:
        return True
    try:
        return bool(predicate(state, event))
    except Exception:  # noqa: BLE001
        log.exception("Action predicate raised; treating as False")
        return False


def _partition_actions(
    all_actions: dict[str, Any],
    state: Any,
    event: Any,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """
    Split actions into ``(active, latent)`` based on their predicates.

    Hidden actions (precondition is set and returns False) are dropped.
    """
    active: list[tuple[str, Any]] = []
    latent: list[tuple[str, Any]] = []

    for name, method in sorted(all_actions.items()):
        precondition = getattr(method, "_precondition", None)
        relevance = getattr(method, "_relevance", None)

        # Hidden: precondition is set and false
        if precondition is not None and not _evaluate_predicate(precondition, state, event):
            continue

        # Latent: relevance is set and false
        if relevance is not None and not _evaluate_predicate(relevance, state, event):
            latent.append((name, method))
        else:
            active.append((name, method))

    return active, latent


def _tail_bullets(markdown: str, n: int) -> list[str]:
    """Return the last *n* bullet lines (``- ...``) from a markdown document."""
    bullets = [line for line in markdown.split("\n") if line.startswith("- ")]
    return bullets[-n:]
