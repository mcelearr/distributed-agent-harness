"""
PromptBuilder — lifts WorldEnvironment context into the LLM's system prompt.

The system prompt is composed in a fixed order:

    1. Project Summary    — the always-on "card view" from ``render_summary()``
    2. Available Actions  — @action signatures, docstrings, optional source
    3. Recent Activity    — tail of the event log, to help the agent avoid loops
    4. Current State      — exact JSON state, for precise tool-call arguments

The summary is the most important orienting context; recent activity is the
loop-prevention safety net; the JSON state is the precise machine reference.
"""
from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .world import BaseWorldEnvironment


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

    def build_actions_prompt(self) -> str:
        """Return a Markdown-formatted description of all ``@action`` methods."""
        actions = self._world_class.get_actions()
        if not actions:
            return "## Available Actions\n\n_(none defined)_"

        sections = ["## Available Actions\n"]
        for name, method in sorted(actions.items()):
            sections.append(self._format_action(name, method))

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

    def build_full_prompt(self, world: "BaseWorldEnvironment") -> str:
        """Return the complete system prompt in the standard section order."""
        return "\n\n---\n\n".join([
            self.build_summary_prompt(world),
            self.build_actions_prompt(),
            self.build_recent_activity_prompt(world),
            self.build_state_prompt(world),
        ])

    # ----------------------------------------------------------------------- #
    # Internals                                                                #
    # ----------------------------------------------------------------------- #

    def _format_action(self, name: str, method: object) -> str:
        sig = inspect.signature(method)  # type: ignore[arg-type]

        # Drop 'self' from the displayed signature
        params = [p for k, p in sig.parameters.items() if k != "self"]
        param_str = ", ".join(str(p) for p in params)

        ret = sig.return_annotation
        if ret is inspect.Parameter.empty:
            ret_str = ""
        elif hasattr(ret, "__name__"):
            ret_str = f" -> {ret.__name__}"
        else:
            ret_str = f" -> {ret}"

        display_sig = f"{name}({param_str}){ret_str}"
        lines = [f"### `{display_sig}`"]

        doc = inspect.getdoc(method)  # type: ignore[arg-type]
        if doc:
            lines.append(doc)

        if self._include_source:
            source = getattr(method, "_source", None)
            if source:
                lines.append(f"```python\n{textwrap.dedent(source)}\n```")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

def _tail_bullets(markdown: str, n: int) -> list[str]:
    """Return the last *n* bullet lines (``- ...``) from a markdown document."""
    bullets = [line for line in markdown.split("\n") if line.startswith("- ")]
    return bullets[-n:]
