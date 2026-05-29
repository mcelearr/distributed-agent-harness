"""
PromptBuilder — lifts WorldEnvironment context into the LLM's system prompt.

The system prompt is composed in a fixed order:

    1. Project Summary    — the always-on "card view" from ``render_summary()``
    2. Available Actions  — every ``@action`` whose ``show_when`` is None or
                             returns True against the current state + trigger
    3. Recent Activity    — tail of the event log, to help the agent avoid loops
    4. Current State      — exact JSON state, for precise tool-call arguments

Actions whose ``show_when`` returns False (or raises) are hidden from the
prompt entirely AND uncallable at runtime — direct invocation raises
``ActionNotAvailable``.
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
    """Builds the agent's system prompt from a WorldEnvironment class.

    For each ``@action`` method it surfaces:

    - **Signature** — parameter names, types, and return type
    - **Docstring** — the human-readable description
    - **Source body** — the actual Python implementation (opt-in via
      ``include_source=True``) so the LLM can reason about second-order effects

    For each running ``world`` instance it also surfaces:

    - The contents of ``summary.md`` (rendered by ``world.render_summary()``)
    - The last N lines of ``event_log.md`` (configurable)
    - The exact JSON state

    Visibility: pass a ``world`` (and optionally the triggering ``event``)
    to ``build_actions_prompt`` / ``build_full_prompt`` so each action's
    ``show_when`` can be evaluated. Without a world, ``show_when`` is not
    evaluated and every action is shown (backward-compat mode for tests and
    introspection).
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
        """Return a Markdown-formatted description of all visible ``@action`` methods.

        When ``world`` is provided, each action's ``show_when`` predicate is
        evaluated against the current state + trigger; actions whose
        predicate returns False (or raises) are hidden.

        When ``world`` is None, predicates are not evaluated and every action
        is shown (used by tests and introspection callers).
        """
        all_actions = self._world_class.get_actions()
        if not all_actions:
            return "## Available Actions\n\n_(none defined)_"

        if world is not None:
            all_actions = {
                name: method
                for name, method in all_actions.items()
                if _evaluate_predicate(
                    getattr(method, "_show_when", None), world.state, event,
                )
            }
            if not all_actions:
                return "## Available Actions\n\n_(none currently available)_"

        sections = ["## Available Actions\n"]
        for name, method in sorted(all_actions.items()):
            sections.append(self._format_action_full(name, method))
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

    def build_subagents_prompt(
        self,
        subagents: list[Any] | None,
        state: Any = None,
        event: "TriggerEvent | None" = None,
    ) -> str | None:
        """Return the "Available Subagents" section.

        Lists each registered subagent as ``consult_<name>(message, session_id)``.
        Filtered by ``show_when`` against the current state when both ``state``
        and ``event`` are provided. Returns None when there are no visible
        subagents — the runtime then skips the section entirely.
        """
        if not subagents:
            return None

        visible: list[Any] = []
        for sub in subagents:
            predicate = getattr(sub, "show_when", None)
            if predicate is None or state is None:
                visible.append(sub)
                continue
            try:
                if bool(predicate(state, event)):
                    visible.append(sub)
            except Exception:  # noqa: BLE001
                log.exception(
                    "subagent %r show_when raised; hiding from prompt",
                    getattr(sub, "name", "?"),
                )

        if not visible:
            return None

        lines = ["## Available Subagents (external specialists)\n"]
        for sub in sorted(visible, key=lambda s: s.name):
            lines.append(self._format_subagent(sub))
        return "\n\n".join(lines)

    def build_search_event_log_prompt(self) -> str:
        """Return the always-on "Searching the Event Log" section.

        Tells the LLM that ``search_event_log`` exists and when to reach for it.
        """
        return (
            "## Searching the Event Log\n\n"
            "Recent Activity above shows only the last few entries. For older "
            "history — finding a prior `consult_<subagent>` session_id, "
            "confirming whether a particular action has already been taken, "
            "or building a picture of subagent activity on this project — "
            "call `search_event_log` with filters such as "
            "`action_name_glob='consult_*'`."
        )

    def build_namespace_browse_prompt(self) -> str:
        """Return the always-on "Exploring the Namespace" section.

        Tells the LLM the project namespace is a virtual filesystem and the
        three baseline read tools exist (``ls`` / ``read`` / ``grep``).
        """
        return (
            "## Exploring the Namespace\n\n"
            "The project namespace is the agent's virtual filesystem. The "
            "Project Summary and Current State above are lifted excerpts; "
            "the full set of documents — `event_log.md`, `audit.jsonl`, "
            "subagent artefacts under `<project>/artefacts/`, and anything "
            "else stored under the project — is accessible through three "
            "read-only meta-tools:\n\n"
            "- `ls(path)` — list documents and subdirectories at a path.\n"
            "- `read(path, offset?, limit?)` — read a text document.\n"
            "- `grep(pattern, path?, glob?, ignore_case?)` — search content "
            "across documents.\n\n"
            "These tools do not change project state. To change state, call "
            "an `@action`. To consult an external specialist, use one of "
            "the `consult_<name>` subagent tools."
        )

    def build_full_prompt(
        self,
        world: "BaseWorldEnvironment",
        event: "TriggerEvent | None" = None,
        subagents: list[Any] | None = None,
    ) -> str:
        """Return the complete system prompt in the standard section order.

        Sections, in order:
        1. Project Summary
        2. Available Actions
        3. Available Subagents (omitted when none registered / visible)
        4. Searching the Event Log
        5. Recent Activity
        6. Current State
        """
        sections = [
            self.build_summary_prompt(world),
            self.build_actions_prompt(world, event=event),
        ]
        subagents_section = self.build_subagents_prompt(
            subagents, state=world.state, event=event,
        )
        if subagents_section is not None:
            sections.append(subagents_section)
        sections.extend([
            self.build_namespace_browse_prompt(),
            self.build_search_event_log_prompt(),
            self.build_recent_activity_prompt(world),
            self.build_state_prompt(world),
        ])
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _format_subagent(sub: Any) -> str:
        """Render one subagent entry."""
        skills_text = ""
        skills = getattr(getattr(sub, "card", None), "skills", []) or []
        if skills:
            skill_names = [s.name for s in skills if getattr(s, "name", None)]
            if skill_names:
                skills_text = f"\nSkills: {', '.join(skill_names)}"
        provider = getattr(getattr(sub, "card", None), "provider", None)
        provider_text = f"\nProvider: {provider}" if provider else ""
        return (
            f"### `consult_{sub.name}(message: str, session_id: str | None = None) -> str`\n"
            f"{sub.description}"
            f"{skills_text}"
            f"{provider_text}\n"
            "`session_id` round-trips through this tool's response — pass it "
            "back to continue the same A2A context."
        )

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


def _evaluate_predicate(predicate: Any, state: Any, event: Any) -> bool:
    """Evaluate ``show_when``, treating None as True and exceptions as False.

    A predicate that raises is treated as False so that a buggy predicate
    hides its action rather than crashing prompt construction.
    """
    if predicate is None:
        return True
    try:
        return bool(predicate(state, event))
    except Exception:  # noqa: BLE001
        log.exception("Action show_when raised; treating as False")
        return False


def _tail_bullets(markdown: str, n: int) -> list[str]:
    """Return the last *n* bullet lines (``- ...``) from a markdown document."""
    bullets = [line for line in markdown.split("\n") if line.startswith("- ")]
    return bullets[-n:]
