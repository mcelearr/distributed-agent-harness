"""
PromptBuilder — lifts WorldEnvironment actions into LLM context.

Rather than exposing a minimal MCP-style interface (name + one-line description),
the PromptBuilder can surface the full method signature, docstring, and optionally
the source code body so the LLM understands *how* each action mutates world state.
"""
from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .world import BaseWorldEnvironment


class PromptBuilder:
    """
    Builds the agent's system prompt from a WorldEnvironment class.

    For each ``@action`` method it can include:

    - **Signature** — parameter names, types, and return type (always included)
    - **Docstring** — the human-readable description (always included if present)
    - **Source body** — the actual Python implementation extracted at import time
      via ``inspect.getsource`` (opt-in via ``include_source=True``)

    Including source bodies gives the LLM a richer understanding of second-order
    effects: it can see *how* a call will change ``self.state``, not just what the
    method is named. This is particularly valuable for complex domain logic.

    Example output (``include_source=False``)::

        ## Available Actions

        ### `submit_pitch(client_name: str, contact_email: str) -> Client`
        Submit a pitch to win a data protection engagement.

        ### `report_breach(description: str, discovered_at: datetime) -> DataBreach`
        Report a personal data breach internally.
    """

    def __init__(
        self,
        world_class: Type["BaseWorldEnvironment"],
        include_source: bool = False,
    ) -> None:
        self._world_class = world_class
        self._include_source = include_source

    # ----------------------------------------------------------------------- #
    # Public API                                                               #
    # ----------------------------------------------------------------------- #

    def build_actions_prompt(self) -> str:
        """Return a Markdown-formatted description of all ``@action`` methods."""
        actions = self._world_class.get_actions()
        if not actions:
            return "No agent actions defined."

        sections = ["## Available Actions\n"]
        for name, method in sorted(actions.items()):
            sections.append(self._format_action(name, method))

        return "\n\n".join(sections)

    def build_state_prompt(self, world: "BaseWorldEnvironment") -> str:
        """Return a Markdown-formatted snapshot of the current world state."""
        state_json = world.state.model_dump_json(indent=2)
        return f"## Current World State\n\n```json\n{state_json}\n```"

    def build_full_prompt(self, world: "BaseWorldEnvironment") -> str:
        """Return the complete system prompt: available actions + current state."""
        return "\n\n---\n\n".join([
            self.build_actions_prompt(),
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
