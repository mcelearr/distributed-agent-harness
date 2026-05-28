"""
Conflict resolution — agent-as-rebaser.

When two actors try to write to a project concurrently, one succeeds and the
other receives a ``Conflict`` from the event log. This module turns that
mechanical conflict into a decision the agent (or operator policy) can
reason about: ``Continue``, ``Recover``, or ``Abandon``.

A cheap *structural pre-check* runs before any LLM round-trip: if the
planned action's declared read/write fields are disjoint from the writes of
the intervening events, the conflict is materially harmless and the runtime
auto-``Continue``s without consulting the resolver.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Union

from .eventlog import Event

if TYPE_CHECKING:
    from .llm import LLMProvider, Message, ToolCall


# --------------------------------------------------------------------------- #
# Decisions                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class Continue:
    """Retry the same tool call against the refreshed state."""


@dataclass
class Recover:
    """Rebuild the system prompt with fresh state and let the LLM re-plan.

    The conversation is preserved; only the assistant's next turn is
    regenerated. A synthetic system note listing the intervening events is
    prepended so the LLM understands why it's being asked to re-plan.
    """


@dataclass
class Abandon:
    """Stop the run and report ``reason`` back to the user."""
    reason: str


Decision = Union[Continue, Recover, Abandon]


# --------------------------------------------------------------------------- #
# Exception raised by the @action wrapper on a stale-CAS append                #
# --------------------------------------------------------------------------- #

class ConcurrentUpdate(Exception):
    """Raised by the @action wrapper when its append loses the CAS race.

    Carries everything the runtime needs to surface a conflict to the
    resolver: the intervening events and the new log offset.
    """

    def __init__(
        self,
        action_name: str,
        last_seen_offset: int,
        current_offset: int,
        intervening_events: list[Event],
    ) -> None:
        self.action_name = action_name
        self.last_seen_offset = last_seen_offset
        self.current_offset = current_offset
        self.intervening_events = intervening_events
        super().__init__(
            f"{action_name}: log advanced from {last_seen_offset} to "
            f"{current_offset} during this call ("
            f"{len(intervening_events)} intervening event(s))"
        )


# --------------------------------------------------------------------------- #
# Conflict context                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class ConflictContext:
    """Everything a ``ConflictResolver`` needs to decide an outcome."""
    project_id: str
    last_seen_offset: int
    current_offset: int
    intervening_events: list[Event]
    planned_action: "ToolCall"
    conversation_so_far: list["Message"] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Structural pre-check                                                         #
# --------------------------------------------------------------------------- #

def fields_disjoint(
    planned_reads: Iterable[str],
    planned_writes: Iterable[str],
    intervening_events: Iterable[Event],
    action_writes_by_name: dict[str, tuple[str, ...]],
) -> bool:
    """Return True iff the planned action and intervening events touch
    structurally distinct state fields.

    The runtime calls this before invoking the LLM resolver. Conservative
    default: if any input is missing field declarations (caller passes
    ``("__unknown__",)`` for the touch set), this returns False so the
    resolver is consulted.
    """
    planned = set(planned_reads) | set(planned_writes)
    if not planned or "__unknown__" in planned:
        return False

    for event in intervening_events:
        writes = action_writes_by_name.get(event.action_name)
        if writes is None:
            # Unknown action — be conservative.
            return False
        if "__unknown__" in writes:
            return False
        if planned & set(writes):
            return False

    return True


# --------------------------------------------------------------------------- #
# Resolver interface and built-ins                                             #
# --------------------------------------------------------------------------- #

class ConflictResolver(ABC):
    """Turn a ``ConflictContext`` into a ``Decision``."""

    @abstractmethod
    async def resolve(
        self,
        ctx: ConflictContext,
        llm: "LLMProvider | None" = None,
    ) -> Decision: ...


class AlwaysRecoverResolver(ConflictResolver):
    """Always rebuild the prompt from fresh state and let the LLM re-plan.

    Useful as a safe default in low-trust environments and as a deterministic
    fixture in tests where you don't want to mock an LLM.
    """

    async def resolve(
        self,
        ctx: ConflictContext,
        llm: "LLMProvider | None" = None,
    ) -> Decision:
        return Recover()


class ScriptedResolver(ConflictResolver):
    """Test helper — returns scripted decisions in order, then ``Abandon``."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = list(decisions)
        self.calls: list[ConflictContext] = []

    async def resolve(
        self,
        ctx: ConflictContext,
        llm: "LLMProvider | None" = None,
    ) -> Decision:
        self.calls.append(ctx)
        if not self._decisions:
            return Abandon(reason="ScriptedResolver exhausted")
        return self._decisions.pop(0)


# --------------------------------------------------------------------------- #
# LLM-driven resolver                                                          #
# --------------------------------------------------------------------------- #

CONFLICT_PROMPT_TEMPLATE = """\
## Concurrent State Change Detected

While you were planning, another actor updated the project state.

- Your last seen offset: {last_seen_offset}
- Current offset: {current_offset}

Intervening events:
{intervening}

Your originally planned next action:
  `{planned}`

Decide one of:
- `continue` — your plan is still valid; retry the action against the new state.
- `recover` — your plan is stale; re-plan from scratch against the new state.
- `abandon: <reason>` — stop and report back to the user.

Reply with exactly one of those three lines."""


class AgentDrivenConflictResolver(ConflictResolver):
    """Default resolver — asks the LLM to choose between Continue/Recover/Abandon."""

    async def resolve(
        self,
        ctx: ConflictContext,
        llm: "LLMProvider | None" = None,
    ) -> Decision:
        if llm is None:
            # Without an LLM we cannot resolve agentically — fall back to Recover.
            return Recover()

        # Local import to avoid a top-level circular dep.
        from .llm import Message, Role

        prompt = _render_conflict_prompt(ctx)
        response = await llm.chat_complete(
            messages=[
                Message(role=Role.SYSTEM, content=(
                    "You are resolving a concurrent-update conflict. Reply with "
                    "exactly one decision line and nothing else."
                )),
                Message(role=Role.USER, content=prompt),
            ],
        )
        return _parse_decision(response.content or "")


def _render_conflict_prompt(ctx: ConflictContext) -> str:
    intervening_lines = "\n".join(
        f"- offset {e.offset}, by {e.actor}: `{e.action_name}({_format_args(e)})`"
        for e in ctx.intervening_events
    ) or "_(none)_"
    planned_args = _format_tool_call_args(ctx.planned_action)
    return CONFLICT_PROMPT_TEMPLATE.format(
        last_seen_offset=ctx.last_seen_offset,
        current_offset=ctx.current_offset,
        intervening=intervening_lines,
        planned=f"{ctx.planned_action.name}({planned_args})",
    )


def _format_args(event: Event) -> str:
    parts = [repr(a) for a in event.args]
    parts.extend(f"{k}={v!r}" for k, v in event.kwargs.items())
    return ", ".join(parts)


def _format_tool_call_args(call: "ToolCall") -> str:
    return ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())


def _parse_decision(text: str) -> Decision:
    """Parse one of `continue` / `recover` / `abandon: <reason>` (case-insensitive)."""
    line = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
    if line.startswith("continue"):
        return Continue()
    if line.startswith("recover"):
        return Recover()
    if line.startswith("abandon"):
        _, _, reason = line.partition(":")
        return Abandon(reason=reason.strip() or "abandoned by resolver")
    # Unrecognised response → safest default is Recover.
    return Recover()
