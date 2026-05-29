"""
Lifecycle hooks — pluggable Python callables that fire at well-defined
points in the agent runtime.

Use cases:

- **Approval gates**: register a ``pre_action`` hook for sensitive actions
  (e.g. ``notify_ico``) that returns a ``BlockDecision`` until a human signs off.
- **External integrations**: register a ``post_action`` hook for
  ``report_breach`` that posts to Slack.
- **Telemetry**: register a ``run_complete`` hook that emits a metric for
  every agent run.
- **Policy enforcement**: register a ``pre_trigger`` hook that rejects
  webhook payloads from unknown sources.

All hooks are async. Hooks that return ``None`` allow execution to proceed;
pre-* hooks may instead return a ``BlockDecision`` to halt the operation
(the runtime surfaces this to the LLM, or to the OutputChannel for triggers).

The model is deliberately simpler than Claude Code's hooks: ours are
in-process Python callables, not external shell commands. We don't need a
serialisation boundary because we already live in the same Python process
as the runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .llm import Message
    from .transport import TriggerEvent

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class ActionContext:
    """
    Context passed to action-level hooks (pre_action, post_action, action_error).

    Hooks receive enough information to make policy decisions and route
    notifications without needing to dig into runtime internals.
    """
    project_id: str
    action_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    trigger: "TriggerEvent | None" = None   # the TriggerEvent that led here, if any


@dataclass
class SubagentContext:
    """Context passed to subagent-level hooks (pre_subagent_call, post_subagent_call).

    A distinct shape from ``ActionContext`` because subagent policies are
    semantically different (cost ceilings, allowlists, session tracking).
    """
    project_id: str
    subagent_name: str
    message: str
    session_id: str | None = None
    trigger: "TriggerEvent | None" = None


@dataclass(frozen=True)
class BlockDecision:
    """
    Returned by a pre-* hook to halt the operation.

    The runtime surfaces the ``reason`` to the LLM (for blocked actions or
    subagents) or to the OutputChannel as an ERROR event (for blocked
    triggers).
    """
    reason: str


# Type aliases for hook callables. All hooks are async.
PreActionHook = Callable[[ActionContext], Awaitable["BlockDecision | None"]]
PostActionHook = Callable[[ActionContext, Any], Awaitable[None]]
ActionErrorHook = Callable[[ActionContext, BaseException], Awaitable[None]]
PreTriggerHook = Callable[["TriggerEvent"], Awaitable["BlockDecision | None"]]
RunCompleteHook = Callable[["TriggerEvent", "Message"], Awaitable[None]]
PreSubagentCallHook = Callable[[SubagentContext], Awaitable["BlockDecision | None"]]
PostSubagentCallHook = Callable[[SubagentContext, Any], Awaitable[None]]


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

class HookRegistry:
    """
    Pluggable lifecycle hook registry.

    Five events are supported:

    - ``pre_action(action_name=None)``  — fires before an @action runs.
      Returning a ``BlockDecision`` halts the call and surfaces the reason
      to the LLM.
    - ``post_action(action_name=None)`` — fires after an @action returns.
      Side-effect only; return value is ignored.
    - ``action_error(action_name=None)`` — fires if an @action raises.
      Cannot suppress the exception; the runtime still surfaces it.
    - ``pre_trigger`` — fires when a TriggerEvent is received. Can block.
    - ``run_complete`` — fires when an agent run finishes (success, block,
      or max-iterations).

    Action hooks support per-name filtering: pass an action name to fire
    only for that action, or omit it (``None``) to fire for every action.
    Both specific and wildcard hooks fire — specific first.

    Hooks raised during execution are logged and swallowed for ``post_action``,
    ``action_error``, and ``run_complete`` (they should never break a run).
    ``pre_action`` and ``pre_trigger`` raise propagate, since they sit on the
    critical path of allowing the operation through.
    """

    def __init__(self) -> None:
        self._pre_action: dict[str | None, list[PreActionHook]] = {}
        self._post_action: dict[str | None, list[PostActionHook]] = {}
        self._action_error: dict[str | None, list[ActionErrorHook]] = {}
        self._pre_trigger: list[PreTriggerHook] = []
        self._run_complete: list[RunCompleteHook] = []
        self._pre_subagent_call: dict[str | None, list[PreSubagentCallHook]] = {}
        self._post_subagent_call: dict[str | None, list[PostSubagentCallHook]] = {}

    # ----------------------------------------------------------------------- #
    # Registration — decorators                                                #
    # ----------------------------------------------------------------------- #

    def on_pre_action(
        self, action: str | None = None
    ) -> Callable[[PreActionHook], PreActionHook]:
        """
        Register a pre-action hook. Use as a decorator::

            @registry.on_pre_action("notify_ico")
            async def require_partner_signoff(ctx: ActionContext):
                if not await approved(ctx):
                    return BlockDecision(reason="Awaiting partner sign-off")

        Pass ``action=None`` (the default) to fire for every action.
        """
        def decorator(fn: PreActionHook) -> PreActionHook:
            self._pre_action.setdefault(action, []).append(fn)
            return fn
        return decorator

    def on_post_action(
        self, action: str | None = None
    ) -> Callable[[PostActionHook], PostActionHook]:
        """Register a post-action hook (fires after the action returns successfully)."""
        def decorator(fn: PostActionHook) -> PostActionHook:
            self._post_action.setdefault(action, []).append(fn)
            return fn
        return decorator

    def on_action_error(
        self, action: str | None = None
    ) -> Callable[[ActionErrorHook], ActionErrorHook]:
        """Register a hook that fires when an action raises an exception."""
        def decorator(fn: ActionErrorHook) -> ActionErrorHook:
            self._action_error.setdefault(action, []).append(fn)
            return fn
        return decorator

    def on_pre_trigger(self, fn: PreTriggerHook) -> PreTriggerHook:
        """Register a hook that fires when a TriggerEvent is received."""
        self._pre_trigger.append(fn)
        return fn

    def on_run_complete(self, fn: RunCompleteHook) -> RunCompleteHook:
        """Register a hook that fires when an agent run finishes."""
        self._run_complete.append(fn)
        return fn

    def on_pre_subagent_call(
        self, name: str | None = None
    ) -> Callable[[PreSubagentCallHook], PreSubagentCallHook]:
        """Register a hook that fires before a subagent ``consult`` runs.

        Returning a ``BlockDecision`` halts the call and surfaces the
        reason to the LLM as a blocked TOOL message — same shape as a
        blocked ``pre_action``. Use for cost ceilings, allowlists, etc.

        Pass ``name=None`` (default) to fire for every subagent.
        """
        def decorator(fn: PreSubagentCallHook) -> PreSubagentCallHook:
            self._pre_subagent_call.setdefault(name, []).append(fn)
            return fn
        return decorator

    def on_post_subagent_call(
        self, name: str | None = None
    ) -> Callable[[PostSubagentCallHook], PostSubagentCallHook]:
        """Register a hook that fires after a subagent ``consult`` returns."""
        def decorator(fn: PostSubagentCallHook) -> PostSubagentCallHook:
            self._post_subagent_call.setdefault(name, []).append(fn)
            return fn
        return decorator

    # ----------------------------------------------------------------------- #
    # Firing — called by AgentRuntime                                          #
    # ----------------------------------------------------------------------- #

    async def fire_pre_action(self, ctx: ActionContext) -> BlockDecision | None:
        """Run all matching pre_action hooks. Return first BlockDecision, if any."""
        for key in (ctx.action_name, None):
            for hook in self._pre_action.get(key, []):
                result = await hook(ctx)
                if isinstance(result, BlockDecision):
                    return result
        return None

    async def fire_post_action(self, ctx: ActionContext, result: Any) -> None:
        """Run all matching post_action hooks. Exceptions are logged and swallowed."""
        for key in (ctx.action_name, None):
            for hook in self._post_action.get(key, []):
                try:
                    await hook(ctx, result)
                except Exception:
                    log.exception(
                        "post_action hook for %r raised; ignoring", ctx.action_name
                    )

    async def fire_action_error(
        self, ctx: ActionContext, exc: BaseException
    ) -> None:
        """Run all matching action_error hooks. Exceptions are logged and swallowed."""
        for key in (ctx.action_name, None):
            for hook in self._action_error.get(key, []):
                try:
                    await hook(ctx, exc)
                except Exception:
                    log.exception(
                        "action_error hook for %r raised; ignoring", ctx.action_name
                    )

    async def fire_pre_trigger(
        self, event: "TriggerEvent"
    ) -> BlockDecision | None:
        """Run all pre_trigger hooks. Return first BlockDecision, if any."""
        for hook in self._pre_trigger:
            result = await hook(event)
            if isinstance(result, BlockDecision):
                return result
        return None

    async def fire_run_complete(
        self, event: "TriggerEvent", final: "Message"
    ) -> None:
        """Run all run_complete hooks. Exceptions are logged and swallowed."""
        for hook in self._run_complete:
            try:
                await hook(event, final)
            except Exception:
                log.exception("run_complete hook raised; ignoring")

    async def fire_pre_subagent_call(
        self, ctx: SubagentContext
    ) -> BlockDecision | None:
        """Run all matching pre_subagent_call hooks. Return first BlockDecision."""
        for key in (ctx.subagent_name, None):
            for hook in self._pre_subagent_call.get(key, []):
                result = await hook(ctx)
                if isinstance(result, BlockDecision):
                    return result
        return None

    async def fire_post_subagent_call(
        self, ctx: SubagentContext, response: Any
    ) -> None:
        """Run all matching post_subagent_call hooks. Exceptions are logged."""
        for key in (ctx.subagent_name, None):
            for hook in self._post_subagent_call.get(key, []):
                try:
                    await hook(ctx, response)
                except Exception:
                    log.exception(
                        "post_subagent_call hook for %r raised; ignoring",
                        ctx.subagent_name,
                    )
