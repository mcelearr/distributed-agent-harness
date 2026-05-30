"""
AgentRuntime — the LLM loop that ties everything together.

Takes a WorldEnvironment class plus a pluggable LLMProvider and an EventLog
and runs a ReAct-style loop:

    LLM ──tool calls──► WorldEnvironment @action methods
        ◄──results / conflict resolution───
    LLM ──final message──► OutputChannel

The runtime is fully async. Tool calls execute via ``asyncio.to_thread`` so
the sync @action machinery (read offset / catch up / execute / append)
doesn't block the event loop.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Type

from pydantic import ValidationError, create_model

from .conflict import (
    Abandon,
    AgentDrivenConflictResolver,
    ConcurrentUpdate,
    ConflictContext,
    ConflictResolver,
    Continue,
    Recover,
    fields_disjoint,
)
from .eventlog import EventLog
from .hooks import ActionContext, BlockDecision, HookRegistry
from .llm import LLMProvider, Message, Role, ToolCall, ToolSchema
from .meta_tools import (
    GREP_TOOL,
    LS_TOOL,
    META_TOOL_NAMES,
    READ_TOOL,
    SEARCH_EVENT_LOG_TOOL,
    all_meta_tool_schemas,
    execute_grep,
    execute_ls,
    execute_read,
    execute_search_event_log,
)
from .namespace import NamespaceAdapter
from .prompt_builder import PromptBuilder
from .subagents import SubagentRegistry
from .subagents.dispatch import (
    SUBAGENT_TOOL_PREFIX,
    execute_subagent_call,
    subagent_tool_schemas,
)
from .transport import OutputEvent, OutputEventKind, TriggerEvent
from .world import ActionNotAvailable, BaseWorldEnvironment


# Reserved built-in tool names. Any @action whose name collides with one of
# these would be shadowed by the meta-tool dispatch — fail fast at runtime
# construction rather than silently swallowing user actions.
_RESERVED_TOOL_NAMES: frozenset[str] = META_TOOL_NAMES

log = logging.getLogger(__name__)


DEFAULT_SYSTEM_PREAMBLE = """\
You are an agent operating on a shared, audited project state via a fixed
set of actions. Each action mutates the project's persisted state.

The sections below give you everything you need:

- **Project Summary** — the high-level "card view" of where the project stands.
  Start here. It is the authoritative human-readable narrative.
- **Available Actions** — the only operations you may call. You have no shell
  access and may not invent actions outside this list.
- **Recent Activity** — the last actions taken on this project. Read this
  carefully before acting. If you see you have just done something, do not do
  it again — either move on or report back to the user.
- **Current State** — exact machine values (IDs, timestamps, enums) for when
  you need precise arguments for an action.

Rules:
- Call as many actions as you need to satisfy the user's request, then respond
  with a final plain-text message (no further tool calls) summarising what you did.
- If you do not have enough information to act safely, ask the user instead of guessing.
- Never repeat an action that already appears in Recent Activity with the same arguments
  unless the user has explicitly asked you to redo it.
"""


class AgentRuntime:
    """
    Drives the LLM ↔ WorldEnvironment interaction for a single trigger event.

    Usage::

        runtime = AgentRuntime(
            world_class=DataProtectionWorldEnvironment,
            namespace=InMemoryNamespace(),
            eventlog=InMemoryEventLog(),
            llm=MistralProvider(),
        )

        async for event in cli_chat.events():
            await runtime.handle(event)
    """

    #: Caps applied per call-loop turn.
    CONTINUE_STREAK_CAP = 3   # Consecutive Continue conflicts → escalate to Recover
    RECOVER_CAP = 3           # Recover cycles → Abandon

    def __init__(
        self,
        world_class: Type[BaseWorldEnvironment],
        namespace: NamespaceAdapter,
        eventlog: EventLog,
        llm: LLMProvider,
        max_iterations: int = 12,
        system_preamble: str = DEFAULT_SYSTEM_PREAMBLE,
        include_source_in_prompt: bool = True,
        hooks: HookRegistry | None = None,
        conflict_resolver: ConflictResolver | None = None,
    ) -> None:
        # Fail fast on @action names that would be shadowed by built-in
        # meta-tools (search_event_log, ls, read, grep) or by the
        # consult_<name> subagent prefix.
        action_names = set(world_class.get_actions().keys())
        clashes = action_names & _RESERVED_TOOL_NAMES
        if clashes:
            raise ValueError(
                f"@action name(s) collide with built-in meta-tools: "
                f"{sorted(clashes)}. Reserved names are: "
                f"{sorted(_RESERVED_TOOL_NAMES)}."
            )
        prefix_clashes = {n for n in action_names if n.startswith(SUBAGENT_TOOL_PREFIX)}
        if prefix_clashes:
            raise ValueError(
                f"@action name(s) start with reserved subagent prefix "
                f"{SUBAGENT_TOOL_PREFIX!r}: {sorted(prefix_clashes)}."
            )

        self.world_class = world_class
        self.namespace = namespace
        self.eventlog = eventlog
        self.llm = llm
        self.max_iterations = max_iterations
        self.system_preamble = system_preamble
        self.hooks = hooks if hooks is not None else HookRegistry()
        self.conflict_resolver = conflict_resolver or AgentDrivenConflictResolver()
        self.subagents = SubagentRegistry()
        self._builder = PromptBuilder(world_class, include_source=include_source_in_prompt)
        # Cache tool schemas for @actions; subagent + meta-tool schemas are
        # appended dynamically per turn because subagents can be registered
        # after the runtime is created.
        self._action_tool_schemas, self._param_models = _build_tool_schemas(world_class)
        self._action_writes = world_class.action_writes_by_name()

    # ----------------------------------------------------------------------- #
    # Convenience hook registration — delegate to self.hooks                   #
    # ----------------------------------------------------------------------- #

    def on_pre_action(self, action: str | None = None):
        """Register a pre-action hook. See ``HookRegistry.on_pre_action``."""
        return self.hooks.on_pre_action(action)

    def on_post_action(self, action: str | None = None):
        """Register a post-action hook. See ``HookRegistry.on_post_action``."""
        return self.hooks.on_post_action(action)

    def on_action_error(self, action: str | None = None):
        """Register an action-error hook. See ``HookRegistry.on_action_error``."""
        return self.hooks.on_action_error(action)

    def on_pre_trigger(self, fn):
        """Register a pre-trigger hook. See ``HookRegistry.on_pre_trigger``."""
        return self.hooks.on_pre_trigger(fn)

    def on_run_complete(self, fn):
        """Register a run-complete hook. See ``HookRegistry.on_run_complete``."""
        return self.hooks.on_run_complete(fn)

    # ----------------------------------------------------------------------- #
    # Public entry point                                                       #
    # ----------------------------------------------------------------------- #

    async def handle(self, event: TriggerEvent) -> Message:
        """
        Process one TriggerEvent end-to-end.

        Builds an agent loop, dispatches tool calls to WorldEnvironment
        actions, streams events to ``event.reply_to`` if present, and returns
        the final assistant message.

        The system prompt is **rebuilt before every LLM call** so that the
        Project Summary, Recent Activity, and Current State sections always
        reflect the latest namespace contents — including actions taken
        earlier in this very run. This is the key mechanism for in-run
        loop prevention.

        Hook lifecycle (in firing order):
            pre_trigger → [ pre_action → action → post_action ]* → run_complete
        """
        reply_to = event.reply_to
        final: Message = Message(role=Role.ASSISTANT, content="")

        # ----- pre_trigger: gate the whole run before we even instantiate the world
        decision = await self.hooks.fire_pre_trigger(event)
        if decision is not None:
            final = Message(
                role=Role.ASSISTANT,
                content=f"[trigger blocked] {decision.reason}",
            )
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ERROR,
                    payload={"error": f"Trigger blocked: {decision.reason}"},
                ))
                await reply_to.emit(OutputEvent(kind=OutputEventKind.FINAL))
            await self.hooks.fire_run_complete(event, final)
            return final

        world = self.world_class(
            project_id=event.project_id,
            namespace=self.namespace,
            eventlog=self.eventlog,
        )

        user_message = self._user_message_for(event)
        # The conversation grows with each assistant/tool turn. The system
        # prompt is regenerated fresh on every iteration and prepended.
        conversation: list[Message] = []

        try:
            recover_count = 0
            for _ in range(self.max_iterations):
                # Refresh world state from the namespace so the system prompt
                # reflects mutations from the previous tool calls (or other agents).
                world._hydrate()
                system_message = self._build_system_message(world, event=event)
                messages = [system_message, user_message, *conversation]

                turn_tools = self._tool_schemas_for_turn(world, event)
                assistant = await self.llm.chat_complete(messages, tools=turn_tools)
                conversation.append(assistant)

                if not assistant.tool_calls:
                    final = assistant
                    if reply_to and assistant.content:
                        await reply_to.emit(OutputEvent(
                            kind=OutputEventKind.MESSAGE,
                            payload={"content": assistant.content},
                        ))
                    break

                # Execute each tool call. Any call may surface a
                # ConcurrentUpdate, which the conflict pipeline resolves into
                # Continue / Recover / Abandon.
                recover_requested = False
                abandon_reason: str | None = None
                tool_messages_in_turn: list[Message] = []
                for call in assistant.tool_calls:
                    outcome, result_message = await self._execute_with_conflict_handling(
                        call, world, reply_to, event,
                    )
                    if result_message is not None:
                        conversation.append(result_message)
                        tool_messages_in_turn.append(result_message)
                    if isinstance(outcome, Recover):
                        recover_requested = True
                        break
                    if isinstance(outcome, Abandon):
                        abandon_reason = outcome.reason
                        break

                if abandon_reason is not None:
                    final = Message(
                        role=Role.ASSISTANT,
                        content=f"[abandoned] {abandon_reason}",
                    )
                    if reply_to:
                        await reply_to.emit(OutputEvent(
                            kind=OutputEventKind.MESSAGE,
                            payload={"content": final.content},
                        ))
                    break

                if recover_requested:
                    recover_count += 1
                    if recover_count >= self.RECOVER_CAP:
                        abandon_msg = (
                            f"retry cap exceeded ({recover_count} Recover cycles)"
                        )
                        final = Message(
                            role=Role.ASSISTANT,
                            content=f"[abandoned] {abandon_msg}",
                        )
                        if reply_to:
                            await reply_to.emit(OutputEvent(
                                kind=OutputEventKind.MESSAGE,
                                payload={"content": final.content},
                            ))
                        break
                    # Drop the stale assistant turn (and any tool replies it
                    # produced) and let the next loop iteration re-plan
                    # against the fresh system prompt. A synthetic SYSTEM
                    # note tells the model why.
                    for _ in range(len(tool_messages_in_turn)):
                        conversation.pop()
                    if conversation and conversation[-1] is assistant:
                        conversation.pop()
                    conversation.append(_recover_note(world._last_seen_offset))
                    continue
            else:
                # Hit max_iterations without a final message
                if reply_to:
                    await reply_to.emit(OutputEvent(
                        kind=OutputEventKind.ERROR,
                        payload={"error": f"Hit max_iterations={self.max_iterations}"},
                    ))
        except Exception as exc:  # noqa: BLE001 — surface anything to the channel
            log.exception("AgentRuntime error")
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ERROR,
                    payload={"error": str(exc)},
                ))
            raise
        finally:
            if reply_to:
                await reply_to.emit(OutputEvent(kind=OutputEventKind.FINAL))
            # run_complete fires regardless of how the run ended — success,
            # block, max_iterations, or exception in the loop body.
            await self.hooks.fire_run_complete(event, final)

        return final

    # ----------------------------------------------------------------------- #
    # Internals                                                                #
    # ----------------------------------------------------------------------- #

    def _build_system_message(
        self,
        world: BaseWorldEnvironment,
        event: TriggerEvent | None = None,
    ) -> Message:
        """Build a fresh system message reflecting the world's current state."""
        content = (
            self.system_preamble
            + "\n\n"
            + self._builder.build_full_prompt(
                world, event=event, subagents=self.subagents.list(),
            )
        )
        return Message(role=Role.SYSTEM, content=content)

    @staticmethod
    def _user_message_for(event: TriggerEvent) -> Message:
        """Convert a TriggerEvent into a user-role message for the LLM."""
        text = event.payload.get("text")
        if text:
            return Message(role=Role.USER, content=text)
        return Message(
            role=Role.USER,
            content=(
                f"Trigger received from '{event.source}' of kind "
                f"'{event.kind.value}'. Payload: {event.payload}"
            ),
        )

    async def _execute_call(
        self,
        call: ToolCall,
        world: BaseWorldEnvironment,
        reply_to: Any,
        trigger: TriggerEvent,
    ) -> Message:
        """Run one tool call and return a TOOL message.

        Three dispatch categories:
        - ``search_event_log``    → read-only meta-tool, no hooks, no event append
        - ``consult_<name>``      → subagent registry
        - any other name          → ``@action`` on the WorldEnvironment
        """
        if reply_to:
            await reply_to.emit(OutputEvent(
                kind=OutputEventKind.ACTION_CALLED,
                payload={"name": call.name, "args": call.arguments, "id": call.id},
            ))

        # ----- meta-tool: search_event_log
        if call.name == SEARCH_EVENT_LOG_TOOL:
            return await execute_search_event_log(
                call, reply_to, self.eventlog, world._project_id,
            )

        # ----- meta-tools: namespace navigation (ls / read / grep)
        if call.name == LS_TOOL:
            return await execute_ls(call, reply_to, self.namespace)
        if call.name == READ_TOOL:
            return await execute_read(call, reply_to, self.namespace)
        if call.name == GREP_TOOL:
            return await execute_grep(call, reply_to, self.namespace)

        # ----- subagent: consult_<name>
        if call.name.startswith(SUBAGENT_TOOL_PREFIX):
            subagent_name = call.name[len(SUBAGENT_TOOL_PREFIX):]
            subagent = self.subagents.get(subagent_name)
            if subagent is not None:
                return await execute_subagent_call(
                    call, subagent_name, subagent, world, reply_to, trigger,
                    hooks=self.hooks,
                    eventlog=self.eventlog,
                    namespace=self.namespace,
                )

        method = getattr(world, call.name, None)
        if method is None or not getattr(method, "_is_action", False):
            error = f"Unknown action: {call.name!r}"
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ACTION_RESULT,
                    payload={"name": call.name, "id": call.id, "error": error},
                ))
            return Message(
                role=Role.TOOL,
                content=error,
                tool_call_id=call.id,
                name=call.name,
            )

        # Validate and coerce arguments via the cached Pydantic model.
        # This turns e.g. "erasure" → DSRType.ERASURE, ISO strings → datetime, etc.
        try:
            params_model = self._param_models[call.name]
            validated = params_model.model_validate(call.arguments)
            kwargs = {
                name: getattr(validated, name)
                for name in params_model.model_fields
            }
        except ValidationError as exc:
            error = f"Invalid arguments for {call.name}: {exc}"
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ACTION_RESULT,
                    payload={"name": call.name, "id": call.id, "error": error},
                ))
            return Message(
                role=Role.TOOL,
                content=error,
                tool_call_id=call.id,
                name=call.name,
            )

        # Build the context that every action-level hook receives.
        ctx = ActionContext(
            project_id=world._project_id,
            action_name=call.name,
            args=(),
            kwargs=kwargs,
            trigger=trigger,
        )

        # ----- pre_action: blocking hooks may halt the call
        decision = await self.hooks.fire_pre_action(ctx)
        if decision is not None:
            error = f"Action blocked: {decision.reason}"
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ACTION_RESULT,
                    payload={
                        "name": call.name,
                        "id": call.id,
                        "error": error,
                        "blocked": True,
                    },
                ))
            return Message(
                role=Role.TOOL,
                content=error,
                tool_call_id=call.id,
                name=call.name,
            )

        # Stash the trigger on the world so the @action wrapper can pass it
        # to the ``show_when`` predicate. Cleared in ``finally`` even on
        # error so direct callers never see a stale value.
        world._pending_trigger = trigger

        # The @action wrapper does its own optimistic CAS; offload to a
        # thread so we don't block the event loop. ConcurrentUpdate is
        # propagated up so the conflict pipeline can resolve it.
        try:
            result = await asyncio.to_thread(method, **kwargs)
            await self.hooks.fire_post_action(ctx, result)
            result_text = _stringify_result(result)
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ACTION_RESULT,
                    payload={"name": call.name, "id": call.id, "result": result_text},
                ))
            return Message(
                role=Role.TOOL,
                content=result_text,
                tool_call_id=call.id,
                name=call.name,
            )
        except ConcurrentUpdate:
            # Surfaced to the conflict pipeline by the caller; not a TOOL
            # message and not an error event.
            raise
        except ActionNotAvailable as exc:
            # show_when=False is surfaced the same way as a blocked
            # pre_action hook: the LLM sees a clear "this is not allowed
            # right now" message and can adapt rather than thrashing.
            error = f"Action blocked: {exc.reason}"
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ACTION_RESULT,
                    payload={
                        "name": call.name,
                        "id": call.id,
                        "error": error,
                        "blocked": True,
                    },
                ))
            return Message(
                role=Role.TOOL,
                content=error,
                tool_call_id=call.id,
                name=call.name,
            )
        except Exception as exc:  # noqa: BLE001 — must surface to the LLM
            await self.hooks.fire_action_error(ctx, exc)
            error = f"{type(exc).__name__}: {exc}"
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ACTION_RESULT,
                    payload={"name": call.name, "id": call.id, "error": error},
                ))
            return Message(
                role=Role.TOOL,
                content=f"ERROR: {error}",
                tool_call_id=call.id,
                name=call.name,
            )
        finally:
            world._pending_trigger = None

    async def _execute_with_conflict_handling(
        self,
        call: ToolCall,
        world: BaseWorldEnvironment,
        reply_to: Any,
        trigger: TriggerEvent,
    ) -> tuple[Any, Message | None]:
        """Drive ``_execute_call`` with optimistic-retry + conflict resolution.

        Returns ``(outcome, message)``:
        - outcome is a string ``"ok"`` on plain success, an instance of
          ``Recover`` if the caller's turn should be re-planned, or an
          instance of ``Abandon`` if the run should stop.
        - message is the TOOL message to thread back into the conversation,
          or None when no TOOL message should be inserted (Recover / Abandon).
        """
        method = getattr(world, call.name, None)
        planned_reads = tuple(getattr(method, "_reads", ())) if method else ()
        planned_writes = tuple(getattr(method, "_writes", ())) if method else ()

        continue_streak = 0
        while True:
            try:
                msg = await self._execute_call(call, world, reply_to, trigger)
                return "ok", msg
            except ConcurrentUpdate as conflict:
                if reply_to:
                    await reply_to.emit(OutputEvent(
                        kind=OutputEventKind.ERROR,
                        payload={
                            "error": str(conflict),
                            "kind": "conflict",
                            "action": call.name,
                            "intervening": [
                                e.action_name for e in conflict.intervening_events
                            ],
                        },
                    ))

                # Structural pre-check — auto-Continue when intervening
                # writes can't have invalidated the planned read/write set.
                auto_continue = (
                    bool(planned_reads or planned_writes)
                    and fields_disjoint(
                        planned_reads,
                        planned_writes,
                        conflict.intervening_events,
                        self._action_writes,
                    )
                )

                if auto_continue or continue_streak >= self.CONTINUE_STREAK_CAP:
                    decision = (
                        Continue()
                        if auto_continue
                        else Recover()
                    )
                else:
                    ctx = ConflictContext(
                        project_id=world._project_id,
                        last_seen_offset=conflict.last_seen_offset,
                        current_offset=conflict.current_offset,
                        intervening_events=conflict.intervening_events,
                        planned_action=call,
                    )
                    decision = await self.conflict_resolver.resolve(ctx, self.llm)

                # Once the conflict has been surfaced and a decision made,
                # the agent has acknowledged the intervening events. Advance
                # the CAS token past them; the snapshot is assumed
                # consistent (every successful append flushes ``state.json``
                # before returning).
                world._hydrate()
                world._last_seen_offset = conflict.current_offset

                if isinstance(decision, Continue):
                    continue_streak += 1
                    continue
                if isinstance(decision, Recover):
                    return Recover(), None
                if isinstance(decision, Abandon):
                    return decision, None

                # Unknown decision type — treat as Recover.
                return Recover(), None


    # ----------------------------------------------------------------------- #
    # Per-turn tool schema assembly                                            #
    # ----------------------------------------------------------------------- #

    def _tool_schemas_for_turn(
        self,
        world: BaseWorldEnvironment,
        event: TriggerEvent | None,
    ) -> list[ToolSchema]:
        """Return the full tool list visible to the LLM this turn.

        Composition:
        - all ``@action`` schemas whose ``show_when`` matches current state
        - one ``consult_<name>`` schema per registered subagent whose
          ``show_when`` matches
        - the always-on built-in meta-tools: ``ls``, ``read``, ``grep``,
          ``search_event_log``
        """
        visible_actions = _visible_action_names(self.world_class, world.state, event)
        schemas = [s for s in self._action_tool_schemas if s.name in visible_actions]
        schemas.extend(subagent_tool_schemas(self.subagents, world.state, event))
        schemas.extend(all_meta_tool_schemas())
        return schemas


# --------------------------------------------------------------------------- #
# Tool schema generation                                                       #
# --------------------------------------------------------------------------- #

def _build_tool_schemas(
    world_class: Type[BaseWorldEnvironment],
) -> tuple[list[ToolSchema], dict[str, Any]]:
    """
    Generate JSON-Schema tool definitions for every @action on ``world_class``.

    Returns:
        - A list of ``ToolSchema`` ready to pass to an LLMProvider
        - A dict mapping action name → Pydantic params model (for arg validation)
    """
    schemas: list[ToolSchema] = []
    models: dict[str, Any] = {}

    for name, method in world_class.get_actions().items():
        params_model = _params_model_for(name, method)
        json_schema = params_model.model_json_schema()
        # Strip Pydantic's "title" noise and ensure type=object at the top level
        json_schema.pop("title", None)
        json_schema.setdefault("type", "object")
        doc = inspect.getdoc(method) or f"Call {name}."
        schemas.append(ToolSchema(
            name=name,
            description=doc,
            parameters=json_schema,
        ))
        models[name] = params_model

    return schemas, models


def _params_model_for(name: str, method: Any):
    """
    Build a Pydantic model representing the keyword arguments of ``method``.

    The model is used both to generate the JSON schema for the LLM and to
    validate incoming tool-call arguments before invoking the method.

    Annotations are resolved via ``typing.get_type_hints`` so that worlds
    defined under ``from __future__ import annotations`` (where every
    annotation is a string at runtime) yield real types — otherwise pydantic
    would receive forward references like ``'BreachSeverity'`` and fail at
    ``model_json_schema()`` time with a "class not fully defined" error.
    """
    import typing

    sig = inspect.signature(method)
    # ``get_type_hints`` resolves forward references against the method's
    # module globals. The ``include_extras=True`` keeps any ``Annotated[...]``
    # metadata (we don't use it today, but cheap to preserve).
    try:
        resolved_hints = typing.get_type_hints(method, include_extras=True)
    except Exception:  # noqa: BLE001 — fall back to raw annotations
        resolved_hints = {}

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name in resolved_hints:
            annotation = resolved_hints[param_name]
        elif param.annotation is not inspect.Parameter.empty:
            annotation = param.annotation
        else:
            annotation = Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param_name] = (annotation, default)

    model_name = f"{name.title().replace('_', '')}Params"
    return create_model(model_name, **fields)


def _stringify_result(result: Any) -> str:
    """Render a method's return value as a string for the LLM to read."""
    if result is None:
        return "OK"
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json()
    return str(result)


def _recover_note(offset: int) -> Message:
    """Synthetic system message injected after a Recover decision.

    The next loop iteration will rebuild the full system prompt against the
    freshly-projected state; this note tells the model *why* it is being
    asked to re-plan.
    """
    return Message(
        role=Role.SYSTEM,
        content=(
            "Concurrent state changes were detected during your last turn. "
            f"The project state has advanced to offset {offset}. Please re-plan "
            "from scratch against the latest state shown above; do not assume "
            "your previous plan is still valid."
        ),
    )


# --------------------------------------------------------------------------- #
# Visibility helpers — used per-turn to filter the tool list                   #
# --------------------------------------------------------------------------- #

def _visible_action_names(
    world_class: Type[BaseWorldEnvironment],
    state: Any,
    event: TriggerEvent | None,
) -> set[str]:
    """Return action names whose ``show_when`` matches now (or is unset)."""
    visible: set[str] = set()
    for name, method in world_class.get_actions().items():
        predicate = getattr(method, "_show_when", None)
        if predicate is None:
            visible.add(name)
            continue
        try:
            if bool(predicate(state, event)):
                visible.add(name)
        except Exception:  # noqa: BLE001 — buggy predicate hides the action
            log.exception("show_when raised for %r; hiding action", name)
    return visible


