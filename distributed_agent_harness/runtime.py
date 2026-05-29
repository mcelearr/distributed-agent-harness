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
import hashlib
import inspect
import logging
import re
from typing import Any, Type

from pydantic import ValidationError, create_model

from datetime import datetime, timezone

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
from .event_search import EventQuery, render_events_markdown, search_events
from .eventlog import Appended, Event, EventLog
from .hooks import ActionContext, BlockDecision, HookRegistry, SubagentContext
from .llm import LLMProvider, Message, Role, ToolCall, ToolSchema
from .namespace import NamespaceAdapter
from .namespace_browse import (
    DEFAULT_GREP_LIMIT,
    DEFAULT_READ_LIMIT,
    grep_docs,
    list_dir,
    read_doc,
    render_grep,
    render_ls,
    render_read,
)
from .prompt_builder import PromptBuilder
from .subagents import (
    Artefact,
    SubagentRegistry,
    SubagentResponse,
    SubagentTimeout,
)
from .transport import OutputEvent, OutputEventKind, TriggerEvent
from .world import ActionNotAvailable, BaseWorldEnvironment


# Reserved built-in tool names. Any @action whose name collides with one of
# these would be shadowed by the meta-tool dispatch — fail fast at runtime
# construction rather than silently swallowing user actions.
_SEARCH_EVENT_LOG_TOOL = "search_event_log"
_LS_TOOL = "ls"
_READ_TOOL = "read"
_GREP_TOOL = "grep"
_SUBAGENT_TOOL_PREFIX = "consult_"
_RESERVED_TOOL_NAMES = frozenset({
    _SEARCH_EVENT_LOG_TOOL, _LS_TOOL, _READ_TOOL, _GREP_TOOL,
})

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
        prefix_clashes = {n for n in action_names if n.startswith(_SUBAGENT_TOOL_PREFIX)}
        if prefix_clashes:
            raise ValueError(
                f"@action name(s) start with reserved subagent prefix "
                f"{_SUBAGENT_TOOL_PREFIX!r}: {sorted(prefix_clashes)}."
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
        if call.name == _SEARCH_EVENT_LOG_TOOL:
            return await self._execute_search_event_log(call, world, reply_to)

        # ----- meta-tools: namespace navigation (ls / read / grep)
        if call.name == _LS_TOOL:
            return await self._execute_ls(call, reply_to)
        if call.name == _READ_TOOL:
            return await self._execute_read(call, reply_to)
        if call.name == _GREP_TOOL:
            return await self._execute_grep(call, reply_to)

        # ----- subagent: consult_<name>
        if call.name.startswith(_SUBAGENT_TOOL_PREFIX):
            subagent_name = call.name[len(_SUBAGENT_TOOL_PREFIX):]
            subagent = self.subagents.get(subagent_name)
            if subagent is not None:
                return await self._execute_subagent_call(
                    call, subagent_name, subagent, world, reply_to, trigger,
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
    # Tool dispatch — subagent (consult_<name>)                                #
    # ----------------------------------------------------------------------- #

    async def _execute_subagent_call(
        self,
        call: ToolCall,
        subagent_name: str,
        subagent: Any,
        world: BaseWorldEnvironment,
        reply_to: Any,
        trigger: TriggerEvent,
    ) -> Message:
        """Dispatch a ``consult_<name>`` tool call to the subagent registry."""
        message = call.arguments.get("message")
        session_id = call.arguments.get("session_id")
        if not isinstance(message, str) or not message:
            error = f"Invalid arguments for {call.name}: 'message' is required"
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

        # ----- pre_subagent_call hook
        sub_ctx = SubagentContext(
            project_id=world._project_id,
            subagent_name=subagent_name,
            message=message,
            session_id=session_id if isinstance(session_id, str) else None,
            trigger=trigger,
        )
        decision = await self.hooks.fire_pre_subagent_call(sub_ctx)
        if decision is not None:
            error = f"Subagent blocked: {decision.reason}"
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.ACTION_RESULT,
                    payload={
                        "name": call.name, "id": call.id,
                        "error": error, "blocked": True,
                    },
                ))
            return Message(
                role=Role.TOOL,
                content=error,
                tool_call_id=call.id,
                name=call.name,
            )

        # ----- progress callback: forward `working` deltas as THINKING events
        async def _on_progress(delta: str) -> None:
            if reply_to:
                await reply_to.emit(OutputEvent(
                    kind=OutputEventKind.THINKING,
                    payload={
                        "subagent": subagent_name,
                        "delta": delta,
                    },
                ))

        # ----- the consult itself
        try:
            response: SubagentResponse = await subagent.consult(
                message=message,
                session_id=sub_ctx.session_id,
                on_progress=_on_progress,
            )
        except SubagentTimeout as exc:
            await self._append_subagent_event(
                world, subagent_name, message, sub_ctx.session_id,
                status="failed", content=f"timeout: {exc}",
            )
            error = f"Subagent timed out: {exc}"
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
        except Exception as exc:  # noqa: BLE001 — surface to LLM
            await self._append_subagent_event(
                world, subagent_name, message, sub_ctx.session_id,
                status="failed", content=str(exc),
            )
            error = f"Subagent error: {type(exc).__name__}: {exc}"
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

        # ----- persist artefacts before recording the event so the event's
        # result_summary can include their paths (grep-discoverable later)
        artefact_records = _persist_artefacts(
            self.namespace, world._project_id, response.artefacts,
        )

        # ----- record the consult event
        await self._append_subagent_event(
            world, subagent_name, message, sub_ctx.session_id,
            status=response.status,
            content=response.content,
            returned_session_id=response.session_id,
            artefact_records=artefact_records,
        )

        # ----- post_subagent_call hook
        await self.hooks.fire_post_subagent_call(sub_ctx, response)

        result_text = _render_subagent_response(
            subagent_name, response, artefact_records,
        )
        if reply_to:
            await reply_to.emit(OutputEvent(
                kind=OutputEventKind.ACTION_RESULT,
                payload={
                    "name": call.name, "id": call.id,
                    "result": result_text,
                    "status": response.status,
                    "session_id": response.session_id,
                },
            ))
        return Message(
            role=Role.TOOL,
            content=result_text,
            tool_call_id=call.id,
            name=call.name,
        )

    async def _append_subagent_event(
        self,
        world: BaseWorldEnvironment,
        subagent_name: str,
        message: str,
        session_id: str | None,
        status: str,
        content: str,
        returned_session_id: str | None = None,
        artefact_records: list[dict] | None = None,
    ) -> None:
        """Append a `consult_<name>` event with transparent CAS retry.

        Subagent consults don't depend on the world's state, so any
        intervening @action writes cannot invalidate them — we just keep
        trying until our append wins. No conflict is surfaced to the LLM.

        Artefact paths are inlined into ``result_summary`` so the LLM
        can rediscover them later via ``search_event_log`` or ``grep``.
        """
        summary_parts = [
            f"status={status}",
            f"session={returned_session_id or session_id}",
            f"content={_truncate_for_log(content)}",
        ]
        if artefact_records:
            paths_inline = ", ".join(a["path"] for a in artefact_records)
            summary_parts.append(f"artefacts=[{paths_inline}]")

        event = Event(
            project_id=world._project_id,
            action_name=f"{_SUBAGENT_TOOL_PREFIX}{subagent_name}",
            args=[],
            kwargs={
                "message": _truncate_for_log(message),
                "session_id": session_id,
            },
            actor="agent",
            result_summary=" ".join(summary_parts),
        )
        while True:
            offset = await self.eventlog.current_offset(world._project_id)
            result = await self.eventlog.append(
                world._project_id, event, expected_offset=offset,
            )
            if isinstance(result, Appended):
                return
            # Conflict — refresh and try again. Subagent calls are
            # commutative w.r.t. any concurrent @action.

    # ----------------------------------------------------------------------- #
    # Tool dispatch — search_event_log (read-only meta-tool)                   #
    # ----------------------------------------------------------------------- #

    async def _execute_search_event_log(
        self,
        call: ToolCall,
        world: BaseWorldEnvironment,
        reply_to: Any,
    ) -> Message:
        """Dispatch the built-in ``search_event_log`` meta-tool."""
        args = call.arguments or {}
        try:
            query = EventQuery(
                action_name_glob=args.get("action_name_glob"),
                grep=args.get("grep"),
                actor=args.get("actor"),
                since=_parse_iso_datetime(args.get("since")),
                until=_parse_iso_datetime(args.get("until")),
                limit=int(args.get("limit") or 20),
                offset_from=(
                    int(args["offset_from"]) if args.get("offset_from") is not None else None
                ),
            )
        except (TypeError, ValueError) as exc:
            error = f"Invalid arguments for search_event_log: {exc}"
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

        events = await search_events(self.eventlog, world._project_id, query)
        result_text = render_events_markdown(events)
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

    # ----------------------------------------------------------------------- #
    # Tool dispatch — filesystem-style navigation (ls / read / grep)           #
    # ----------------------------------------------------------------------- #
    #
    # These mirror Claude Code's / PI's baseline read tools so the agent can
    # browse the project namespace as a virtual filesystem. They are
    # read-only — no event appended, no hook fired. Writes still go through
    # ``@actions``.

    async def _execute_ls(self, call: ToolCall, reply_to: Any) -> Message:
        args = call.arguments or {}
        path = str(args.get("path", "") or "")
        entries = list_dir(self.namespace, path)
        result_text = render_ls(entries)
        return await self._meta_tool_response(call, reply_to, result_text)

    async def _execute_read(self, call: ToolCall, reply_to: Any) -> Message:
        args = call.arguments or {}
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return await self._meta_tool_response(
                call, reply_to,
                "Invalid arguments for read: 'path' is required",
                is_error=True,
            )
        offset = _coerce_optional_int(args.get("offset"))
        limit = _coerce_optional_int(args.get("limit"))
        content, meta = read_doc(self.namespace, path, offset=offset, limit=limit)
        result_text = render_read(content, path, meta)
        return await self._meta_tool_response(call, reply_to, result_text)

    async def _execute_grep(self, call: ToolCall, reply_to: Any) -> Message:
        args = call.arguments or {}
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return await self._meta_tool_response(
                call, reply_to,
                "Invalid arguments for grep: 'pattern' is required",
                is_error=True,
            )
        path = str(args.get("path", "") or "")
        glob = args.get("glob")
        glob = glob if isinstance(glob, str) and glob else None
        ignore_case = bool(args.get("ignore_case", False))
        limit = _coerce_optional_int(args.get("limit")) or DEFAULT_GREP_LIMIT
        matches = grep_docs(
            self.namespace, pattern,
            path=path, glob=glob, ignore_case=ignore_case, limit=limit,
        )
        result_text = render_grep(matches, limit=limit)
        return await self._meta_tool_response(call, reply_to, result_text)

    async def _meta_tool_response(
        self,
        call: ToolCall,
        reply_to: Any,
        result_text: str,
        is_error: bool = False,
    ) -> Message:
        """Shared envelope for read-only meta-tools."""
        if reply_to:
            payload: dict[str, Any] = {"name": call.name, "id": call.id}
            payload["error" if is_error else "result"] = result_text
            await reply_to.emit(OutputEvent(
                kind=OutputEventKind.ACTION_RESULT, payload=payload,
            ))
        return Message(
            role=Role.TOOL,
            content=result_text,
            tool_call_id=call.id,
            name=call.name,
        )

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
        schemas.extend(_subagent_tool_schemas(self.subagents, world.state, event))
        schemas.append(_ls_tool_schema())
        schemas.append(_read_tool_schema())
        schemas.append(_grep_tool_schema())
        schemas.append(_search_event_log_tool_schema())
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
    """
    sig = inspect.signature(method)
    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )
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


def _subagent_tool_schemas(
    registry: SubagentRegistry,
    state: Any,
    event: TriggerEvent | None,
) -> list[ToolSchema]:
    """One ``consult_<name>`` schema per registered subagent (filtered by show_when)."""
    schemas: list[ToolSchema] = []
    for sub in registry.list():
        predicate = getattr(sub, "show_when", None)
        if predicate is not None:
            try:
                if not bool(predicate(state, event)):
                    continue
            except Exception:  # noqa: BLE001
                log.exception("subagent %r show_when raised; hiding", sub.name)
                continue
        schemas.append(ToolSchema(
            name=f"{_SUBAGENT_TOOL_PREFIX}{sub.name}",
            description=(
                f"{sub.description}\n\n"
                "Consult the external specialist. Pass session_id from a "
                "prior response to continue the same conversation context; "
                "omit it to start fresh."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The prompt to send the subagent.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": (
                            "Optional A2A contextId from a prior response. "
                            "When set, the subagent resumes that conversation."
                        ),
                    },
                },
                "required": ["message"],
            },
        ))
    return schemas


def _search_event_log_tool_schema() -> ToolSchema:
    """The built-in meta-tool every runtime exposes."""
    return ToolSchema(
        name=_SEARCH_EVENT_LOG_TOOL,
        description=(
            "Search this project's event log. All filters are optional and "
            "compose with AND. Use this when Recent Activity in the system "
            "prompt is too short to answer a question about prior activity — "
            "for example to find a prior consult_<subagent> session_id, or "
            "to confirm whether a particular action has already been taken."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action_name_glob": {
                    "type": "string",
                    "description": (
                        "fnmatch glob over action names. "
                        "Examples: 'consult_*' (all subagent calls), "
                        "'consult_legal_research' (one specific subagent), "
                        "'notify_*' (everything starting with notify_)."
                    ),
                },
                "grep": {
                    "type": "string",
                    "description": "Case-insensitive substring over the rendered line.",
                },
                "actor": {
                    "type": "string",
                    "description": "Exact match on actor (e.g. 'agent', 'human').",
                },
                "since": {
                    "type": "string",
                    "description": "ISO 8601 timestamp — only events at or after this time.",
                },
                "until": {
                    "type": "string",
                    "description": "ISO 8601 timestamp — only events before this time.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results. Default 20.",
                },
                "offset_from": {
                    "type": "integer",
                    "description": "Only events at log offset >= this value (for pagination).",
                },
            },
            "required": [],
        },
    )


# --------------------------------------------------------------------------- #
# Namespace navigation tool schemas (ls / read / grep)                         #
# --------------------------------------------------------------------------- #

def _ls_tool_schema() -> ToolSchema:
    return ToolSchema(
        name=_LS_TOOL,
        description=(
            "List entries under a path in the project namespace. The "
            "namespace is the agent's virtual filesystem: project state, "
            "logs, subagent artefacts. Returns names with a '/' suffix for "
            "subdirectories. Use this to discover documents that are not "
            "already lifted into the system prompt."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path prefix to list. Use the project id (e.g. "
                        "'demo/') to see the top of the project, or a "
                        "subdirectory like 'demo/artefacts/'. Defaults to "
                        "the namespace root."
                    ),
                },
            },
            "required": [],
        },
    )


def _read_tool_schema() -> ToolSchema:
    return ToolSchema(
        name=_READ_TOOL,
        description=(
            "Read a text document from the project namespace. Use this to "
            "open documents found via `ls` or `grep` that are not lifted "
            "into the system prompt — e.g. the full `event_log.md`, the "
            "raw `audit.jsonl`, or any artefact dropped under "
            "<project>/artefacts/. Reading does not change project state."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Full path of the document.",
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "1-indexed starting line. Use with `limit` to "
                        "paginate large documents."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum lines to return. Defaults to "
                        f"{DEFAULT_READ_LIMIT}; the response indicates "
                        "whether content was truncated and how to continue."
                    ),
                },
            },
            "required": ["path"],
        },
    )


def _grep_tool_schema() -> ToolSchema:
    return ToolSchema(
        name=_GREP_TOOL,
        description=(
            "Search document content across the project namespace. The "
            "pattern is a Python regex (a bad pattern falls back to literal "
            "substring search). Use this to find references — names, IDs, "
            "specific topics — across all project documents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex (or literal substring) to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Restrict the search to documents whose path starts "
                        "with this prefix. Defaults to the namespace root."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Optional fnmatch glob over full document paths "
                        "(e.g. '*.md', '*/artefacts/*')."
                    ),
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive match. Default False.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of matches. Default "
                        f"{DEFAULT_GREP_LIMIT}."
                    ),
                },
            },
            "required": ["pattern"],
        },
    )


def _coerce_optional_int(value: Any) -> int | None:
    """Coerce LLM-supplied numeric kwargs into ``int | None`` defensively."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Subagent response rendering + small utilities                                #
# --------------------------------------------------------------------------- #

def _render_subagent_response(
    subagent_name: str,
    response: SubagentResponse,
    artefact_records: list[dict] | None = None,
) -> str:
    """Format a SubagentResponse as the TOOL-message body the LLM reads.

    The status word is at the start so the LLM can branch on it quickly.
    For ``input-required`` we tell the model exactly what to do — call
    consult_<name> again with the same session_id. Persisted artefact
    paths are surfaced inline so the LLM can ``read`` them immediately.
    """
    parts = [f"[{response.status}]"]
    if response.session_id:
        parts.append(f"session_id={response.session_id}")
    parts.append(response.content or "")
    body = " ".join(p for p in parts if p)
    if artefact_records:
        bullet_lines = [
            f"- `{a['path']}` — {a['name']}"
            + (f" ({a['mime']})" if a.get("mime") else "")
            + (f" — {a['description']}" if a.get("description") else "")
            for a in artefact_records
        ]
        body += (
            "\n\nArtefacts written to the project namespace:\n"
            + "\n".join(bullet_lines)
            + "\nUse `read` on any of the paths above to inspect them."
        )
    if response.status == "input-required":
        body += (
            f"\n\nTo continue, call `consult_{subagent_name}` again with "
            f"`session_id=\"{response.session_id}\"` and your reply as `message`."
        )
    return body


# --------------------------------------------------------------------------- #
# Subagent artefact persistence                                                #
# --------------------------------------------------------------------------- #

_ARTEFACT_DIR = "artefacts"
_SANITISE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _persist_artefacts(
    namespace: NamespaceAdapter,
    project_id: str,
    artefacts: list[Artefact],
) -> list[dict]:
    """Write each artefact to the project namespace.

    Paths are content-addressable: ``<project>/artefacts/<sha[:8]>__<name>``.
    This decouples the path from the event offset (so CAS retries don't
    relocate files) while keeping the name human-readable. Identical
    bytes dedupe naturally.

    Returns one record per artefact: ``{path, name, size, mime, sha256,
    description}``. The runtime threads these into the consult event and
    the TOOL response.
    """
    records: list[dict] = []
    for art in artefacts:
        sha = hashlib.sha256(art.content).hexdigest()
        safe_name = _sanitise_artefact_name(art.name)
        path = f"{project_id}/{_ARTEFACT_DIR}/{sha[:8]}__{safe_name}"
        # write_binary may raise NotImplementedError on text-only adapters
        # — that's a legitimate operator error; surface it rather than
        # silently dropping the artefact.
        namespace.write_binary(path, art.content)
        mime = art.mime
        if mime is None:
            info = namespace.doc_info(path)
            mime = info.mime if info is not None else None
        records.append({
            "path": path,
            "name": art.name,
            "size": len(art.content),
            "mime": mime,
            "sha256": sha,
            "description": art.description,
        })
    return records


def _sanitise_artefact_name(name: str) -> str:
    """Replace path-unsafe characters with ``_`` while keeping the extension.

    Trims to a sane length so the path stays well below filesystem limits
    on the various namespace backends we plan to support (S3, SharePoint).
    """
    cleaned = _SANITISE_RE.sub("_", name).strip("._")
    if not cleaned:
        cleaned = "artefact.bin"
    if len(cleaned) > 80:
        # Keep the extension; truncate the stem.
        if "." in cleaned:
            stem, _, ext = cleaned.rpartition(".")
            cleaned = stem[: 80 - len(ext) - 1] + "." + ext
        else:
            cleaned = cleaned[:80]
    return cleaned


def _truncate_for_log(text: str, max_len: int = 80) -> str:
    """Trim long strings for compact event-log entries."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _parse_iso_datetime(raw: Any) -> datetime | None:
    """Parse an ISO-8601 string; return None for empty/None inputs."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"expected ISO-8601 string, got {type(raw).__name__}")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
