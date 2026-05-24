"""
AgentRuntime — the LLM loop that ties everything together.

Takes a WorldEnvironment class plus a pluggable LLMProvider and runs a
ReAct-style loop:

    LLM ──tool calls──► WorldEnvironment @action methods
        ◄──results───
    LLM ──final message──► OutputChannel

The runtime is fully async. Tool calls execute via ``asyncio.to_thread`` so
the sync @action machinery (lock / hydrate / execute / flush / release)
doesn't block the event loop.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Type

from pydantic import ValidationError, create_model

from .concurrency import ConcurrencyHandler
from .llm import LLMProvider, Message, Role, ToolCall, ToolSchema
from .namespace import NamespaceAdapter
from .prompt_builder import PromptBuilder
from .transport import OutputEvent, OutputEventKind, TriggerEvent
from .world import BaseWorldEnvironment

log = logging.getLogger(__name__)


DEFAULT_SYSTEM_PREAMBLE = """\
You are an agent operating on a shared, audited project state via a fixed
set of actions. Each action mutates the project's persisted state.

Rules:
- Use ONLY the actions listed under "Available Actions". You have no shell access.
- The current project state is shown under "Current World State". It is the
  authoritative source of truth.
- Call as many actions as you need to satisfy the user's request, then respond
  with a final plain-text message (no further tool calls) summarising what you did.
- If you do not have enough information to act safely, ask the user instead of guessing.
"""


class AgentRuntime:
    """
    Drives the LLM ↔ WorldEnvironment interaction for a single trigger event.

    Usage::

        runtime = AgentRuntime(
            world_class=DataProtectionWorldEnvironment,
            namespace=InMemoryNamespace(),
            concurrency=InProcessLock(),
            llm=MistralProvider(),
        )

        async for event in cli_chat.events():
            await runtime.handle(event)
    """

    def __init__(
        self,
        world_class: Type[BaseWorldEnvironment],
        namespace: NamespaceAdapter,
        concurrency: ConcurrencyHandler,
        llm: LLMProvider,
        max_iterations: int = 12,
        system_preamble: str = DEFAULT_SYSTEM_PREAMBLE,
        include_source_in_prompt: bool = True,
    ) -> None:
        self.world_class = world_class
        self.namespace = namespace
        self.concurrency = concurrency
        self.llm = llm
        self.max_iterations = max_iterations
        self.system_preamble = system_preamble
        self._builder = PromptBuilder(world_class, include_source=include_source_in_prompt)
        # Cache tool schemas — they're derived from class definitions, not state
        self._tool_schemas, self._param_models = _build_tool_schemas(world_class)

    # ----------------------------------------------------------------------- #
    # Public entry point                                                       #
    # ----------------------------------------------------------------------- #

    async def handle(self, event: TriggerEvent) -> Message:
        """
        Process one TriggerEvent end-to-end.

        Builds an agent loop, dispatches tool calls to WorldEnvironment
        actions, streams events to ``event.reply_to`` if present, and returns
        the final assistant message.
        """
        world = self.world_class(
            project_id=event.project_id,
            namespace=self.namespace,
            concurrency=self.concurrency,
        )
        messages = self._initial_messages(world, event)
        reply_to = event.reply_to

        final: Message = Message(role=Role.ASSISTANT, content="")

        try:
            for _ in range(self.max_iterations):
                assistant = await self.llm.chat_complete(messages, tools=self._tool_schemas)
                messages.append(assistant)

                if not assistant.tool_calls:
                    final = assistant
                    if reply_to and assistant.content:
                        await reply_to.emit(OutputEvent(
                            kind=OutputEventKind.MESSAGE,
                            payload={"content": assistant.content},
                        ))
                    break

                # Execute each tool call and append results
                for call in assistant.tool_calls:
                    result_message = await self._execute_call(call, world, reply_to)
                    messages.append(result_message)
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

        return final

    # ----------------------------------------------------------------------- #
    # Internals                                                                #
    # ----------------------------------------------------------------------- #

    def _initial_messages(
        self, world: BaseWorldEnvironment, event: TriggerEvent
    ) -> list[Message]:
        system_content = (
            self.system_preamble
            + "\n\n"
            + self._builder.build_full_prompt(world)
        )
        messages = [Message(role=Role.SYSTEM, content=system_content)]

        # Convert the trigger payload into a user message. Chat messages have
        # 'text'; other trigger kinds get a structured description.
        text = event.payload.get("text")
        if text:
            messages.append(Message(role=Role.USER, content=text))
        else:
            messages.append(Message(
                role=Role.USER,
                content=(
                    f"Trigger received from '{event.source}' of kind "
                    f"'{event.kind.value}'. Payload: {event.payload}"
                ),
            ))
        return messages

    async def _execute_call(
        self,
        call: ToolCall,
        world: BaseWorldEnvironment,
        reply_to: Any,
    ) -> Message:
        """Run one tool call against the WorldEnvironment and return a TOOL message."""
        if reply_to:
            await reply_to.emit(OutputEvent(
                kind=OutputEventKind.ACTION_CALLED,
                payload={"name": call.name, "args": call.arguments, "id": call.id},
            ))

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

        # The @action wrapper handles its own locking; we offload to a thread
        # so we don't block the event loop.
        try:
            result = await asyncio.to_thread(method, **kwargs)
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
        except Exception as exc:  # noqa: BLE001 — must surface to the LLM
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
