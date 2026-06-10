"""
Subagent dispatch — the runtime's ``consult_<name>`` tool-call handling.

Sits between ``runtime.AgentRuntime`` and the concrete ``SubagentClient``
implementations. Owns:

- Argument validation (``message`` required, optional ``session_id``).
- The hook lifecycle (``pre_subagent_call`` may block, ``post_subagent_call``
  always fires on success or failure).
- Progress-callback wiring so ``working`` deltas land on the
  ``OutputChannel`` as ``THINKING`` events.
- Artefact persistence: bytes returned in ``SubagentResponse.artefacts``
  are written to the project namespace at a content-addressable path
  (``<project>/artefacts/<sha256[:8]>__<sanitised_name>``) and the paths
  are recorded both in the consult event's ``result_summary`` (so
  ``search_event_log`` and ``grep`` find them later) and in the TOOL
  message the LLM reads (so the next turn can ``read`` them immediately).
- Event-log appends for every consult, with transparent CAS retry —
  subagent calls are commutative w.r.t. concurrent ``@actions``, so we
  never surface a conflict.

The runtime stays small by delegating here; the subagent flow stays
testable in isolation because everything it touches comes in via
parameters.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

from ..eventlog import Appended, Event
from ..hooks import HookRegistry, RunBudget, SubagentContext
from ..identity import AgentIdentity
from ..llm import Message, Role, ToolCall, ToolSchema
from ..transport import OutputEvent, OutputEventKind, TriggerEvent
from .base import (
    Artefact,
    SubagentClient,
    SubagentRegistry,
    SubagentResponse,
    SubagentTimeout,
)

if TYPE_CHECKING:
    from ..eventlog import EventLog
    from ..namespace import NamespaceAdapter
    from ..world import BaseWorldEnvironment

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reserved naming                                                              #
# --------------------------------------------------------------------------- #

SUBAGENT_TOOL_PREFIX = "consult_"

_ARTEFACT_DIR = "artefacts"
_SANITISE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_LOG_TRUNCATE_LEN = 80


# --------------------------------------------------------------------------- #
# Tool schemas                                                                 #
# --------------------------------------------------------------------------- #

def subagent_tool_schemas(
    registry: SubagentRegistry,
    state: Any,
    event: TriggerEvent | None,
) -> list[ToolSchema]:
    """One ``consult_<name>`` schema per registered subagent (``show_when``-filtered).

    A subagent with no ``show_when`` (or one whose predicate returns True)
    is exposed to the LLM. A buggy predicate that raises is treated as
    False and the subagent is hidden — same defensive shape as
    ``@action`` predicates.
    """
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
            name=f"{SUBAGENT_TOOL_PREFIX}{sub.name}",
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


# --------------------------------------------------------------------------- #
# Dispatch                                                                     #
# --------------------------------------------------------------------------- #

async def execute_subagent_call(
    call: ToolCall,
    subagent_name: str,
    subagent: SubagentClient,
    world: "BaseWorldEnvironment",
    reply_to: Any,
    trigger: TriggerEvent,
    *,
    hooks: HookRegistry,
    eventlog: "EventLog",
    namespace: "NamespaceAdapter",
    identity: AgentIdentity | None = None,
    budget: RunBudget | None = None,
) -> Message:
    """Dispatch one ``consult_<name>`` tool call.

    All side effects flow through the injected dependencies. The runtime
    can call this without exposing its own internals; tests can call it
    with stand-in objects.
    """
    message = call.arguments.get("message") if call.arguments else None
    session_id_raw = call.arguments.get("session_id") if call.arguments else None
    if not isinstance(message, str) or not message:
        return await _emit_error_message(
            call, reply_to,
            f"Invalid arguments for {call.name}: 'message' is required",
        )

    sub_ctx = SubagentContext(
        project_id=world._project_id,
        subagent_name=subagent_name,
        message=message,
        session_id=session_id_raw if isinstance(session_id_raw, str) else None,
        trigger=trigger,
        identity=identity,
        trigger_id=trigger.id,
        budget=budget,
    )

    # ----- pre_subagent_call hook
    decision = await hooks.fire_pre_subagent_call(sub_ctx)
    if decision is not None:
        return await _emit_blocked_message(
            call, reply_to, f"Subagent blocked: {decision.reason}",
        )

    # ----- progress callback: forward `working` text deltas as THINKING events
    async def _on_progress(delta: str) -> None:
        if reply_to is not None:
            await reply_to.emit(OutputEvent(
                kind=OutputEventKind.THINKING,
                payload={"subagent": subagent_name, "delta": delta},
            ))

    # ----- the consult itself
    try:
        response: SubagentResponse = await subagent.consult(
            message=message,
            session_id=sub_ctx.session_id,
            on_progress=_on_progress,
        )
    except SubagentTimeout as exc:
        await _append_subagent_event(
            eventlog, world._project_id, subagent_name, message,
            sub_ctx.session_id, status="failed", content=f"timeout: {exc}",
            identity=identity, trigger_id=trigger.id,
        )
        return await _emit_error_message(
            call, reply_to, f"Subagent timed out: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 — surface to LLM
        await _append_subagent_event(
            eventlog, world._project_id, subagent_name, message,
            sub_ctx.session_id, status="failed", content=str(exc),
            identity=identity, trigger_id=trigger.id,
        )
        return await _emit_error_message(
            call, reply_to,
            f"Subagent error: {type(exc).__name__}: {exc}",
        )

    # ----- persist artefacts before recording the event so the event's
    # result_summary can include their paths (grep-discoverable later)
    artefact_records = persist_artefacts(
        namespace, world._project_id, response.artefacts,
    )

    # ----- record the consult event
    await _append_subagent_event(
        eventlog, world._project_id, subagent_name, message,
        sub_ctx.session_id,
        status=response.status,
        content=response.content,
        returned_session_id=response.session_id,
        artefact_records=artefact_records,
        identity=identity,
        trigger_id=trigger.id,
    )

    # ----- post_subagent_call hook
    await hooks.fire_post_subagent_call(sub_ctx, response)

    result_text = render_subagent_response(
        subagent_name, response, artefact_records,
    )
    if reply_to is not None:
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


# --------------------------------------------------------------------------- #
# Response rendering                                                           #
# --------------------------------------------------------------------------- #

def render_subagent_response(
    subagent_name: str,
    response: SubagentResponse,
    artefact_records: list[dict] | None = None,
) -> str:
    """Format a SubagentResponse as the TOOL-message body the LLM reads.

    Status at the front so the LLM can branch on it quickly. Persisted
    artefact paths are bulleted with a ``Use `read` ...`` hint. For
    ``input-required`` we tell the LLM exactly how to follow up.
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
# Artefact persistence                                                         #
# --------------------------------------------------------------------------- #

def persist_artefacts(
    namespace: "NamespaceAdapter",
    project_id: str,
    artefacts: list[Artefact],
) -> list[dict]:
    """Write each artefact to the project namespace.

    Paths are content-addressable: ``<project>/artefacts/<sha[:8]>__<name>``.
    Decoupling from the event offset means CAS retries never relocate
    files; identical bytes dedupe naturally. Returns one record per
    artefact: ``{path, name, size, mime, sha256, description}``.
    """
    records: list[dict] = []
    for art in artefacts:
        sha = hashlib.sha256(art.content).hexdigest()
        safe_name = sanitise_artefact_name(art.name)
        path = f"{project_id}/{_ARTEFACT_DIR}/{sha[:8]}__{safe_name}"
        # write_binary may raise NotImplementedError on text-only adapters
        # — that's a legitimate operator error; let it propagate rather
        # than silently dropping the artefact.
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


def sanitise_artefact_name(name: str) -> str:
    """Replace path-unsafe characters with ``_``, keep the extension,
    trim to a sane length.

    Sane length keeps paths well below the limits of every namespace
    backend we plan to support (S3 keys, SharePoint paths, local fs).
    """
    cleaned = _SANITISE_RE.sub("_", name).strip("._")
    if not cleaned:
        cleaned = "artefact.bin"
    if len(cleaned) > 80:
        if "." in cleaned:
            stem, _, ext = cleaned.rpartition(".")
            cleaned = stem[: 80 - len(ext) - 1] + "." + ext
        else:
            cleaned = cleaned[:80]
    return cleaned


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #

async def _append_subagent_event(
    eventlog: "EventLog",
    project_id: str,
    subagent_name: str,
    message: str,
    session_id: str | None,
    status: str,
    content: str,
    returned_session_id: str | None = None,
    artefact_records: list[dict] | None = None,
    identity: AgentIdentity | None = None,
    trigger_id: str | None = None,
) -> None:
    """Append a ``consult_<name>`` event with transparent CAS retry.

    Subagent consults don't depend on the world's state, so any
    intervening ``@action`` writes cannot invalidate them — we keep
    trying until the append wins. No conflict is surfaced to the LLM.
    Artefact paths are inlined into ``result_summary`` so the LLM can
    rediscover them later via ``search_event_log`` or ``grep``.
    """
    summary_parts = [
        f"status={status}",
        f"session={returned_session_id or session_id}",
        f"content={_truncate_for_log(content)}",
    ]
    if artefact_records:
        paths_inline = ", ".join(a["path"] for a in artefact_records)
        summary_parts.append(f"artefacts=[{paths_inline}]")

    actor = identity.label if identity is not None else "agent"
    event = Event(
        project_id=project_id,
        action_name=f"{SUBAGENT_TOOL_PREFIX}{subagent_name}",
        args=[],
        kwargs={
            "message": _truncate_for_log(message),
            "session_id": session_id,
        },
        actor=actor,
        result_summary=" ".join(summary_parts),
        identity=identity,
        trigger_id=trigger_id,
    )
    while True:
        offset = await eventlog.current_offset(project_id)
        result = await eventlog.append(
            project_id, event, expected_offset=offset,
        )
        if isinstance(result, Appended):
            return
        # Conflict — refresh and try again. Subagent calls are
        # commutative w.r.t. any concurrent @action.


async def _emit_error_message(
    call: ToolCall,
    reply_to: Any,
    error: str,
) -> Message:
    if reply_to is not None:
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


async def _emit_blocked_message(
    call: ToolCall,
    reply_to: Any,
    error: str,
) -> Message:
    if reply_to is not None:
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


def _truncate_for_log(text: str, max_len: int = _LOG_TRUNCATE_LEN) -> str:
    """Trim long strings for compact event-log entries."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
