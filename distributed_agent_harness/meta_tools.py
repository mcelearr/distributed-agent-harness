"""
Meta-tools — the always-on, read-only tools every runtime surfaces to the
LLM alongside ``@actions`` and ``consult_<name>`` subagents.

Four tools:

- ``ls`` / ``read`` / ``grep`` — virtual-filesystem navigation over the
  project ``NamespaceAdapter`` (the read side of the filesystem-as-world
  pattern; writes still go through ``@actions``).
- ``search_event_log`` — queryable view over the project event log.

All four are stateless from the runtime's perspective: they only read
from the namespace / event log and never append events or fire hooks.
The runtime delegates here so ``runtime.py`` doesn't have to know how
each meta-tool builds its schema or formats its result.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .event_search import EventQuery, render_events_markdown, search_events
from .llm import Message, Role, ToolCall, ToolSchema
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
from .transport import OutputEvent, OutputEventKind

if TYPE_CHECKING:
    from .eventlog import EventLog
    from .namespace import NamespaceAdapter


# --------------------------------------------------------------------------- #
# Reserved tool names                                                          #
# --------------------------------------------------------------------------- #

SEARCH_EVENT_LOG_TOOL = "search_event_log"
LS_TOOL = "ls"
READ_TOOL = "read"
GREP_TOOL = "grep"

#: The full set the runtime treats as reserved (an ``@action`` with one
#: of these names would be silently shadowed by the meta-tool dispatch).
META_TOOL_NAMES: frozenset[str] = frozenset({
    SEARCH_EVENT_LOG_TOOL, LS_TOOL, READ_TOOL, GREP_TOOL,
})


# --------------------------------------------------------------------------- #
# Tool schemas                                                                 #
# --------------------------------------------------------------------------- #

def ls_tool_schema() -> ToolSchema:
    return ToolSchema(
        name=LS_TOOL,
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


def read_tool_schema() -> ToolSchema:
    return ToolSchema(
        name=READ_TOOL,
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


def grep_tool_schema() -> ToolSchema:
    return ToolSchema(
        name=GREP_TOOL,
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


def search_event_log_tool_schema() -> ToolSchema:
    """The built-in event-log search meta-tool every runtime exposes."""
    return ToolSchema(
        name=SEARCH_EVENT_LOG_TOOL,
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


def all_meta_tool_schemas() -> list[ToolSchema]:
    """The four always-on schemas in their canonical order."""
    return [
        ls_tool_schema(),
        read_tool_schema(),
        grep_tool_schema(),
        search_event_log_tool_schema(),
    ]


# --------------------------------------------------------------------------- #
# Dispatchers                                                                  #
# --------------------------------------------------------------------------- #

async def execute_ls(
    call: ToolCall,
    reply_to: Any,
    namespace: "NamespaceAdapter",
) -> Message:
    """Handle one ``ls`` tool call."""
    args = call.arguments or {}
    path = str(args.get("path", "") or "")
    entries = list_dir(namespace, path)
    result_text = render_ls(entries)
    return await _emit_and_return(call, reply_to, result_text)


async def execute_read(
    call: ToolCall,
    reply_to: Any,
    namespace: "NamespaceAdapter",
) -> Message:
    """Handle one ``read`` tool call."""
    args = call.arguments or {}
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return await _emit_and_return(
            call, reply_to,
            "Invalid arguments for read: 'path' is required",
            is_error=True,
        )
    offset = _coerce_optional_int(args.get("offset"))
    limit = _coerce_optional_int(args.get("limit"))
    content, meta = read_doc(namespace, path, offset=offset, limit=limit)
    result_text = render_read(content, path, meta)
    return await _emit_and_return(call, reply_to, result_text)


async def execute_grep(
    call: ToolCall,
    reply_to: Any,
    namespace: "NamespaceAdapter",
) -> Message:
    """Handle one ``grep`` tool call."""
    args = call.arguments or {}
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return await _emit_and_return(
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
        namespace, pattern,
        path=path, glob=glob, ignore_case=ignore_case, limit=limit,
    )
    result_text = render_grep(matches, limit=limit)
    return await _emit_and_return(call, reply_to, result_text)


async def execute_search_event_log(
    call: ToolCall,
    reply_to: Any,
    eventlog: "EventLog",
    project_id: str,
) -> Message:
    """Handle one ``search_event_log`` tool call."""
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
        return await _emit_and_return(
            call, reply_to,
            f"Invalid arguments for search_event_log: {exc}",
            is_error=True,
        )

    events = await search_events(eventlog, project_id, query)
    result_text = render_events_markdown(events)
    return await _emit_and_return(call, reply_to, result_text)


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #

async def _emit_and_return(
    call: ToolCall,
    reply_to: Any,
    result_text: str,
    is_error: bool = False,
) -> Message:
    """Shared envelope: emit ACTION_RESULT (if a channel is present) then
    return the TOOL message the runtime threads back into the conversation.
    """
    if reply_to is not None:
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


def _coerce_optional_int(value: Any) -> int | None:
    """Coerce LLM-supplied numeric kwargs into ``int | None`` defensively."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
