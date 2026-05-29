"""Distributed Agent Harness — shared, auditable world state for multi-agent systems."""

from .conflict import (
    Abandon,
    AgentDrivenConflictResolver,
    AlwaysRecoverResolver,
    ConcurrentUpdate,
    ConflictContext,
    ConflictResolver,
    Continue,
    Decision,
    Recover,
    ScriptedResolver,
)
from .event_search import EventQuery, render_events_markdown, search_events
from .eventlog import (
    Appended,
    AppendResult,
    Conflict,
    Event,
    EventLog,
    InMemoryEventLog,
)
from .hooks import (
    ActionContext,
    ActionErrorHook,
    BlockDecision,
    HookRegistry,
    PostActionHook,
    PostSubagentCallHook,
    PreActionHook,
    PreSubagentCallHook,
    PreTriggerHook,
    RunCompleteHook,
    SubagentContext,
)
from .llm import (
    CompletionChunk,
    LLMProvider,
    Message,
    Role,
    ToolCall,
    ToolSchema,
)
from .namespace import NamespaceAdapter
from .prompt_builder import PromptBuilder
from .runtime import AgentRuntime
from .subagents import (
    A2ASubagent,
    AgentCard,
    AgentRegistry,
    HttpAgentRegistry,
    Skill,
    StaticAgentRegistry,
    SubagentClient,
    SubagentRegistry,
    SubagentResponse,
    SubagentTimeout,
    load_subagents_from_registry,
)
from .transport import (
    OutputChannel,
    OutputEvent,
    OutputEventKind,
    TriggerEvent,
    TriggerKind,
    TriggerSource,
)
from .world import ActionNotAvailable, BaseWorldEnvironment, Predicate, action

__all__ = [
    # Core
    "BaseWorldEnvironment",
    "action",
    "Predicate",
    "ActionNotAvailable",
    "NamespaceAdapter",
    "PromptBuilder",
    # Event sourcing
    "Event",
    "EventLog",
    "InMemoryEventLog",
    "Appended",
    "Conflict",
    "AppendResult",
    # Conflict resolution
    "ConcurrentUpdate",
    "ConflictContext",
    "ConflictResolver",
    "AgentDrivenConflictResolver",
    "AlwaysRecoverResolver",
    "ScriptedResolver",
    "Decision",
    "Continue",
    "Recover",
    "Abandon",
    # LLM
    "LLMProvider",
    "Message",
    "Role",
    "ToolCall",
    "ToolSchema",
    "CompletionChunk",
    # Transport
    "TriggerEvent",
    "TriggerKind",
    "TriggerSource",
    "OutputEvent",
    "OutputEventKind",
    "OutputChannel",
    # Hooks
    "HookRegistry",
    "ActionContext",
    "SubagentContext",
    "BlockDecision",
    "PreActionHook",
    "PostActionHook",
    "ActionErrorHook",
    "PreTriggerHook",
    "RunCompleteHook",
    "PreSubagentCallHook",
    "PostSubagentCallHook",
    # Event search
    "EventQuery",
    "search_events",
    "render_events_markdown",
    # Subagents
    "SubagentClient",
    "SubagentResponse",
    "SubagentRegistry",
    "SubagentTimeout",
    "AgentCard",
    "Skill",
    "A2ASubagent",
    "AgentRegistry",
    "StaticAgentRegistry",
    "HttpAgentRegistry",
    "load_subagents_from_registry",
    # Runtime
    "AgentRuntime",
]
