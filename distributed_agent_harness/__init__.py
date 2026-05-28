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
    PreActionHook,
    PreTriggerHook,
    RunCompleteHook,
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
from .transport import (
    OutputChannel,
    OutputEvent,
    OutputEventKind,
    TriggerEvent,
    TriggerKind,
    TriggerSource,
)
from .world import BaseWorldEnvironment, Predicate, PreconditionViolation, action

__all__ = [
    # Core
    "BaseWorldEnvironment",
    "action",
    "Predicate",
    "PreconditionViolation",
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
    "BlockDecision",
    "PreActionHook",
    "PostActionHook",
    "ActionErrorHook",
    "PreTriggerHook",
    "RunCompleteHook",
    # Runtime
    "AgentRuntime",
]
