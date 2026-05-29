"""Subagent support — pluggable external specialists callable as tools."""

from .a2a import A2ASubagent
from .base import (
    AgentCard,
    Artefact,
    Skill,
    SubagentClient,
    SubagentRegistry,
    SubagentResponse,
    SubagentTimeout,
)
from .inprocess import (
    AsyncSubagent,
    InMemoryMessageBus,
    InProcessSubagent,
    MessageBus,
    MessagingSubagent,
    MessagingSubagentWorker,
    NoSubscriberError,
    Subscription,
)
from .registry import (
    AgentRegistry,
    HttpAgentRegistry,
    StaticAgentRegistry,
    load_subagents_from_registry,
)

__all__ = [
    "AgentCard",
    "Artefact",
    "Skill",
    "SubagentClient",
    "SubagentRegistry",
    "SubagentResponse",
    "SubagentTimeout",
    "A2ASubagent",
    "AgentRegistry",
    "StaticAgentRegistry",
    "HttpAgentRegistry",
    "load_subagents_from_registry",
    # In-process subagents
    "InProcessSubagent",
    "AsyncSubagent",
    "MessageBus",
    "InMemoryMessageBus",
    "MessagingSubagent",
    "MessagingSubagentWorker",
    "NoSubscriberError",
    "Subscription",
]
