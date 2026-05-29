"""Subagent support — pluggable external specialists callable as tools."""

from .a2a import A2ASubagent
from .base import (
    AgentCard,
    Skill,
    SubagentClient,
    SubagentRegistry,
    SubagentResponse,
    SubagentTimeout,
)
from .registry import (
    AgentRegistry,
    HttpAgentRegistry,
    StaticAgentRegistry,
    load_subagents_from_registry,
)

__all__ = [
    "AgentCard",
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
]
