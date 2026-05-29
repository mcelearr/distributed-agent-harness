"""In-process subagents — run inside the harness, no HTTP+SSE boundary.

The LLM-facing interface is identical to ``A2ASubagent``: each in-process
subagent is registered on the runtime and surfaces as a
``consult_<name>(message, session_id)`` tool. Only the transport differs
— ``AsyncSubagent`` wraps an async function on the same loop;
``MessagingSubagent`` publishes a correlated request on a ``MessageBus``
and awaits the response.

Neither variant mutates project state directly. Anything an in-process
subagent wants to persist comes back inside its ``SubagentResponse`` —
text content lands in the consult event's result_summary, binary
artefacts are written to the project namespace at content-addressable
paths by the runtime.
"""

from .async_subagent import AsyncSubagent, InProcessSubagent
from .messaging import (
    InMemoryMessageBus,
    MessageBus,
    MessagingSubagent,
    MessagingSubagentWorker,
    NoSubscriberError,
    Subscription,
)

__all__ = [
    "InProcessSubagent",
    "AsyncSubagent",
    "MessageBus",
    "InMemoryMessageBus",
    "MessagingSubagent",
    "MessagingSubagentWorker",
    "NoSubscriberError",
    "Subscription",
]
