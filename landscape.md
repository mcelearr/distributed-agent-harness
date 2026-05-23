# Distributed Agent Harness — Open Source Landscape

This document surveys major open-source agent frameworks and assesses how well each maps to the three core requirements of the Distributed Agent Harness:

1. **Shared memory** — state must live in a location accessible by all actors (not local filesystem)
2. **Concurrent actors** — multiple agents and humans must be able to read/write memory simultaneously
3. **Controlled, auditable tool execution** — generic bash/shell commands replaced with specific, named, schema-validated, logged actions

---

## 1. LangGraph (LangChain)

**GitHub:** https://github.com/langchain-ai/langgraph | ~10,000+ stars (standalone repo, rapidly growing)

LangGraph models agent workflows as directed cyclic graphs over a shared typed state object. Every node reads and writes to this central state, making state the primary communication medium. It is the dominant substrate for "deep research" agent patterns in 2024–2025.

### Memory / State

- **Thread-scoped checkpoints**: Full graph state is snapshotted at every step to a configurable backend (SQLite, PostgreSQL, Redis, MongoDB). PostgreSQL and Redis checkpointers are production-grade and officially maintained.
- **Cross-thread Store**: Separate `BaseStore` interface provides namespaced, queryable key-value storage that persists across sessions. Backends include Redis, Postgres, MongoDB with optional vector search.

**Fit for Requirement 1:** Excellent. Any agent instance connecting to the same database sees the same state.

### Concurrent Actors

- `interrupt()` primitive pauses graph execution at any node, serializes state to the checkpointer, and waits indefinitely. Execution resumes when any caller (human via UI or another agent) sends a `Command`. This supports asynchronous human approval flows spanning seconds or days.
- Parallel node execution (fan-out/fan-in) and supervisor patterns supported.
- Concurrent write conflicts require application-level reducer design per state field.

**Fit for Requirement 2:** Strong. `interrupt()` + persistent checkpointer is the canonical pattern.

### Controlled Tool Execution

- Tools are Python functions; no built-in pre-execution schema enforcement or approval gate at the tool level.
- The canonical pattern for approval is to call `interrupt()` before executing a sensitive tool.
- LangSmith (commercial) provides tracing; open-source tracing requires OpenTelemetry integration.

**Fit for Requirement 3:** Moderate — the interrupt pattern gives approval flows, but a tool registry with audit logging must be built on top.

### Gaps

- No built-in tool registry with per-action schema enforcement or audit log
- Concurrent writes from multiple actors to a single thread require careful reducer design
- No native RBAC over who can read/write which state fields

---

## 2. LlamaIndex

**GitHub:** https://github.com/run-llama/llama_index | ~40,000–46,000 stars

Primarily a data/RAG framework that has grown a full agent layer. Its `AgentWorkflow` and event-driven `Workflows` engine are async-native and well-suited for long-running tasks.

### Memory / State

- Workflow `Context` object is passed between steps; serializable and externalizable with custom backends.
- Pluggable agent memory modules: `VectorMemory` (any vector DB), `SummaryMemory`, `SimpleComposableMemory`.
- Short-term context requires custom serialization for true shared access across processes.

**Fit for Requirement 1:** Good to moderate. Vector-based long-term memory can use any external vector DB; short-term workflow context is less opinionated.

### Concurrent Actors

- AgentWorkflow can run multiple specialist agents in parallel.
- HITL support exists in principle but is less mature than LangGraph's `interrupt()`.

**Fit for Requirement 2:** Moderate. Human actor integration requires a custom layer.

### Controlled Tool Execution

- Tools are Python callables; `ToolMetadata` provides schema declaration (a foundation for a tool registry).
- No built-in pre-execution approval gate or audit log.

**Fit for Requirement 3:** Low without custom work.

### Strengths / Gaps

Strong RAG and knowledge-base capabilities; rich tool ecosystem (100+ integrations via LlamaHub). Workflow context persistence and tool governance need to be added.

---

## 3. AutoGen v0.4 (Microsoft) → now in maintenance mode

**GitHub:** https://github.com/microsoft/autogen | ~50,000–57,000 stars

AutoGen v0.4 (January 2025) redesigned around an **actor model** — each agent processes one message at a time via typed messages through a runtime. Pioneered conversational multi-agent patterns. Now in maintenance mode; new development has moved to the Microsoft Agent Framework.

### Memory / State

- No first-class cross-agent state dict. Shared state is achieved via a dedicated "Memory Bank" agent backed by any external DB — idiomatic but more architectural effort than LangGraph checkpoints.

**Fit for Requirement 1:** Moderate via Memory Bank pattern.

### Concurrent Actors

- Actor model is inherently concurrent. Experimental `DistributedAgentRuntime` allows agents on different machines.
- Human actors modeled as `HumanAgent` (blocks on input). No built-in approval/interrupt primitive at the framework level.

**Fit for Requirement 2:** Good architecture; specific multi-human concurrent write patterns require significant custom work.

### Controlled Tool Execution

- `CodeExecutorAgent` runs code in Docker sandboxes (strong isolation).
- No built-in per-tool policy or audit log.

**Fit for Requirement 3:** Moderate. Message-passing means all tool invocations are interceptable events, but governance requires custom work.

### Gaps

- Maintenance mode: no new features
- Approval gates require a purpose-built human-proxy agent
- Distributed runtime is experimental

---

## 4. Microsoft Agent Framework (AutoGen + Semantic Kernel merged)

**GitHub:** https://github.com/microsoft/agent-framework | GA April 2026
**Semantic Kernel:** https://github.com/microsoft/semantic-kernel | ~27,000–28,000 stars

The production convergence of Semantic Kernel (.NET/Python plugin/kernel architecture) and AutoGen (multi-agent orchestration). Supports MCP and A2A (Agent-to-Agent) protocol for cross-framework interoperability.

### Memory / State

- **Kernel Memory** microservice (`microsoft/kernel-memory`): a standalone deployable service for persistent, searchable memory accessible by multiple agents — an excellent fit for the shared-location requirement.
- Pluggable vector store connectors: Pinecone, Qdrant, Azure Cognitive Search, Postgres pgvector, Chroma, etc.

**Fit for Requirement 1:** Good. Kernel Memory as a microservice is purpose-built for this.

### Concurrent Actors

- Supports concurrent orchestration patterns (parallel agents, handoff, group chat).
- Human-in-the-loop present but documentation still maturing.
- A2A protocol enables cross-framework agent interop.

**Fit for Requirement 2:** Good for multi-agent; human actor integration is less primitive-native.

### Controlled Tool Execution

- **Plugin model**: All tools are "plugins" with explicit schema (name, description, typed parameters) — the most mature built-in tool registry of any surveyed framework.
- **Kernel filters**: Function invocation filters intercept every tool call before and after execution — precisely the hook needed for audit logging and pre-execution validation.
- **Agent Governance Toolkit (AGT)**: Append-only, hash-chained audit logs, per-agent permission policies, adapters for 20+ frameworks.
- **MCP support**: First-class MCP tool integration provides standardized schema enforcement.

**Fit for Requirement 3:** Strongest of all frameworks surveyed. Plugin model + kernel filters + AGT directly addresses the need for schema-validated, interceptable, audit-logged tool calls.

### Gaps

- Most mature features are Azure-centric
- A2A and MCP governance patterns are new (GA April 2026); community patterns still forming
- Kernel Memory microservice adds operational complexity

---

## 5. CrewAI

**GitHub:** https://github.com/crewaiinc/crewai | ~47,000–51,000 stars

Lean Python framework for role-based multi-agent orchestration. Agents are assigned roles, goals, and backstories; tasks are assigned to agents; a "crew" orchestrates execution. Prioritizes developer ergonomics; claimed 2 billion+ agent executions in production.

### Memory / State

Four explicit memory types: short-term (ChromaDB), long-term (SQLite), entity memory, and external memory (Mem0, custom). Defaults are local; shared deployment requires replacing backends with network-accessible stores.

**Fit for Requirement 1:** Moderate. External memory integration exists but defaults to local.

### Concurrent Actors

- `human_input=True` on tasks pauses for human review. `HumanTool` allows agents to ask questions mid-task.
- Crews can execute tasks in parallel. True concurrent multi-process writes to shared state are not natively supported.
- Enterprise tier adds HITL management with SLA and escalation policies.

**Fit for Requirement 2:** Moderate. Single-human HITL works; multi-human concurrent access is not a first-class pattern.

### Controlled Tool Execution

- Tools are Python callables with `@tool` decorator. No built-in pre-execution approval gate.
- Observability, SOC2, and PII masking are in the paid Enterprise tier.

**Fit for Requirement 3:** Low in open-source tier. Governance features are paywalled.

### Strengths / Gaps

Fastest time-to-first-agent; excellent developer ergonomics. Not suitable as the foundation for a distributed harness due to local-first memory defaults and paywalled governance.

---

## 6. OpenAI Agents SDK (successor to Swarm)

**Swarm GitHub:** https://github.com/openai/swarm | **Agents SDK:** https://github.com/openai/openai-agents-python

Swarm (October 2024) was an educational, stateless multi-agent framework. The Agents SDK (March 2025) is the production successor, adding sessions, guardrails, built-in tracing, and human approval gates.

### Memory / State

No built-in shared or distributed memory. Session context is in-process only. Long-term memory requires external integration (Mem0, Pinecone, etc.).

**Fit for Requirement 1:** Weak. All persistence requires external integration.

### Concurrent Actors

- **Approvals API**: Tool calls can require human approval; the run suspends until approved or rejected.
- No multi-human coordination primitive. Concurrent agent instances require external shared storage.

**Fit for Requirement 2:** Moderate for single-human approval flows.

### Controlled Tool Execution

- **Guardrails**: Input/output/tool-call validators run in parallel with agent execution (Pydantic-based schema validation).
- **Tracing**: Built-in traces capture every LLM call, tool call, guardrail event, and handoff.
- Audit traces route to OpenAI's proprietary cloud dashboard, not a self-hosted store.

**Fit for Requirement 3:** Good for single-process scenarios; vendor lock-in on audit storage.

### Gaps

Deep OpenAI API dependency; no self-hosted audit log; no distributed memory.

---

## 7. Haystack (deepset)

**GitHub:** https://github.com/deepset-ai/haystack | ~18,000–21,500 stars

Open-source AI orchestration framework with an **explicit-over-implicit** philosophy. Pipelines are directed graphs of typed components with explicit data flow — agents are first-class pipeline components.

### Memory / State

- Richest document store ecosystem: Elasticsearch, Weaviate, Qdrant, Pinecone, Milvus, Pgvector, OpenSearch — all external and network-accessible.
- Agent `state_schema` accumulates state within a single execution; cross-agent shared state requires routing through a document store.

**Fit for Requirement 1:** Good for document/knowledge memory; transient execution state requires custom work.

### Concurrent Actors

- `HumanFeedbackNode` and configurable approval strategies (`AlwaysAskPolicy`, `AskOncePolicy`). Redis-backed HITL example shows a production pattern where execution pauses and a human responds via a separate process.
- Multi-agent is built by composing pipelines (no first-class multi-agent coordination layer).

**Fit for Requirement 2:** Moderate. Redis HITL pattern is extensible; multi-human concurrent access requires manual orchestration.

### Controlled Tool Execution

- `ComponentTool` wraps any Haystack component as a tool with a full typed interface.
- `MCPTool` connects to MCP servers with inherent schema enforcement.
- Explicit pipeline design makes dataflow auditable; no dedicated per-tool audit log, but easy to add via OpenTelemetry.
- `Hayhooks` exposes pipelines/agents as HTTP APIs, enabling external approval workflows.

**Fit for Requirement 3:** Strong foundation. Explicit typing + MCP/ComponentTool + pipeline explicitness make this the most inherently auditable open-source framework.

### Gaps

No cross-agent shared mutable state primitive; no built-in audit log for tool invocations; HITL requires external orchestration.

---

## 8. Agno (formerly Phidata)

**GitHub:** https://github.com/agno-agi/agno | ~39,000+ stars

High-performance async Python framework for multi-agent systems. Strong self-hosted emphasis — data never leaves your infrastructure. Claims ~10,000 agents/sec in some async benchmarks.

### Memory / State

Three-layer model: memory (per-agent, Postgres/SQLite), storage (session state, Postgres/DynamoDB), knowledge (vector DB RAG). All layers configurable to external, network-accessible databases.

**Fit for Requirement 1:** Good. Self-hosted emphasis is strong; all persistence layers are external.

### Concurrent Actors / Tool Execution

Teams of agents share context with coordinator-managed state routing. Basic HITL support. No tool governance layer. Similar gaps to CrewAI open-source tier.

**Fit for Requirements 2 & 3:** Moderate / Low.

---

## 9. MetaGPT

**GitHub:** https://github.com/FoundationAgents/MetaGPT | ~50,000+ stars

Encodes software engineering workflows as multi-agent systems using Standardized Operating Procedures (SOPs). Agents simulate roles in a software company.

### Why It's Relevant

- **Shared message pool**: All agents publish to and subscribe from a global message pool — the most explicit shared-memory primitive surveyed, and architecturally aligned with the distributed harness model.
- **Structured Action model**: Every agent action is a named, typed `Action` subclass — not an arbitrary bash command. This is the closest existing design to Requirement 3's "specific, controlled, traceable actions."

**Fit for Requirements 1 & 3:** Strong conceptual alignment; limited production tooling for general use cases.

### Gaps

Highly opinionated toward software engineering; default message pool is in-memory; no formal audit log; limited documentation for non-SW-engineering domains.

---

## 10. SmolAgents (HuggingFace)

**GitHub:** https://github.com/huggingface/smolagents | ~15,000+ stars

Minimalist (~1,000 lines of core logic) "code agent" framework where agents write Python to accomplish tasks, executed in a sandbox (E2B, Docker, WASM/Deno). Useful as a sandboxed sub-agent component within a larger harness.

**Fit for all three requirements:** Weak — not designed for distributed/shared memory, multi-agent coordination, or auditable tool execution.

---

## Comparative Summary

| Framework | Shared Memory | Concurrent Actors | Human-in-the-Loop | Controlled Tools | Notes |
|---|---|---|---|---|---|
| **LangGraph** | Excellent | Strong | Excellent (interrupt()) | Moderate | Best overall foundation |
| **Microsoft Agent Framework** | Good | Good | Moderate | Excellent (AGT + filters) | Best tool governance |
| **Haystack** | Good | Moderate | Good | Strong (explicit typing + MCP) | Most auditable pipeline |
| **AutoGen v0.4** | Moderate | Good | Moderate | Moderate | Maintenance mode |
| **LlamaIndex** | Good | Moderate | Basic | Low | Best for RAG-heavy use cases |
| **Agno** | Good | Moderate | Basic | Low | Best for high-throughput |
| **MetaGPT** | Moderate | Good | None | Strong (Action model) | Conceptually aligned; SW-eng focus |
| **CrewAI** | Weak (local default) | Moderate | Moderate | Low (OSS) | Best prototyping speed |
| **OpenAI Agents SDK** | Weak | Moderate | Good | Good (cloud-only audit) | Vendor lock-in |
| **SmolAgents** | None | None | None | Moderate (sandboxed) | Sub-agent only |

---

## Adaptation Recommendations

### Requirement 1 — Shared Memory

Adopt **LangGraph's two-layer model**:
1. **Execution state** (current step, decisions, task progress): PostgreSQL-backed checkpointer with row-level locking per thread
2. **Long-term knowledge memory** (facts, artifacts, research outputs): external vector DB (pgvector or Redis with vector search) via the `BaseStore` interface

### Requirement 2 — Concurrent Actors

No framework fully solves simultaneous multi-human writes out of the box. The recommended pattern:

- Use **LangGraph's `interrupt()` as the human gate**: execution pauses, state persists, humans (or agents) resume via a thin HTTP API
- For true concurrent writes (multiple humans editing the same memory namespace): adopt **event sourcing** — actors append immutable events rather than overwriting state; current state is derived from the event log. This eliminates write conflicts by design.

### Requirement 3 — Controlled, Auditable Tool Execution

Combine two patterns:

1. **Tool registry** based on **Semantic Kernel's plugin schema** (name, description, parameter JSON Schema, return schema) — all tools declared upfront, no arbitrary shell access
2. **Pre-execution filter** inspired by SK's kernel filters: validates parameters against schema → checks caller permissions → records to append-only audit log (PostgreSQL or a ledger) → optionally routes to human approval queue → executes

Adapt **MetaGPT's `Action` model** as the conceptual template: every tool invocation is a named, typed object, not a string command.

### Anti-Patterns to Avoid

- Do not build on **CrewAI** (local-first memory, paywalled governance), **OpenAI Agents SDK** (vendor lock-in, no distributed memory), or **SmolAgents** (no multi-agent support) as a foundation — all require rebuilding the most important parts from scratch.
- Avoid any framework that permits **arbitrary bash/shell execution** as a tool primitive — it is incompatible with Requirement 3.

---

## Further Reading

- [LangGraph Persistence Docs](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Human-in-the-Loop](https://www.langchain.com/blog/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt)
- [Microsoft Agent Governance Toolkit](https://devblogs.microsoft.com/agent-framework/governance-at-the-speed-of-agents-microsoft-agent-framework-and-agent-governance-toolkit-better-together/)
- [Haystack Multi-Agent Tutorial](https://haystack.deepset.ai/tutorials/45_creating_a_multi_agent_system)
- [JustAct+ — Auditable Multi-Agent Systems](https://arxiv.org/abs/2502.00138)
- [AI Agent Memory: Comparative Analysis](https://dev.to/foxgem/ai-agent-memory-a-comparative-analysis-of-langgraph-crewai-and-autogen-31dp)
