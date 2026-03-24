# isotope-core — Product Roadmap

## Vision

A minimal, composable Python library for building AI agent loops. Provides the core primitives — types, streaming, tool execution, context management — so developers can build agents without buying into a framework.

## Reference Architecture

Patterns drawn from:
- **pi-mono** (`@mariozechner/pi-agent-core`) — agent loop, event system, tool hooks
- **gemini-cli** — agent protocol, sub-agents, elicitation, safety
- **openclaw** — skills, session management, multi-channel routing

---

## Milestone 1: Core Types & Agent Loop

> Foundation. Get the loop running.

**Branch:** `feat/isotope-core/dev-m1`
**PRD:** `docs/PRD-M1.md`

**Deliverables:**
- Core types: messages, content blocks, events, context, usage
- Provider protocol (streaming interface)
- Tool framework with JSON Schema validation
- Async event stream
- Agent loop (turn-based, parallel/sequential tool execution)
- Stateful Agent class
- Test suite with mock provider

**Definition of Done:** Agent can prompt → stream response → execute tools → loop → return events. All tests pass, ruff/mypy clean.

---

## Milestone 2: OpenAI & Anthropic Providers

> Make it actually talk to LLMs.

**Branch:** `feat/isotope-core/dev-m2`

**Deliverables:**
- `providers/openai.py` — OpenAI Chat Completions API (streaming)
  - gpt-4o, gpt-4o-mini, o3, o4-mini support
  - Function calling / tool_use mapping
  - Streaming text deltas + tool call deltas
  - Reasoning/thinking content support (o-series)
- `providers/anthropic.py` — Anthropic Messages API (streaming)
  - claude-sonnet, claude-opus support
  - Tool use with content blocks
  - Extended thinking support
  - Cache control headers (prompt caching)
- Shared provider utilities (retry logic, rate limit handling, error mapping)
- Integration tests (can be skipped in CI without API keys)
- Provider-specific options pass-through

**Definition of Done:** `Agent(provider=OpenAIProvider(...))` and `Agent(provider=AnthropicProvider(...))` work end-to-end with real API calls. Tool use round-trips correctly.

---

## Milestone 3: Context Management

> Handle real conversations that exceed context windows.

**Branch:** `feat/isotope-core/dev-m3`

**Deliverables:**
- `context.py` — Context management module
  - Token counting (tiktoken for OpenAI, character estimation fallback)
  - Context window overflow detection
  - Message pruning strategies:
    - Sliding window (drop oldest turns)
    - Summarization (compress old context via LLM call)
    - Selective pruning (keep system + recent N turns + pinned messages)
  - `transform_context` hook implementations
- Message pinning (mark messages as unprunable)
- Context budget tracking (warn when approaching limit)
- Tests with various overflow scenarios

**Definition of Done:** Long conversations automatically manage context window. Pruning strategies are pluggable. No silent data loss — events emitted when context is pruned.

---

## Milestone 4: Steering, Follow-up & Session Control

> Interactive control: interrupt, steer, queue, resume.

**Branch:** `feat/isotope-core/dev-m4`

**Deliverables:**
- Steering messages — inject user messages mid-execution (after current tool calls finish)
- Follow-up queue — queue messages for after agent would stop
- `agent.steer(message)` / `agent.follow_up(message)` APIs
- `agent.continue_()` — resume from current context (retry on error)
- Session ID support for provider-side caching
- Abort improvements — clean cancellation with proper event emission
- Max turns / max tokens budget limits with `agent_end` reason
- Tests for steering/follow-up interleaving

**Definition of Done:** Can steer agent mid-run, queue follow-ups, resume after errors. Budget limits enforced. All control flows emit proper events.

---

## Milestone 5: Multi-Provider & Model Routing

> Swap models mid-conversation, fall back on failures.

**Branch:** `feat/isotope-core/dev-m5`

**Deliverables:**
- `providers/router.py` — Model routing layer
  - Dynamic model switching (`agent.set_provider(...)` mid-session)
  - Fallback chains (try model A → fall back to model B on error/rate-limit)
  - Provider health tracking (circuit breaker pattern)
- `providers/proxy.py` — Generic OpenAI-compatible proxy provider
  - Works with any OpenAI-compatible endpoint (LiteLLM, vLLM, Ollama, etc.)
  - Custom base URL + headers
- Dynamic API key resolution (for expiring OAuth tokens)
- Provider-level retry with exponential backoff
- Usage aggregation across provider switches

**Definition of Done:** Can configure fallback chains. Proxy provider works with OpenAI-compatible endpoints. API key refresh works for long-running sessions.

---

## Milestone 6: Hooks & Middleware

> Extensibility without subclassing.

**Branch:** `feat/isotope-core/dev-m6`

**Deliverables:**
- Hook system formalization:
  - `before_tool_call` / `after_tool_call` (already in M1, enhance)
  - `before_llm_call` — inspect/modify context before sending to LLM
  - `after_llm_call` — inspect/modify response before processing
  - `on_event` — tap into any event for logging/telemetry
- Middleware pattern — composable wrappers:
  - Logging middleware (structured event logging)
  - Token tracking middleware (accumulate usage across turns)
  - Safety middleware (block dangerous tool calls by pattern)
- Hook ordering and priority
- Tests for hook composition and error handling in hooks

**Definition of Done:** Users can add hooks/middleware without touching core code. Hooks can block, modify, or observe at every lifecycle point.

---

## Milestone 7: Testing & Quality

> Harden everything for production use.

**Branch:** `feat/isotope-core/dev-m7`

**Deliverables:**
- Mock provider enhancements:
  - Scripted responses (sequence of responses for deterministic testing)
  - Simulated delays and errors
  - Tool call simulation
- `isotope_core.testing` module — public test utilities
  - `MockProvider` with response scripting
  - `assert_events()` helper for event sequence validation
  - `record_events()` context manager
- Property-based tests (hypothesis) for serialization round-trips
- Edge case tests:
  - Empty tool results
  - Tool throwing exceptions
  - Provider returning malformed responses
  - Abort during tool execution
  - Concurrent prompts on same agent
- CI configuration (GitHub Actions)
- 90%+ code coverage target

**Definition of Done:** CI green. Coverage ≥ 90%. Testing utilities are documented and exported.

---

## Technical Principles (All Milestones)

- **Python 3.11+**, Pydantic v2
- **Pure async/await** — no threads, no sync wrappers
- **Zero required dependencies** beyond pydantic (providers are optional extras)
- **Type-safe** — mypy strict mode, full annotations
- **Event-driven** — every state change emits an event
- **Composable** — prefer hooks and protocols over inheritance
- **Tested** — every module has tests, mock provider for unit tests

## Out of Scope (For Now)

- CLI / TUI (separate package)
- MCP server/client support
- Sub-agent orchestration (multi-agent)
- Persistent storage / conversation database
- Web UI
- Deployment / containerization
