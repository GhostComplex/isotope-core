# isotope-core — Milestone 3: Context Management

## Objective

Implement context window management — token counting, overflow detection, and pluggable pruning strategies to handle long conversations gracefully.

## Prerequisites

- M1 merged — core types, agent loop
- M2 merged — providers (OpenAI, Anthropic)

## Deliverables

### 1. Token Counting (`src/isotope_core/context.py`)

Token counting utilities:

- `count_tokens(messages, model?)` — count tokens in a message list
  - For OpenAI models: use `tiktoken` (optional dep) for accurate counts
  - Fallback: character-based estimation (~4 chars per token)
  - Count system prompt tokens separately
- `count_message_tokens(message, model?)` — count tokens in a single message
- `estimate_context_usage(context, model?)` — return `ContextUsage` with:
  - `total_tokens`, `system_tokens`, `message_tokens`, `tool_tokens`
  - `remaining_tokens` (based on model's context window)
  - `utilization` (0.0-1.0)

```python
@dataclass
class ContextUsage:
    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    context_window: int
    remaining_tokens: int
    utilization: float  # 0.0 to 1.0
```

### 2. Model Context Windows

Built-in context window sizes for known models:

```python
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o3": 200_000,
    "o4-mini": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    # ... etc
}
```

Allow custom overrides via `get_context_window(model, custom_windows?)`.

### 3. Pruning Strategies

Implement pluggable pruning strategies via a `PruningStrategy` protocol:

```python
class PruningStrategy(Protocol):
    async def prune(
        self,
        messages: list[Message],
        target_tokens: int,
        *,
        model: str | None = None,
    ) -> PruneResult: ...

@dataclass
class PruneResult:
    messages: list[Message]
    pruned_count: int
    pruned_tokens: int
```

**Built-in strategies:**

#### a) `SlidingWindowStrategy`
- Drop oldest turns (user + assistant pairs) until under budget
- Always keep system prompt
- Keep the most recent N turns
- Option: `keep_first_n` — preserve first N messages (for context-setting messages)

#### b) `SummarizationStrategy`
- Compress old messages into a summary via an LLM call
- Requires a `Provider` instance for the summarization call
- Configurable summary prompt
- Keep recent N turns verbatim, summarize the rest
- The summary becomes a new system message or pinned user message

#### c) `SelectivePruningStrategy`
- Keep system + recent N turns + any pinned messages
- Drop everything else
- Respects message pinning (see below)

### 4. Message Pinning

Add pinning support to prevent important messages from being pruned:

- Add `pinned: bool = False` field to `UserMessage`, `AssistantMessage`, `ToolResultMessage`
- All pruning strategies must respect pinned messages
- `pin_message(messages, index)` / `unpin_message(messages, index)` helpers

### 5. Context Budget Events

New event types for context management:

```python
@dataclass
class ContextPrunedEvent:
    type: Literal["context_pruned"] = "context_pruned"
    strategy: str
    pruned_count: int
    pruned_tokens: int
    remaining_tokens: int
```

Emit `ContextPrunedEvent` when context is pruned (via the existing `transform_context` hook).

### 6. Transform Context Hook Implementations

Provide ready-to-use `transform_context` hook factories:

```python
def create_sliding_window_transform(
    max_tokens: int | None = None,
    keep_recent: int = 10,
    model: str | None = None,
) -> TransformContextHook: ...

def create_summarization_transform(
    provider: Provider,
    max_tokens: int | None = None,
    keep_recent: int = 5,
    model: str | None = None,
    summary_prompt: str | None = None,
) -> TransformContextHook: ...
```

### 7. Tests

- `tests/test_context.py`:
  - Token counting (with tiktoken if available, fallback otherwise)
  - Context usage estimation
  - Sliding window pruning (various edge cases)
  - Summarization pruning (with mock provider)
  - Selective pruning with pinned messages
  - Transform context hook factories
  - ContextPrunedEvent emission
  - Model context window lookup
  - Edge cases: empty messages, single message, all pinned

## Technical Constraints

- `tiktoken` is an optional dependency (install via `uv add "isotope-core[tiktoken]"`)
- Fallback token counting must work without tiktoken
- Pruning strategies must never drop pinned messages
- Summarization strategy must use the existing Provider protocol
- All operations should be async-compatible
- Type-safe — mypy strict

## Branch

`feat/isotope-core/dev-m3`

## Definition of Done

- [ ] Token counting works (tiktoken + fallback)
- [ ] Context usage estimation with model-aware context windows
- [ ] Sliding window pruning passes all tests
- [ ] Summarization pruning works with mock provider
- [ ] Selective pruning respects pinned messages
- [ ] Transform hook factories work with agent loop
- [ ] ContextPrunedEvent emitted on prune
- [ ] All tests pass, ruff/mypy clean
