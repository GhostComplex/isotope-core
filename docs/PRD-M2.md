# isotope-core — Milestone 2: OpenAI & Anthropic Providers

## Objective

Implement real LLM providers for OpenAI and Anthropic that plug into the existing Provider protocol from M1.

## Prerequisites

- M1 merged — all core types, Provider protocol, and agent loop are in place
- Provider protocol defined in `src/isotope_core/providers/base.py`

## Deliverables

### 1. OpenAI Provider (`src/isotope_core/providers/openai.py`)

Implement the Provider protocol for OpenAI's Chat Completions API with streaming.

**Requirements:**
- Use the `openai` Python SDK (optional dependency)
- Support models: gpt-4o, gpt-4o-mini, o3, o4-mini (and any chat completions compatible model)
- Streaming via `client.chat.completions.create(stream=True)`
- Map OpenAI's streaming chunks to our `StreamEvent` types:
  - `choices[0].delta.content` → `StreamTextDeltaEvent`
  - `choices[0].delta.tool_calls` → `StreamToolCallStartEvent`, `StreamToolCallDeltaEvent`, `StreamToolCallEndEvent`
  - Reasoning tokens (o-series) → `StreamThinkingDeltaEvent` (if available via API)
- Handle `finish_reason`: `stop` → `StopReason.END_TURN`, `tool_calls` → `StopReason.TOOL_USE`, `length` → `StopReason.MAX_TOKENS`
- Map our `Context` → OpenAI messages format:
  - `UserMessage` → `{"role": "user", "content": [...]}`
  - `AssistantMessage` → `{"role": "assistant", "content": ..., "tool_calls": [...]}`
  - `ToolResultMessage` → `{"role": "tool", "tool_call_id": ..., "content": ...}`
- Map our `ToolSchema` → OpenAI function calling format
- Support `temperature`, `max_tokens` parameters
- Handle abort via `signal` (cancel the HTTP stream)
- Extract `Usage` from final streaming chunk
- Error handling: API errors → `StreamErrorEvent` (never throw)

**Class signature:**
```python
class OpenAIProvider:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,  # Falls back to OPENAI_API_KEY env var
        base_url: str | None = None,  # For OpenAI-compatible proxies
        organization: str | None = None,
        default_headers: dict[str, str] | None = None,
    ): ...
    
    async def stream(self, context, *, temperature, max_tokens, signal) -> AsyncGenerator[StreamEvent]:
        ...
```

### 2. Anthropic Provider (`src/isotope_core/providers/anthropic.py`)

Implement the Provider protocol for Anthropic's Messages API with streaming.

**Requirements:**
- Use the `anthropic` Python SDK (optional dependency)
- Support models: claude-sonnet-4-20250514, claude-opus-4-20250514 (and any messages API compatible model)
- Streaming via `client.messages.stream()`
- Map Anthropic's streaming events to our `StreamEvent` types:
  - `content_block_start` (text) + `content_block_delta` (text_delta) → `StreamTextDeltaEvent`
  - `content_block_start` (tool_use) + `content_block_delta` (input_json_delta) → `StreamToolCall*Event`
  - `content_block_start` (thinking) + `content_block_delta` (thinking_delta) → `StreamThinkingDeltaEvent`
- Handle `stop_reason`: `end_turn` → `StopReason.END_TURN`, `tool_use` → `StopReason.TOOL_USE`, `max_tokens` → `StopReason.MAX_TOKENS`
- Map our `Context` → Anthropic messages format:
  - System prompt → `system` parameter (not in messages)
  - `UserMessage` → `{"role": "user", "content": [...]}`
  - `AssistantMessage` → `{"role": "assistant", "content": [...]}` with tool_use blocks
  - `ToolResultMessage` → `{"role": "user", "content": [{"type": "tool_result", ...}]}`
- Map our `ToolSchema` → Anthropic tool format
- Support extended thinking (for Claude models that support it)
- Cache control: support `cache_control` headers for prompt caching
- Handle abort via `signal`
- Extract `Usage` from message response
- Error handling: API errors → `StreamErrorEvent` (never throw)

**Class signature:**
```python
class AnthropicProvider:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,  # Falls back to ANTHROPIC_API_KEY env var
        base_url: str | None = None,
        max_tokens: int = 8192,
        thinking: ThinkingConfig | None = None,  # Extended thinking config
    ): ...
    
    async def stream(self, context, *, temperature, max_tokens, signal) -> AsyncGenerator[StreamEvent]:
        ...
```

### 3. Provider Utilities (`src/isotope_core/providers/utils.py`)

Shared utilities for providers:

- `retry_with_backoff()` — Async retry decorator with exponential backoff
  - Configurable max retries, initial delay, max delay
  - Handle rate limit (429) with `Retry-After` header
  - Handle transient errors (500, 502, 503, 504)
- `map_error_to_stop_reason()` — Map HTTP errors to appropriate StopReason
- `create_error_message()` — Create an AssistantMessage with error info

### 4. Provider Registration (`src/isotope_core/providers/__init__.py`)

Update the providers package to export the new providers:

```python
from isotope_core.providers.openai import OpenAIProvider
from isotope_core.providers.anthropic import AnthropicProvider
```

Handle import errors gracefully when optional dependencies aren't installed.

### 5. Tests

**Unit tests (with mocked HTTP):**
- `tests/test_provider_openai.py`:
  - Test context → OpenAI message format conversion
  - Test tool schema conversion
  - Test streaming chunk → StreamEvent mapping
  - Test error handling (API errors, rate limits, timeouts)
  - Test abort signal
  - Test usage extraction
- `tests/test_provider_anthropic.py`:
  - Test context → Anthropic message format conversion
  - Test tool schema conversion
  - Test streaming event → StreamEvent mapping
  - Test thinking content handling
  - Test error handling
  - Test abort signal
  - Test usage extraction
- `tests/test_provider_utils.py`:
  - Test retry logic
  - Test error mapping

**Integration tests (optional, require API keys):**
- `tests/integration/test_openai_integration.py` — marked with `@pytest.mark.integration`
- `tests/integration/test_anthropic_integration.py` — marked with `@pytest.mark.integration`
- These should be skippable in CI without API keys

### 6. Update `__init__.py`

Export `OpenAIProvider` and `AnthropicProvider` from the top-level package.

## Technical Constraints

- Provider SDKs are optional dependencies (install via `uv add "isotope-core[openai]"` etc.)
- Use lazy imports to avoid ImportError when SDKs aren't installed
- All streaming must be proper async generators (not buffered)
- Never throw from `stream()` — all errors go through `StreamErrorEvent`
- Providers must not import each other
- Type annotations must pass mypy strict

## Branch

`feat/isotope-core/dev-m2`

## Definition of Done

- [ ] OpenAI provider passes unit tests with mocked HTTP
- [ ] Anthropic provider passes unit tests with mocked HTTP  
- [ ] Both providers correctly map to/from our type system
- [ ] Tool use round-trips work (context → API → response → events)
- [ ] Thinking/reasoning content handled for both providers
- [ ] Abort signal properly cancels streaming
- [ ] Error handling never throws, always yields StreamErrorEvent
- [ ] Retry utility with exponential backoff
- [ ] All tests pass, ruff/mypy clean
- [ ] Graceful import errors when SDKs not installed
