# API Reference

Complete API reference for `isotope-core`.

## Table of Contents

- [Agent](#agent)
- [Agent Loop](#agent-loop)
- [Types](#types)
  - [Content Types](#content-types)
  - [Message Types](#message-types)
  - [Context](#context)
  - [Events](#events)
  - [Usage](#usage)
- [Tools](#tools)
- [Providers](#providers)
  - [Provider Protocol](#provider-protocol)
  - [Stream Events](#stream-events)
  - [OpenAIProvider](#openaiprovider)
  - [AnthropicProvider](#anthropicprovider)
  - [ProxyProvider](#proxyprovider)
  - [RouterProvider](#routerprovider)
- [Context Management](#context-management)
- [Middleware](#middleware)
- [Events Module](#events-module)
- [Provider Utilities](#provider-utilities)

---

## Agent

### `class Agent`

Stateful agent wrapping the agent loop. Manages messages, tools, subscriptions, steering, and follow-up queues.

```python
from isotope_core import Agent
```

#### Constructor

```python
Agent(
    provider: Provider | None = None,
    system_prompt: str = "",
    tools: list[Tool] | None = None,
    tool_execution: str = "parallel",
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_turns: int | None = None,
    max_total_tokens: int | None = None,
    before_tool_call: BeforeToolCallHook | None = None,
    after_tool_call: AfterToolCallHook | None = None,
    transform_context: TransformContextHook | None = None,
    on_agent_start: OnAgentStartHook | None = None,
    on_agent_end: OnAgentEndHook | None = None,
    on_turn_start: OnTurnStartHook | None = None,
    on_turn_end: OnTurnEndHook | None = None,
    on_error: OnErrorHook | None = None,
    middleware: list[Middleware] | None = None,
)
```

**Parameters:**
- `provider` — LLM provider implementing the `Provider` protocol.
- `system_prompt` — System prompt for the LLM.
- `tools` — List of `Tool` instances available for tool use.
- `tool_execution` — `"parallel"` (default) or `"sequential"`.
- `temperature` — Sampling temperature for the LLM.
- `max_tokens` — Maximum tokens per LLM turn.
- `max_turns` — Maximum number of turns before the loop stops.
- `max_total_tokens` — Maximum cumulative tokens before budget stop.
- `before_tool_call` — Async hook called before each tool execution.
- `after_tool_call` — Async hook called after each tool execution.
- `transform_context` — Async hook to transform context before LLM calls.
- `on_agent_start/end`, `on_turn_start/end`, `on_error` — Lifecycle hooks.
- `middleware` — List of middleware instances for event processing.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `state` | `AgentState` | Current agent state |
| `messages` | `list[Message]` | Message history |
| `is_streaming` | `bool` | Whether agent is currently processing |

#### Methods

| Method | Description |
|--------|-------------|
| `async prompt(text, images, messages)` | Send a prompt; yields `AgentEvent` |
| `async continue_()` | Continue from current context |
| `set_system_prompt(prompt)` | Set the system prompt |
| `set_provider(provider)` | Set the LLM provider |
| `set_tools(tools)` | Set the available tools |
| `replace_messages(messages)` | Replace message history |
| `append_message(message)` | Append a message |
| `clear_messages()` | Clear all messages |
| `reset()` | Reset agent state |
| `subscribe(callback) -> unsubscribe` | Subscribe to events |
| `steer(message)` | Inject a steering message mid-loop |
| `follow_up(message)` | Queue a follow-up message |
| `abort()` | Abort current operation |

### `class AgentState`

```python
@dataclass
class AgentState:
    system_prompt: str = ""
    provider: Provider | None = None
    tools: list[Tool] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: Message | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None
```

---

## Agent Loop

### `async agent_loop(prompts, context, config, signal)`

Core execution engine. Runs the turn-based agent loop.

```python
from isotope_core import agent_loop, AgentLoopConfig
```

**Parameters:**
- `prompts: list[Message]` — Initial messages to add to context.
- `context: Context` — Conversation context.
- `config: AgentLoopConfig` — Loop configuration.
- `signal: asyncio.Event | None` — Optional abort signal.

**Yields:** `AgentEvent`

### `class AgentLoopConfig`

```python
@dataclass
class AgentLoopConfig:
    provider: Provider
    tools: list[Tool] = field(default_factory=list)
    tool_execution: Literal["parallel", "sequential"] = "parallel"
    temperature: float | None = None
    max_tokens: int | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    transform_context: TransformContextHook | None = None
    steering_queue: asyncio.Queue[Message] | None = None
    follow_up_queue: asyncio.Queue[Message] | None = None
    max_turns: int | None = None
    max_total_tokens: int | None = None
    middleware: list[Middleware] | None = None
    lifecycle_hooks: LifecycleHooks | None = None
```

### Hook Types

```python
# Before a tool is executed — can block execution
@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: dict[str, Any]
    context: Context

@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None

# After a tool executes — can modify result
@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: dict[str, Any]
    result: ToolResult
    is_error: bool
    context: Context

@dataclass
class AfterToolCallResult:
    content: list[TextContent | ImageContent] | None = None
    is_error: bool | None = None

# Context transform
TransformContextHook = Callable[
    [list[Message], asyncio.Event | None],
    Awaitable[list[Message]],
]
```

---

## Types

### Content Types

```python
from isotope_core.types import TextContent, ImageContent, ThinkingContent, ToolCallContent

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str          # base64 encoded
    mime_type: str     # e.g., "image/jpeg"

class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False

class ToolCallContent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, object]

Content = TextContent | ImageContent | ThinkingContent | ToolCallContent
```

### Message Types

```python
from isotope_core.types import UserMessage, AssistantMessage, ToolResultMessage

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent]
    timestamp: int      # Unix ms
    pinned: bool = False

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCallContent]
    stop_reason: StopReason | None = None
    usage: Usage = Usage()
    error_message: str | None = None
    timestamp: int
    pinned: bool = False

class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    is_error: bool = False
    timestamp: int
    pinned: bool = False

Message = UserMessage | AssistantMessage | ToolResultMessage
```

### Context

```python
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, object]

class Context(BaseModel):
    system_prompt: str = ""
    messages: list[Message] = []
    tools: list[ToolSchema] = []
```

### Events

```python
from isotope_core.types import (
    AgentStartEvent, AgentEndEvent,
    TurnStartEvent, TurnEndEvent,
    MessageStartEvent, MessageUpdateEvent, MessageEndEvent,
    ToolStartEvent, ToolUpdateEvent, ToolEndEvent,
    ContextPrunedEvent, SteerEvent, FollowUpEvent,
)

# Discriminated union
AgentEvent = (
    AgentStartEvent | AgentEndEvent
    | TurnStartEvent | TurnEndEvent
    | MessageStartEvent | MessageUpdateEvent | MessageEndEvent
    | ToolStartEvent | ToolUpdateEvent | ToolEndEvent
    | ContextPrunedEvent | SteerEvent | FollowUpEvent
)
```

### Usage

```python
class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"
    ABORTED = "aborted"

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int: ...

class AggregatedUsage(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    provider_usage: dict[str, Usage] = {}
    model_usage: dict[str, Usage] = {}
```

---

## Tools

### `class Tool`

```python
from isotope_core.tools import Tool, ToolResult

tool = Tool(
    name="read_file",
    description="Read a file",
    parameters={"type": "object", "properties": {...}, "required": [...]},
    execute=my_execute_fn,
)
```

**Methods:**
- `validate_arguments(args) -> (bool, str | None)` — Validate args against schema.
- `async execute(tool_call_id, arguments, signal, on_update) -> ToolResult` — Execute the tool.
- `to_schema() -> dict` — Convert to JSON-serializable schema.

### `class ToolResult`

```python
@dataclass
class ToolResult:
    content: list[TextContent | ImageContent] = []
    is_error: bool = False

    @classmethod
    def text(cls, text: str, is_error: bool = False) -> ToolResult: ...

    @classmethod
    def error(cls, message: str) -> ToolResult: ...
```

### `@tool` Decorator

```python
from isotope_core.tools import tool

@tool(name="get_weather", description="Get weather", parameters={...})
async def get_weather(tool_call_id, params, signal, on_update):
    return ToolResult.text("Sunny")
```

### Exceptions

- `ToolError` — Base tool exception.
- `ToolValidationError` — Arguments failed validation.
- `ToolNotFoundError` — Tool not found.
- `ToolExecutionError` — Tool execution failed.

---

## Providers

### Provider Protocol

```python
from isotope_core.providers.base import Provider

class Provider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def stream(
        self, context: Context, *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]: ...
```

### Stream Events

```python
from isotope_core.providers.base import (
    StreamStartEvent,      # type="start"
    StreamTextDeltaEvent,  # type="text_delta"
    StreamTextEndEvent,    # type="text_end"
    StreamThinkingDeltaEvent,  # type="thinking_delta"
    StreamThinkingEndEvent,    # type="thinking_end"
    StreamToolCallStartEvent,  # type="tool_call_start"
    StreamToolCallDeltaEvent,  # type="tool_call_delta"
    StreamToolCallEndEvent,    # type="tool_call_end"
    StreamDoneEvent,       # type="done"
    StreamErrorEvent,      # type="error"
)
```

### OpenAIProvider

```python
from isotope_core.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o",
    api_key="sk-...",          # or OPENAI_API_KEY env var
    base_url=None,             # optional custom base URL
    default_headers=None,      # optional headers
    api_key_resolver=None,     # optional async key resolver
)
```

### AnthropicProvider

```python
from isotope_core.providers.anthropic import AnthropicProvider, ThinkingConfig

provider = AnthropicProvider(
    model="claude-sonnet-4-20250514",
    api_key="sk-...",          # or ANTHROPIC_API_KEY env var
    base_url=None,
    max_tokens=8192,
    thinking=ThinkingConfig(enabled=True, budget_tokens=4096),
    api_key_resolver=None,
)
```

#### `ThinkingConfig`

```python
@dataclass
class ThinkingConfig:
    enabled: bool = False
    budget_tokens: int = 1024
```

### ProxyProvider

```python
from isotope_core.providers.proxy import ProxyProvider

provider = ProxyProvider(
    model="llama-3.1-70b",
    base_url="http://localhost:11434/v1",
    api_key=None,
    default_headers=None,
    timeout=120.0,
    api_key_resolver=None,
)
```

### RouterProvider

```python
from isotope_core import RouterProvider, CircuitState

router = RouterProvider(
    primary=openai_provider,
    fallbacks=[anthropic_provider],
    health_check_interval=60.0,
    circuit_breaker_threshold=3,
    circuit_breaker_timeout=120.0,
)
```

**Methods:**
- `set_primary(provider)` — Switch primary provider.
- `get_usage() -> AggregatedUsage` — Get aggregated usage.

**Circuit States:**
- `CircuitState.CLOSED` — Healthy, requests pass through.
- `CircuitState.OPEN` — Tripped, requests skipped.
- `CircuitState.HALF_OPEN` — Testing, one request allowed.

---

## Context Management

### Token Counting

```python
from isotope_core import count_tokens, count_message_tokens, get_context_window

tokens = count_tokens(messages, model="gpt-4o")
msg_tokens = count_message_tokens(message, model="gpt-4o")
window = get_context_window("gpt-4o")  # 128_000
```

### Context Usage

```python
from isotope_core import estimate_context_usage, ContextUsage

usage = estimate_context_usage(context, model="gpt-4o")
# ContextUsage(total_tokens, system_tokens, message_tokens, tool_tokens,
#              context_window, remaining_tokens, utilization)
```

### Message Pinning

```python
from isotope_core import pin_message, unpin_message

messages = pin_message(messages, index=0)    # Returns new list
messages = unpin_message(messages, index=0)
```

### Pruning Strategies

All strategies implement the `PruningStrategy` protocol:

```python
class PruningStrategy(Protocol):
    async def prune(
        self, messages: list[Message], target_tokens: int, *, model: str | None = None
    ) -> PruneResult: ...

@dataclass
class PruneResult:
    messages: list[Message]
    pruned_count: int
    pruned_tokens: int
```

#### `SlidingWindowStrategy`

Drops oldest non-protected messages until within budget. Respects pinned messages.

```python
from isotope_core import SlidingWindowStrategy

strategy = SlidingWindowStrategy(keep_recent=10, keep_first_n=1)
result = await strategy.prune(messages, target_tokens=50_000)
```

#### `SummarizationStrategy`

Summarizes older messages via an LLM call. Keeps recent N messages verbatim.

```python
from isotope_core import SummarizationStrategy

strategy = SummarizationStrategy(
    provider=my_provider,
    keep_recent=5,
    summary_prompt="Summarize this conversation...",
)
result = await strategy.prune(messages, target_tokens=50_000)
```

#### `SelectivePruningStrategy`

Keeps only recent + pinned messages.

```python
from isotope_core import SelectivePruningStrategy

strategy = SelectivePruningStrategy(keep_recent=10)
result = await strategy.prune(messages, target_tokens=50_000)
```

### Transform Context Hook Factories

```python
from isotope_core import create_sliding_window_transform, create_summarization_transform

transform = create_sliding_window_transform(
    max_tokens=50_000, keep_recent=10, model="gpt-4o", keep_first_n=1
)

transform = create_summarization_transform(
    provider=my_provider, max_tokens=50_000, keep_recent=5
)

# Use with Agent
agent = Agent(provider=..., transform_context=transform)
```

### Model Context Windows

```python
from isotope_core import MODEL_CONTEXT_WINDOWS

# Built-in mappings:
# "gpt-4o": 128_000, "gpt-4": 8_192, "o3": 200_000,
# "claude-sonnet-4-20250514": 200_000, etc.
```

---

## Middleware

### Protocol

```python
from isotope_core import Middleware, MiddlewareContext

class Middleware(Protocol):
    async def on_event(
        self, event: AgentEvent, context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None: ...

@dataclass
class MiddlewareContext:
    messages: list[Message]
    turn_number: int
    cumulative_tokens: int
    agent_config: Any
```

### `run_middleware_chain(event, context, middleware)`

Run an event through the middleware chain.

### Built-in Middleware

#### `LoggingMiddleware`

```python
LoggingMiddleware(
    logger=print,                # Custom logger callable
    log_level="normal",          # "minimal" | "normal" | "verbose"
    include_content=False,       # Include event content in log
)
```

#### `TokenTrackingMiddleware`

```python
tracker = TokenTrackingMiddleware()
# After use:
tracker.total_usage     # Usage
tracker.per_turn_usage  # list[Usage]
tracker.turn_count      # int
```

#### `EventFilterMiddleware`

```python
EventFilterMiddleware(exclude={"message_update", "tool_update"})
```

### Lifecycle Hooks

```python
from isotope_core import LifecycleHooks

hooks = LifecycleHooks(
    on_agent_start=async_fn(),        # Callable[[], Awaitable[None]]
    on_agent_end=async_fn(reason),    # Callable[[str], Awaitable[None]]
    on_turn_start=async_fn(turn),     # Callable[[int], Awaitable[None]]
    on_turn_end=async_fn(turn, msg),  # Callable[[int, AssistantMessage], Awaitable[None]]
    on_error=async_fn(exc),           # Callable[[Exception], Awaitable[None]]
)
```

---

## Events Module

### `class EventStream[T, R]`

Generic async event stream supporting pull and push consumption.

```python
from isotope_core import EventStream

stream = EventStream(
    is_complete=lambda e: e.type == "done",
    extract_result=lambda e: e.value,
)

stream.push(event)           # Push event to consumers
stream.end(result)           # End the stream
stream.subscribe(callback)   # Push-based subscription
result = await stream.result()  # Get final result

async for event in stream:   # Pull-based iteration
    ...
```

### `class AgentEventStream`

Specialized `EventStream[AgentEvent, list[Message]]` for agent events.

---

## Provider Utilities

### `RetryConfig`

```python
from isotope_core import RetryConfig

config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retryable_status_codes=(429, 500, 502, 503, 504),
)
```

### `retry_with_backoff(config)`

Decorator for async functions with exponential backoff retry.

```python
from isotope_core import retry_with_backoff

@retry_with_backoff(RetryConfig(max_retries=3))
async def call_api():
    ...
```

### Utility Functions

```python
from isotope_core.providers.utils import (
    is_retryable_error,      # Check if error should be retried
    get_retry_after,         # Extract Retry-After from error
    map_error_to_stop_reason,  # Map exception to StopReason
    create_error_message,    # Create error AssistantMessage
    current_timestamp_ms,    # Unix timestamp in ms
)
```
