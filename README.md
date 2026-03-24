# isotope-core

Core primitives for building AI agent loops in Python. Provides a turn-based execution engine with tool use, streaming events, context management, middleware, and multi-provider LLM support.

## Installation

```bash
# Core (no provider SDKs)
uv add isotope-core

# With extras
uv add "isotope-core[openai]"          # OpenAI provider
uv add "isotope-core[anthropic]"       # Anthropic provider
uv add "isotope-core[tiktoken]"        # Accurate token counting
uv add "isotope-core[openai,anthropic,tiktoken]"  # Everything
```

## Quick Start

```python
import asyncio
from isotope_core import Agent
from isotope_core.providers.anthropic import AnthropicProvider

agent = Agent(
    provider=AnthropicProvider(model="claude-sonnet-4-20250514"),
    system_prompt="You are a helpful assistant.",
)

async def main():
    async for event in agent.prompt("Hello!"):
        if event.type == "message_update" and event.delta:
            print(event.delta, end="")
    print()

asyncio.run(main())
```

## Architecture

```
isotope-core/
├── src/isotope_core/
│   ├── __init__.py        # Public API re-exports
│   ├── types.py           # Core types: Message, Content, Event, Context
│   ├── loop.py            # Agent loop: turn-based execution engine
│   ├── agent.py           # Stateful Agent class
│   ├── tools.py           # Tool framework: schema, validation, execution
│   ├── context.py         # Context management: token counting, pruning
│   ├── events.py          # Event stream: async generator + subscriptions
│   ├── middleware.py       # Middleware system: logging, token tracking, filtering
│   ├── py.typed           # PEP 561 type marker
│   └── providers/
│       ├── __init__.py    # Provider re-exports
│       ├── base.py        # Provider protocol + stream event types
│       ├── openai.py      # OpenAI provider
│       ├── anthropic.py   # Anthropic provider (with extended thinking)
│       ├── proxy.py       # OpenAI-compatible proxy provider
│       ├── router.py      # Router with fallback + circuit breaker
│       └── utils.py       # Retry logic, error mapping
├── tests/                 # 437+ tests, 97% coverage
├── examples/              # Runnable examples (no API keys needed)
├── docs/
│   └── API.md             # Full API reference
└── pyproject.toml
```

## Core Concepts

### Agent Loop

The agent loop is the execution engine. It runs turn-by-turn:

1. Send user message(s) to the LLM
2. Stream the assistant response (text, thinking, tool calls)
3. Execute tool calls (parallel or sequential)
4. Feed tool results back → next turn
5. Repeat until no more tool calls (or budget exceeded)

```python
from isotope_core import agent_loop, AgentLoopConfig
from isotope_core.types import Context, UserMessage, TextContent

config = AgentLoopConfig(
    provider=my_provider,
    tools=[read_file, write_file],
    max_turns=10,
    max_total_tokens=100_000,
)

context = Context(system_prompt="You are a coding assistant.")
prompts = [UserMessage(content=[TextContent(text="Fix the bug")], timestamp=0)]

async for event in agent_loop(prompts, context, config):
    print(event.type)
```

### Agent (Stateful Wrapper)

The `Agent` class wraps the loop with state management, subscriptions, steering, and follow-up queues:

```python
from isotope_core import Agent

agent = Agent(
    provider=my_provider,
    system_prompt="You are helpful.",
    tools=[my_tool],
    max_turns=20,
)

# Send a prompt
async for event in agent.prompt("Hello!"):
    ...

# Continue from current context
async for event in agent.continue_():
    ...

# Inject steering mid-loop
agent.steer("Focus on error handling.")

# Queue follow-up for after completion
agent.follow_up("Now write tests.")

# Abort
agent.abort()
```

### Events

Every phase of the agent loop emits typed events:

| Event | Description |
|-------|-------------|
| `agent_start` | Agent begins processing |
| `agent_end` | Agent completes (with reason) |
| `turn_start` | New turn begins |
| `turn_end` | Turn completes |
| `message_start` | Message begins (user/assistant/tool_result) |
| `message_update` | Streaming delta (assistant only) |
| `message_end` | Message completes |
| `tool_start` | Tool execution begins |
| `tool_update` | Tool streams progress |
| `tool_end` | Tool execution completes |
| `context_pruned` | Context was pruned |
| `steer` | Steering message injected |
| `follow_up` | Follow-up message queued |

### Tools

Tools are defined with typed schemas and async execute functions:

```python
from isotope_core.tools import Tool, ToolResult, tool

# Class-based
read_file = Tool(
    name="read_file",
    description="Read a file's contents",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    execute=read_file_impl,
)

# Decorator-based
@tool(name="get_weather", description="Get weather", parameters={...})
async def get_weather(tool_call_id, params, signal, on_update):
    return ToolResult.text(f"Weather in {params['location']}: Sunny")
```

### Providers

LLM providers implement the `Provider` protocol:

```python
from isotope_core.providers.base import Provider

class MyProvider(Provider):
    @property
    def model_name(self) -> str: return "my-model"

    @property
    def provider_name(self) -> str: return "my-provider"

    async def stream(self, context, *, temperature=None, max_tokens=None, signal=None):
        yield StreamStartEvent(partial=msg)
        yield StreamTextDeltaEvent(...)
        yield StreamDoneEvent(message=msg)
```

Built-in providers:
- **`OpenAIProvider`** — OpenAI Chat Completions API
- **`AnthropicProvider`** — Anthropic Messages API (with extended thinking)
- **`ProxyProvider`** — Any OpenAI-compatible endpoint (LiteLLM, Ollama, vLLM, etc.)
- **`RouterProvider`** — Multi-provider routing with fallback and circuit breaker

### Context Management

Token counting, pruning strategies, and context transforms:

```python
from isotope_core import (
    count_tokens,
    estimate_context_usage,
    SlidingWindowStrategy,
    create_sliding_window_transform,
    pin_message,
)

# Count tokens
total = count_tokens(messages, model="gpt-4o")

# Estimate context usage
usage = estimate_context_usage(context, model="gpt-4o")
print(f"Utilization: {usage.utilization:.0%}")

# Prune with sliding window
strategy = SlidingWindowStrategy(keep_recent=10, keep_first_n=1)
result = await strategy.prune(messages, target_tokens=50_000)

# Pin important messages (survive pruning)
messages = pin_message(messages, index=0)

# Use as a transform_context hook
agent = Agent(
    provider=my_provider,
    transform_context=create_sliding_window_transform(max_tokens=50_000),
)
```

### Middleware

Composable middleware chain for event interception:

```python
from isotope_core import (
    LoggingMiddleware,
    TokenTrackingMiddleware,
    EventFilterMiddleware,
)

agent = Agent(
    provider=my_provider,
    middleware=[
        LoggingMiddleware(log_level="minimal"),
        TokenTrackingMiddleware(),
        EventFilterMiddleware(exclude={"message_update"}),
    ],
)
```

Custom middleware:

```python
class RateLimiterMiddleware:
    async def on_event(self, event, context, next):
        # Pre-processing
        result = await next(event)
        # Post-processing
        return result
```

### Router & Circuit Breaker

```python
from isotope_core import RouterProvider

router = RouterProvider(
    primary=openai_provider,
    fallbacks=[anthropic_provider],
    circuit_breaker_threshold=3,
    circuit_breaker_timeout=120.0,
)

# Automatic fallback on retryable errors
agent = Agent(provider=router)

# Track usage across providers
usage = router.get_usage()
print(usage.provider_usage)

# Dynamically switch primary
router.set_primary(new_provider)
```

## Examples

All examples use mock providers and require no API keys:

- [`examples/basic_agent.py`](examples/basic_agent.py) — Minimal agent with a tool
- [`examples/streaming.py`](examples/streaming.py) — Consuming the event stream
- [`examples/context_management.py`](examples/context_management.py) — Pruning strategies
- [`examples/middleware.py`](examples/middleware.py) — Custom middleware
- [`examples/multi_provider.py`](examples/multi_provider.py) — Router with fallback

## Development

```bash
git clone https://github.com/GhostComplex/isotope-core.git
cd isotope-core
uv sync --all-extras

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=isotope_core --cov-fail-under=90

# Lint
uv run ruff check src/ tests/ examples/

# Type check
uv run mypy src/
```

## License

MIT
