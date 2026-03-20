# isotopo-core

A Python library for building AI agent loops. Provides the core primitives for turn-based agent execution with tool use, streaming events, and multi-provider LLM support.

## Philosophy

- **Minimal and composable** — core loop logic only, no opinions on UI or deployment
- **Event-driven** — async event stream for every lifecycle phase
- **Provider-agnostic** — pluggable LLM backends via a unified streaming interface
- **Type-safe** — full type annotations with dataclass/Protocol patterns

## Architecture

```
isotopo-core/
├── src/isotopo_core/
│   ├── __init__.py
│   ├── types.py          # Core types: Message, Tool, Event, Context
│   ├── loop.py           # Agent loop: turn-based execution engine
│   ├── agent.py          # Stateful Agent class with event subscriptions
│   ├── tools.py          # Tool framework: schema, validation, execution
│   ├── context.py        # Context management: history, pruning, compression
│   ├── events.py         # Event stream: async generator + subscription model
│   └── providers/
│       ├── __init__.py
│       ├── base.py       # Provider protocol/ABC
│       ├── openai.py     # OpenAI provider
│       └── anthropic.py  # Anthropic provider
├── tests/
│   ├── __init__.py
│   ├── test_loop.py
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_events.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Core Concepts

### Agent Loop

The agent loop is the execution engine. It runs turn-by-turn:

1. Send user message to LLM
2. Stream assistant response (text + tool calls)
3. Execute tool calls (parallel or sequential)
4. Feed tool results back → next turn
5. Repeat until no more tool calls

```python
from isotopo_core import Agent, AgentConfig
from isotopo_core.providers.openai import OpenAIProvider

agent = Agent(
    config=AgentConfig(
        system_prompt="You are a helpful assistant.",
        provider=OpenAIProvider(model="gpt-4o", api_key="..."),
    )
)

async for event in agent.prompt("Hello!"):
    if event.type == "message_update":
        print(event.delta, end="")
```

### Events

Every phase of the agent loop emits typed events:

| Event | Description |
|-------|-------------|
| `agent_start` | Agent begins processing |
| `agent_end` | Agent completes |
| `turn_start` | New turn begins |
| `turn_end` | Turn completes |
| `message_start` | Message begins (user/assistant/tool_result) |
| `message_update` | Streaming delta (assistant only) |
| `message_end` | Message completes |
| `tool_start` | Tool execution begins |
| `tool_update` | Tool streams progress |
| `tool_end` | Tool execution completes |

### Tools

Tools are defined with typed schemas and async execute functions:

```python
from isotopo_core.tools import Tool, ToolResult

read_file = Tool(
    name="read_file",
    description="Read a file's contents",
    parameters={"path": {"type": "string", "description": "File path"}},
    execute=read_file_impl,
)
```

### Providers

LLM providers implement a simple streaming protocol:

```python
from isotopo_core.providers.base import Provider

class MyProvider(Provider):
    async def stream(self, context, options) -> AsyncGenerator[StreamEvent, None]:
        ...
```

## Installation

```bash
pip install isotopo-core
```

## Development

```bash
git clone https://github.com/GhostComplex/isotopo-core.git
cd isotopo-core
pip install -e ".[dev]"
pytest
```

## License

MIT
