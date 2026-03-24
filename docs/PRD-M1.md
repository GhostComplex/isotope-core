# isotope-core — Milestone 1: Core Types & Agent Loop

## Objective

Implement the foundational types and the core agent loop that drives turn-based LLM interaction with tool execution.

## Reference Repos

Study these repos (cloned at `/tmp/`) for design patterns:
- `/tmp/pi-mono/packages/agent/` — Agent loop, types, event system
- `/tmp/pi-mono/packages/ai/` — Provider abstraction, streaming types
- `/tmp/gemini-cli/packages/core/src/agent/` — Agent protocol, event types

## Deliverables

### 1. Core Types (`src/isotope_core/types.py`)

Define all foundational types using Pydantic models and Python Protocols:

**Messages:**
- `TextContent` — text content block
- `ImageContent` — image content block (base64 data + mime type)
- `ThinkingContent` — reasoning/thinking content block
- `ToolCallContent` — tool call content block (id, name, arguments)
- `UserMessage` — role="user", content list
- `AssistantMessage` — role="assistant", content list, stop_reason, usage metadata
- `ToolResultMessage` — role="tool_result", tool_call_id, content, is_error flag
- `Message` — union type of all message types
- `Usage` — input_tokens, output_tokens, cache_read_tokens, cache_write_tokens

**Context:**
- `Context` — system_prompt, messages list, tools list

**Events (discriminated union on `type` field):**
- `AgentStartEvent`, `AgentEndEvent`
- `TurnStartEvent`, `TurnEndEvent`
- `MessageStartEvent`, `MessageUpdateEvent`, `MessageEndEvent`
- `ToolStartEvent`, `ToolUpdateEvent`, `ToolEndEvent`
- `AgentEvent` — union of all event types

**Stop reasons:** `end_turn`, `tool_use`, `max_tokens`, `error`, `aborted`

### 2. Provider Protocol (`src/isotope_core/providers/base.py`)

Define the provider interface:

```python
class StreamEvent:
    """Events yielded by provider.stream()"""
    # start, text_delta, text_end, thinking_delta, tool_call_start, tool_call_delta, tool_call_end, done, error

class Provider(Protocol):
    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        ...
```

### 3. Tool Framework (`src/isotope_core/tools.py`)

```python
@dataclass
class ToolResult:
    content: list[TextContent | ImageContent]
    is_error: bool = False

class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    execute: Callable  # async (tool_call_id, params, signal?) -> ToolResult
```

### 4. Event Stream (`src/isotope_core/events.py`)

Async event stream that supports:
- `async for event in stream:` iteration
- `subscribe(callback)` for push-based consumption
- Proper cleanup on abort

### 5. Agent Loop (`src/isotope_core/loop.py`)

The core execution engine:

```python
async def agent_loop(
    prompts: list[Message],
    context: Context,
    config: AgentLoopConfig,
) -> AsyncGenerator[AgentEvent, None]:
```

Loop logic:
1. Emit `agent_start`
2. Emit `turn_start`
3. Emit `message_start/end` for user messages
4. Stream assistant response → emit `message_start`, `message_update`*, `message_end`
5. If tool calls: execute them (parallel by default) → emit `tool_start/update/end`
6. Emit `turn_end`
7. If more tool calls → goto 2
8. Emit `agent_end`

Config options:
- `tool_execution: "parallel" | "sequential"`
- `before_tool_call: async hook → allow/block`
- `after_tool_call: async hook → modify result`
- `transform_context: async hook → prune/modify messages before LLM call`

### 6. Stateful Agent (`src/isotope_core/agent.py`)

```python
class Agent:
    state: AgentState  # system_prompt, provider, tools, messages, is_streaming
    
    async def prompt(text, images?) -> AsyncGenerator[AgentEvent]
    async def continue_() -> AsyncGenerator[AgentEvent]
    def abort()
    def subscribe(callback) -> unsubscribe
    def set_system_prompt(...)
    def set_tools(...)
    def replace_messages(...)
    def clear_messages()
    def reset()
```

### 7. Tests

- `tests/test_types.py` — Message construction, serialization
- `tests/test_tools.py` — Tool execution, error handling
- `tests/test_loop.py` — Full loop with mock provider
- `tests/test_agent.py` — Agent state management, prompt/abort

## Technical Constraints

- Python 3.11+
- Pydantic v2 for types
- Pure async/await — no threads
- No external dependencies beyond pydantic (providers are optional extras)
- All public APIs must have type annotations
- Use `asyncio.Event` for abort signaling (not `threading.Event`)

## Branch

`feat/isotope-core/dev-m1`

## Definition of Done

- [ ] All types defined and importable from `isotope_core`
- [ ] Provider protocol defined with clear streaming contract
- [ ] Tool framework with schema validation
- [ ] Agent loop passes tests with mock provider
- [ ] Agent class wraps loop with state management
- [ ] All tests pass
- [ ] `ruff check` and `mypy` clean
- [ ] README examples are accurate
