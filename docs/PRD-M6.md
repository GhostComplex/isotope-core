# isotope-core — Milestone 6: Hooks & Middleware

## Objective

Build a composable middleware/plugin system for the agent loop, enabling extensibility without modifying core code.

## Prerequisites

- M1-M5 merged

## Deliverables

### 1. Middleware Protocol (`src/isotope_core/middleware.py`)

A middleware layer that intercepts events flowing through the agent loop:

```python
class Middleware(Protocol):
    async def on_event(
        self,
        event: AgentEvent,
        context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None:
        """Process an event. Call next() to pass it through, or return None to suppress."""
        ...
```

```python
@dataclass
class MiddlewareContext:
    """Context available to middleware."""
    messages: list[Message]          # Current message history
    turn_number: int                 # Current turn number
    cumulative_tokens: int           # Total tokens used so far
    agent_config: AgentLoopConfig    # Loop configuration (read-only)
```

**Behavior:**
- Middleware forms a chain — each calls `next()` to pass the event to the next middleware
- Middleware can modify events before passing through
- Middleware can suppress events by returning `None` (not calling `next()`)
- Middleware can emit additional events by yielding them
- Order matters: first added = outermost (sees events first, results last)

### 2. Built-in Middleware

#### 2a. Logging Middleware

```python
class LoggingMiddleware:
    """Logs all agent events with configurable verbosity."""
    def __init__(
        self,
        logger: Callable[[str], None] | None = None,
        log_level: Literal["minimal", "normal", "verbose"] = "normal",
        include_content: bool = False,
    ): ...
```

- `minimal`: only agent_start, agent_end, turn_start, turn_end
- `normal`: + message_start, message_end, tool_start, tool_end
- `verbose`: + message_update, tool_update (streaming deltas)

#### 2b. Token Tracking Middleware

```python
class TokenTrackingMiddleware:
    """Tracks detailed token usage across turns."""
    def __init__(self) -> None: ...
    
    @property
    def total_usage(self) -> Usage: ...
    
    @property
    def per_turn_usage(self) -> list[Usage]: ...
    
    @property
    def turn_count(self) -> int: ...
```

Intercepts `MessageEndEvent` for assistant messages and accumulates usage.

#### 2c. Event Filter Middleware

```python
class EventFilterMiddleware:
    """Filters out specific event types."""
    def __init__(self, exclude: set[str]) -> None: ...
```

For consumers who don't want streaming deltas, tool updates, etc.

### 3. Lifecycle Hooks on Agent

Expand the Agent class with lifecycle hooks:

```python
class Agent:
    def __init__(
        self,
        ...
        on_agent_start: Callable[[], Awaitable[None]] | None = None,
        on_agent_end: Callable[[str], Awaitable[None]] | None = None,  # reason
        on_turn_start: Callable[[int], Awaitable[None]] | None = None,  # turn_number
        on_turn_end: Callable[[int, AssistantMessage], Awaitable[None]] | None = None,
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
        middleware: list[Middleware] | None = None,
    ): ...
```

### 4. Middleware Integration with Agent Loop

Add middleware support to `AgentLoopConfig`:

```python
@dataclass
class AgentLoopConfig:
    # ... existing fields ...
    middleware: list[Any] | None = None  # list[Middleware]
```

The loop wraps each yielded event through the middleware chain. If middleware returns `None`, the event is suppressed (not yielded to the caller).

### 5. Tests

- `tests/test_middleware.py`:
  - Single middleware passthrough
  - Multiple middleware chain ordering
  - Event modification by middleware
  - Event suppression (return None)
  - Middleware context has correct data
  - Error in middleware doesn't break loop
- `tests/test_logging_middleware.py`:
  - Minimal, normal, verbose log levels
  - Content inclusion/exclusion
  - Custom logger callback
- `tests/test_token_tracking.py`:
  - Total usage accumulation
  - Per-turn usage tracking
  - Turn count accuracy
- `tests/test_event_filter.py`:
  - Exclude specific event types
  - All events pass when no exclusions
- `tests/test_lifecycle_hooks.py`:
  - on_agent_start called
  - on_agent_end called with reason
  - on_turn_start/end called with correct args
  - on_error called on failures
  - Hooks are optional (no errors when None)

## Technical Constraints

- Middleware must be async (for I/O in middleware like logging to external services)
- Middleware errors must not crash the loop — log and continue
- The middleware chain must be set up once per loop run (not per event)
- Built-in middleware must work as reference implementations
- All type-safe — mypy strict
- Middleware protocol should use structural typing (Protocol), not ABC

## Branch

`feat/isotope-core/dev-m6`

## Definition of Done

- [ ] Middleware protocol defined and documented
- [ ] Middleware chain integration in agent_loop
- [ ] LoggingMiddleware, TokenTrackingMiddleware, EventFilterMiddleware implemented
- [ ] Lifecycle hooks on Agent class
- [ ] Middleware passes through agent loop events correctly
- [ ] All tests pass, ruff/mypy clean
