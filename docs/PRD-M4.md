# isotope-core — Milestone 4: Steering, Follow-up & Session Control

## Objective

Add interactive control to the agent loop: interrupt mid-run, steer with new messages, queue follow-ups, resume after errors, and enforce budget limits.

## Prerequisites

- M1-M3 merged

## Deliverables

### 1. Steering Messages (`src/isotope_core/loop.py` + `agent.py`)

Allow injecting user messages mid-execution. When a steering message arrives:

1. The current LLM turn completes normally
2. Any pending tool calls from that turn still execute
3. The steering message is appended to context
4. A new turn starts with the steered context

**API:**
```python
agent.steer("Actually, use Python instead of TypeScript")
agent.steer(UserMessage(content=[TextContent(text="...")], timestamp=...))
```

**Implementation:**
- Add a `steering_queue: asyncio.Queue[Message]` to `AgentLoopConfig`
- The loop checks the queue after each tool execution phase
- If messages are queued, append them and start a new turn instead of ending
- Emit `SteerEvent` when a steering message is processed

### 2. Follow-up Queue

Queue messages that execute after the agent would normally stop (no more tool calls):

```python
agent.follow_up("Now summarize what you did")
```

**Behavior:**
- When agent reaches `end_turn` (no tool calls), check follow-up queue
- If messages queued, append and start a new turn
- If queue empty, actually end
- Emit `FollowUpEvent` when a follow-up triggers a new turn

### 3. Continue/Resume (`agent.py`)

```python
# Resume from current context (e.g., after an error or max_tokens)
async for event in agent.continue_():
    ...
```

- Works by sending the current message history back to the LLM
- Useful after `StopReason.MAX_TOKENS` to let the model continue
- Useful after `StopReason.ERROR` to retry
- Should NOT add a new user message — just re-sends existing context

### 4. Budget Limits

Add budget controls to `AgentLoopConfig`:

```python
@dataclass
class AgentLoopConfig:
    # ... existing fields ...
    max_turns: int | None = None          # Max number of turns
    max_total_tokens: int | None = None   # Max total tokens (input + output)
```

**Behavior:**
- Track turn count and cumulative token usage across all turns
- When a limit is hit, stop the loop gracefully
- `AgentEndEvent.reason` should reflect the limit: `"max_turns"`, `"max_budget"`
- Budget check happens at the start of each turn (before LLM call)

### 5. New Event Types

```python
@dataclass
class SteerEvent:
    type: Literal["steer"] = "steer"
    message: Message         # The steering message
    turn_number: int         # Which turn this happened in

@dataclass
class FollowUpEvent:
    type: Literal["follow_up"] = "follow_up"
    message: Message
    turn_number: int
```

Update `AgentEndEvent` with a `reason` field:

```python
@dataclass
class AgentEndEvent:
    type: Literal["agent_end"] = "agent_end"
    messages: list[Message]
    reason: str  # "completed", "aborted", "max_turns", "max_budget", "error"
```

### 6. Abort Improvements

- Ensure abort during tool execution emits proper events:
  - `ToolEndEvent` with `is_error=True` for any aborted tools
  - `TurnEndEvent`
  - `AgentEndEvent` with `reason="aborted"`
- Clean up any pending asyncio tasks on abort
- `agent.abort()` should be safe to call multiple times

### 7. Tests

- `tests/test_steering.py`:
  - Steer mid-run after tool execution
  - Multiple steering messages queued
  - Steer with no pending tool calls
  - Steer during text-only response
- `tests/test_followup.py`:
  - Follow-up triggers new turn after end_turn
  - Multiple follow-ups in sequence
  - Follow-up with empty queue (normal end)
  - Follow-up combined with steering
- `tests/test_budget.py`:
  - Max turns enforcement
  - Max tokens enforcement
  - Budget events and reasons
  - Budget with steering/follow-up
- `tests/test_continue.py`:
  - Continue after max_tokens
  - Continue after error
  - Continue with empty history (error)
- `tests/test_abort.py`:
  - Abort during tool execution
  - Abort during LLM streaming
  - Multiple abort calls
  - Event sequence on abort

## Technical Constraints

- Steering and follow-up must be thread-safe (use asyncio.Queue)
- Budget tracking must account for all turns including steered/follow-up turns
- Abort must clean up all pending tasks
- All new events must be added to the `AgentEvent` union type
- Type-safe — mypy strict

## Branch

`feat/isotope-core/dev-m4`

## Definition of Done

- [ ] Steering messages work mid-execution
- [ ] Follow-up queue triggers new turns after end_turn
- [ ] Continue/resume works after max_tokens and errors
- [ ] Max turns and max tokens budgets enforced
- [ ] AgentEndEvent has a reason field
- [ ] Abort is clean and idempotent
- [ ] All new event types emitted correctly
- [ ] All tests pass, ruff/mypy clean
