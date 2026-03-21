# TUI Concurrent Input & Steering Support

## Problem

The TUI currently uses a single-threaded input loop: it blocks on `async for event in agent.prompt()`, then asks for the next input. There's no way to inject steering or follow-up messages while the model is streaming a response. This means `agent.steer()` and `agent.follow_up()` — core features of isotopo-core — can't be exercised from the TUI.

## Goal

Add concurrent stdin reading during agent streaming so users can type commands while the model is responding. This enables:

1. **`/steer <msg>`** — inject steering mid-stream (calls `agent.steer()`)
2. **`/follow <msg>`** — queue a follow-up (calls `agent.follow_up()`)
3. **`/abort`** — abort mid-stream (calls `agent.abort()`)

## Design

### Concurrent Input Architecture

During `_send_message()`, run two concurrent tasks:

1. **Stream consumer** — reads events from `agent.prompt()` and prints them (existing logic)
2. **Input reader** — reads stdin in a background thread via `run_in_executor`, parses commands

When the stream finishes (agent_end event), cancel the input reader. When the input reader gets a command, dispatch it to the agent.

```python
async def _send_message(self, text: str) -> None:
    # Start streaming
    stream_task = asyncio.create_task(self._consume_stream(text))
    
    # Start concurrent input reader (only during streaming)
    input_task = asyncio.create_task(self._read_input_during_stream())
    
    # Wait for stream to finish; cancel input reader
    try:
        await stream_task
    finally:
        input_task.cancel()
        try:
            await input_task
        except asyncio.CancelledError:
            pass
```

### Input Reader During Stream

The input reader should:
- Show a subtle indicator that input is accepted (e.g., no prompt, or a dim `│ ` prefix)
- Accept `/steer <msg>`, `/follow <msg>`, `/abort`
- Ignore empty lines
- NOT accept regular messages (only slash commands during streaming)
- Use `run_in_executor(None, input)` for non-blocking stdin reads

### Threading Concern

`input()` in an executor blocks the thread until Enter is pressed. When the stream finishes, we `cancel()` the input task — but the thread is still blocked on `input()`. Options:

- **Option A (recommended):** Use `sys.stdin.readline()` in executor + set a threading Event to signal completion. After cancel, the blocked thread will return on next Enter press — acceptable UX.
- **Option B:** Use `select.select()` on `sys.stdin` with a timeout (Unix only, doesn't work on Windows).
- **Option C:** Use a separate thread with daemon=True that posts to an asyncio.Queue.

Go with **Option A** — it's simplest and the UX is fine (worst case, user presses Enter once after stream ends).

### Commands During Streaming

| Command | Action | Feedback |
|---------|--------|----------|
| `/steer <msg>` | `agent.steer(msg)` | Print `[steering: <msg>]` in tool style |
| `/follow <msg>` | `agent.follow_up(msg)` | Print `[follow-up queued: <msg>]` in tool style |
| `/abort` | `agent.abort()` | Print `[aborting...]` in warn style |
| Anything else | Ignore | Print `[only /steer, /follow, /abort during streaming]` in dim |

### Help Text Update

Update the `/help` output and the docstring to include the new commands. Note that `/steer`, `/follow`, `/abort` are only available during streaming.

## Files to Change

- `tui/main.py` — all changes here
- `tui/README.md` — update commands table

## Constraints

- No new dependencies
- Must work on macOS (primary) and Linux
- Don't break existing functionality (all current commands still work)
- Keep it simple — this is a dev testing tool, not a production TUI
- All 437 existing tests must still pass
- `ruff check` must be clean

## Branch

`feat/isotopo-core/dev-tui-steering`

## Testing

Manual testing only — verify:
1. Normal chat still works (type message, get response)
2. `/steer <msg>` during streaming injects and triggers new turn
3. `/follow <msg>` during streaming queues message for after completion
4. `/abort` during streaming stops the response
5. All existing slash commands still work between messages
