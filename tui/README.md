# isotope-core TUI

A standalone interactive TUI for testing isotope-core against a local OpenAI-compatible proxy.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- A proxy running at `http://localhost:4141/v1`

## Running

From the repository root:

```bash
uv run python tui/main.py
```

Tools are disabled by default. Type `/tools` to enable:
- `read_file` — read file contents
- `write_file` — create/overwrite files
- `edit_file` — find-and-replace exact text
- `terminal` — run shell commands
- `get_current_time` — current UTC time

## Commands

### Between messages

| Command            | Description                              |
| ------------------ | ---------------------------------------- |
| `/tools`           | Toggle tools on/off                      |
| `/model <name>`    | Switch model mid-session                 |
| `/system <prompt>` | Change system prompt                     |
| `/clear`           | Clear conversation history               |
| `/history`         | Show message count & total token usage   |
| `/debug`           | Toggle debug mode (shows event types)    |
| `/help`            | Show all available commands              |
| `/quit`            | Exit (also Ctrl+C)                       |

### During streaming

These commands can be typed while the model is generating a response:

| Command            | Description                                          |
| ------------------ | ---------------------------------------------------- |
| Any text           | Cancel the current stream and use that text to steer |
| `/follow <msg>`    | Queue a follow-up message for after completion       |
| `/abort`           | Abort the current response                           |

**How it works:** During streaming, a footer input accepts follow-ups, aborts, and free-text steering. Any non-slash text interrupts the current stream, preserves the partial assistant response in history, and starts a new turn with your text. `/follow` queues a message that triggers after the model would normally stop. `/abort` stops the current response immediately.

## Example Session

```
isotope-core TUI v0.1
Proxy: http://localhost:4141/v1
Model: claude-opus-4.6

> Hello!
Hi there! How can I help you today?
[tokens: in=24, out=12]

> /tools
Tools enabled: read_file, write_file, edit_file, terminal, get_current_time

> What time is it?
  [calling get_current_time]
The current time is 2026-03-21T09:15:00+00:00.
[tokens: in=89, out=32]

> Write me a long essay about AI
The history of artificial intelligence...
Actually, make it about robotics instead
  [→ Actually, make it about robotics instead]
Let me pivot to robotics...
[tokens: in=156, out=89]

> Tell me more
Robotics has evolved significantly...
/abort
  [aborting...]
[tokens: in=200, out=45]

> /quit
Bye!
```
