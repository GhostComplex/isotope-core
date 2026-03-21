# isotopo-core TUI

A standalone interactive TUI for testing isotopo-core against a local OpenAI-compatible proxy.

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

| Command            | Description                              |
| ------------------ | ---------------------------------------- |
| `/tools`           | Toggle tools on/off                      |
| `/model <name>`    | Switch model mid-session                 |
| `/system <prompt>` | Change system prompt                     |
| `/clear`           | Clear conversation history               |
| `/history`         | Show message count & total token usage   |
| `/debug`           | Toggle debug mode (shows event types)    |
| `/quit`            | Exit (also Ctrl+C)                       |

## Example Session

```
isotopo-core TUI v0.1
Proxy: http://localhost:4141/v1
Model: gpt-4o-mini

> Hello!
Hi there! How can I help you today?
[tokens: in=24, out=12]

> /tools
Tools enabled: read_file, write_file, edit_file, terminal, get_current_time

> What time is it?
  [calling get_current_time]
The current time is 2026-03-21T09:15:00+00:00.
[tokens: in=89, out=32]

> /quit
Bye!
```
