# isotopo-core TUI

A standalone interactive TUI for testing isotopo-core against a local OpenAI-compatible proxy.

## Prerequisites

- isotopo-core installed in editable mode: `pip install -e ".[openai,dev]"`
- `httpx` installed: `pip install httpx`
- (Optional) `rich` for colored output: `pip install rich`
- A proxy running at `http://localhost:4141/v1`

## Running

From the repository root:

```bash
python tui/main.py
```

Or:

```bash
python -m tui.main
```

## Commands

| Command            | Description                              |
| ------------------ | ---------------------------------------- |
| `/tools`           | Toggle example tools (time, calculator)  |
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

System prompt (Enter to skip): You are a helpful assistant.

> Hello!
Hi there! How can I help you today?
[tokens: in=24, out=12]

> /tools
Tools enabled: get_current_time, calculator

> What time is it?
  [calling get_current_time]
The current time is 2026-03-21T09:15:00+00:00.
[tokens: in=89, out=32]

> /quit
Bye!
```
