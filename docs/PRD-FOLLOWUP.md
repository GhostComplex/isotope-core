# isotope-core — Follow-up: Integration Tests & TUI

## Objective

Two follow-up items requested by Steins after M7 completion:
1. Integration tests using real API via `localhost:4141/v1` (OpenAI-compatible proxy)
2. A standalone TUI for manual interactive testing (NOT in core package)

## Prerequisites

- M1-M7 merged
- Proxy running at `localhost:4141/v1` (OpenAI-compatible, no API key needed)
- Available models include: `claude-sonnet-4`, `gpt-4o`, `gpt-5.4-mini`, etc.

## Deliverables

### 1. Integration Tests (`tests/integration/`)

Integration tests that hit the real proxy at `localhost:4141/v1`.

**File:** `tests/integration/test_real_proxy.py`

```python
# These tests are skipped by default (require running proxy).
# Run with: pytest tests/integration/ -m integration
# Or: ISOTOPO_INTEGRATION=1 pytest tests/integration/
```

**Test cases:**
- Basic text completion via ProxyProvider — send a simple prompt, verify streaming events (start, text_delta, done)
- Multi-turn conversation — send a prompt, get response, send follow-up, verify context carries
- Tool calling — register a simple tool (e.g., `get_current_time`), verify model calls it and receives result
- Agent class end-to-end — use Agent with ProxyProvider, verify full event flow
- Streaming fidelity — verify all expected event types appear in correct order
- Abort mid-stream — start a long generation, abort, verify clean stop
- Context management — use sliding window pruning with real token counts

**Configuration:**
- Use `pytest.mark.integration` marker
- Skip if proxy is unreachable (connection refused → skip)
- Default model: `gpt-4o-mini` or `gpt-5.4-mini` (cheap/fast)
- Add `conftest.py` with proxy availability check fixture

**Add to `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
markers = ["integration: integration tests requiring live proxy"]
```

### 2. TUI (`tui/`)

A standalone interactive TUI at the repo root (NOT inside `src/isotope_core/`).

**Location:** `tui/` directory at repo root
- `tui/main.py` — entry point
- `tui/README.md` — how to run

**Run:** `python -m tui.main` or `python tui/main.py` from repo root

**Features:**
- Connect to `localhost:4141/v1` using ProxyProvider
- Model selection on startup (list available models from `/v1/models`, let user pick or default)
- System prompt input (optional, with sensible default)
- Interactive chat loop:
  - Type messages, see streaming responses (character by character or chunk by chunk)
  - Show token usage after each turn
  - `/tools` — toggle example tools (get_time, calculator)
  - `/model <name>` — switch model mid-session
  - `/system <prompt>` — change system prompt
  - `/clear` — clear conversation history
  - `/history` — show message count and token usage
  - `/quit` or Ctrl+C — exit cleanly
- Color output (use `rich` library or ANSI codes — prefer `rich` if available, fallback to plain)
- Show event types in debug mode (`/debug` toggle)

**Dependencies:** Only `isotope-core` itself + `httpx` (for model listing) + optionally `rich` for pretty output. NO `textual` or heavy TUI frameworks — keep it simple stdin/stdout.

**Example session:**
```
isotope-core TUI v0.1
Proxy: localhost:4141/v1
Model: gpt-5.4-mini

System prompt (Enter to skip): You are a helpful assistant.

> Hello, what can you do?
I can help you with a variety of tasks...
[tokens: in=24, out=45]

> /tools
Tools enabled: get_current_time, calculator

> What time is it?
Let me check... [calling get_current_time]
The current time is 2026-03-21 09:15:00 UTC.
[tokens: in=89, out=32]

> /quit
Bye!
```

## Branch

`feat/isotope-core/dev-followup`

## Technical Constraints

- Integration tests must be skippable (don't break CI without proxy)
- TUI must NOT be part of the `isotope-core` package (not in `src/`)
- TUI must work with just `uv sync` from the repo (no extra install)
- Keep it simple — stdin/stdout, no curses/textual
- ruff + mypy clean (TUI files can be excluded from mypy if needed)

## Definition of Done

- [ ] Integration tests pass against `localhost:4141/v1`
- [ ] Integration tests skip cleanly when proxy unavailable
- [ ] TUI runs interactively with streaming output
- [ ] Model selection, tool toggling, system prompt changes work
- [ ] TUI shows token usage per turn
- [ ] All existing tests still pass
- [ ] ruff clean
