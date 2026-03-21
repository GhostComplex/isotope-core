"""isotopo-core TUI — interactive chat against a local proxy.

Usage:
    python tui/main.py              # from repo root
    python -m tui.main              # from repo root

Commands:
    /tools          Toggle example tools (get_current_time, calculator)
    /model <name>   Switch model
    /system <text>  Change system prompt
    /clear          Clear conversation history
    /history        Show message count & token usage
    /debug          Toggle debug mode (shows event types)
    /quit           Exit
"""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
from typing import Any

# Bypass system HTTP proxies (e.g. Clash) for localhost
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

# ---------------------------------------------------------------------------
# Optional rich support
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.theme import Theme

    _theme = Theme(
        {
            "info": "dim cyan",
            "warn": "bold yellow",
            "err": "bold red",
            "model": "bold green",
            "user": "bold blue",
            "tool": "bold magenta",
            "dim": "dim",
        }
    )
    _console = Console(theme=_theme)

    def _print(text: str, style: str | None = None, **kw: Any) -> None:
        _console.print(text, style=style, end=kw.get("end", "\n"), highlight=False)

    def _print_inline(text: str, style: str | None = None) -> None:
        _console.print(text, style=style, end="", highlight=False)

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    def _print(text: str, style: str | None = None, **kw: Any) -> None:
        print(text, end=kw.get("end", "\n"))

    def _print_inline(text: str, style: str | None = None) -> None:
        print(text, end="", flush=True)


# ---------------------------------------------------------------------------
# Imports from isotopo-core
# ---------------------------------------------------------------------------

from isotopo_core import Agent  # noqa: E402
from isotopo_core.providers.proxy import ProxyProvider  # noqa: E402
from isotopo_core.tools import Tool, ToolResult  # noqa: E402
from isotopo_core.types import AssistantMessage  # noqa: E402

PROXY_BASE_URL = "http://localhost:4141/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


def _make_tools() -> list[Tool]:
    """Create the built-in example tools."""

    async def _get_time(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        now = datetime.datetime.now(tz=datetime.UTC).isoformat()
        return ToolResult.text(f"Current UTC time: {now}")

    async def _calculator(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        expr = params.get("expression", "")
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in str(expr)):
            return ToolResult.error(f"Invalid expression: {expr}")
        try:
            return ToolResult.text(str(eval(str(expr))))  # noqa: S307
        except Exception as e:
            return ToolResult.error(str(e))

    return [
        Tool(
            name="get_current_time",
            description="Get the current date and time in UTC",
            parameters={"type": "object", "properties": {}},
            execute=_get_time,
        ),
        Tool(
            name="calculator",
            description="Evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '2 + 2'",
                    }
                },
                "required": ["expression"],
            },
            execute=_calculator,
        ),
    ]


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


async def _fetch_models(base_url: str) -> list[str]:
    """Fetch available models from the proxy."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            models: list[str] = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if mid:
                    models.append(mid)
            return sorted(models)
    except Exception as exc:
        _print(f"Warning: could not fetch models: {exc}", style="warn")
        return []


# ---------------------------------------------------------------------------
# Main TUI
# ---------------------------------------------------------------------------


class TUI:
    """Interactive TUI for isotopo-core."""

    def __init__(self) -> None:
        self.model = DEFAULT_MODEL
        self.system_prompt = ""
        self.tools_enabled = False
        self.debug = False
        self.tools = _make_tools()
        self.agent: Agent | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _create_agent(self) -> Agent:
        provider = ProxyProvider(
            model=self.model,
            base_url=PROXY_BASE_URL,
            api_key="not-needed",
        )
        return Agent(
            provider=provider,
            system_prompt=self.system_prompt,
            tools=self.tools if self.tools_enabled else [],
        )

    def _rebuild_agent(self, *, keep_history: bool = True) -> None:
        """Rebuild the agent (e.g. after model / tool change)."""
        old_messages = self.agent.messages[:] if self.agent and keep_history else []
        self.agent = self._create_agent()
        if old_messages:
            self.agent.replace_messages(old_messages)

    async def _select_model(self, models: list[str]) -> str:
        """Let the user pick a model or accept the default."""
        if models:
            _print("\nAvailable models:", style="info")
            for i, m in enumerate(models, 1):
                marker = " (default)" if m == DEFAULT_MODEL else ""
                _print(f"  {i}. {m}{marker}", style="dim")

            _print(f"\nSelect model [Enter for {DEFAULT_MODEL}]: ", style="info", end="")
            try:
                choice = await asyncio.get_event_loop().run_in_executor(None, input)
            except (EOFError, KeyboardInterrupt):
                return DEFAULT_MODEL

            choice = choice.strip()
            if not choice:
                return DEFAULT_MODEL

            # Accept number or name
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    return models[idx]
            except ValueError:
                pass

            # Accept partial name match
            for m in models:
                if choice.lower() in m.lower():
                    return m

            _print(f"Unknown model '{choice}', using {DEFAULT_MODEL}", style="warn")
            return DEFAULT_MODEL
        return DEFAULT_MODEL

    async def _get_system_prompt(self) -> str:
        """Prompt user for system prompt."""
        _print("\nSystem prompt (Enter to skip): ", style="info", end="")
        try:
            prompt = await asyncio.get_event_loop().run_in_executor(None, input)
        except (EOFError, KeyboardInterrupt):
            return ""
        return prompt.strip()

    async def _handle_command(self, line: str) -> bool:
        """Handle a slash command. Returns True if handled."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit":
            _print("Bye!", style="info")
            return True

        if cmd == "/tools":
            self.tools_enabled = not self.tools_enabled
            if self.tools_enabled:
                names = ", ".join(t.name for t in self.tools)
                _print(f"Tools enabled: {names}", style="tool")
            else:
                _print("Tools disabled", style="tool")
            self._rebuild_agent()
            return False

        if cmd == "/model":
            if arg:
                self.model = arg
                _print(f"Model switched to: {self.model}", style="model")
                self._rebuild_agent()
            else:
                _print("Usage: /model <name>", style="warn")
            return False

        if cmd == "/system":
            if arg:
                self.system_prompt = arg
                _print("System prompt updated.", style="info")
                self._rebuild_agent()
            else:
                _print("Usage: /system <prompt>", style="warn")
            return False

        if cmd == "/clear":
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self._rebuild_agent(keep_history=False)
            _print("Conversation cleared.", style="info")
            return False

        if cmd == "/history":
            if self.agent:
                msg_count = len(self.agent.messages)
                _print(f"Messages: {msg_count}", style="info")
            _print(
                f"Total tokens: in={self.total_input_tokens}, out={self.total_output_tokens}",
                style="info",
            )
            return False

        if cmd == "/debug":
            self.debug = not self.debug
            _print(f"Debug mode: {'on' if self.debug else 'off'}", style="info")
            return False

        _print(f"Unknown command: {cmd}", style="warn")
        _print("Commands: /tools /model /system /clear /history /debug /quit", style="dim")
        return False

    async def _send_message(self, text: str) -> None:
        """Send a user message and stream the response."""
        if self.agent is None:
            self._rebuild_agent()
        assert self.agent is not None

        try:
            async for event in self.agent.prompt(text):
                if self.debug:
                    _print(f"  [{event.type}]", style="dim")

                if event.type == "message_update":
                    delta = getattr(event, "delta", None)
                    if delta:
                        _print_inline(delta, style="model")

                elif event.type == "tool_start":
                    tool_name = getattr(event, "tool_name", "?")
                    _print(f"\n  [calling {tool_name}]", style="tool")

                elif event.type == "tool_end":
                    is_error = getattr(event, "is_error", False)
                    if is_error:
                        _print("  [tool error]", style="err")

                elif event.type == "turn_end":
                    msg = getattr(event, "message", None)
                    if isinstance(msg, AssistantMessage):
                        self.total_input_tokens += msg.usage.input_tokens
                        self.total_output_tokens += msg.usage.output_tokens

                elif event.type == "agent_end":
                    pass  # handled below

        except Exception as exc:
            _print(f"\nError: {exc}", style="err")
            return

        # Print newline after streamed text + token usage
        print()
        # Show usage for last assistant message
        assistant_msgs = [
            m for m in self.agent.messages if isinstance(m, AssistantMessage)
        ]
        if assistant_msgs:
            usage = assistant_msgs[-1].usage
            _print(
                f"[tokens: in={usage.input_tokens}, out={usage.output_tokens}]",
                style="dim",
            )

    async def run(self) -> None:
        """Main TUI loop."""
        _print("isotopo-core TUI v0.1", style="info")
        _print(f"Proxy: {PROXY_BASE_URL}", style="dim")

        # Fetch and select model
        models = await _fetch_models(PROXY_BASE_URL)
        self.model = await self._select_model(models)
        _print(f"Model: {self.model}", style="model")

        # System prompt
        self.system_prompt = await self._get_system_prompt()
        if self.system_prompt:
            _print(f"System prompt: {self.system_prompt}", style="dim")

        # Create agent
        self.agent = self._create_agent()

        _print("\nType your message (or /help for commands). Ctrl+C to quit.\n", style="dim")

        loop = asyncio.get_event_loop()
        while True:
            try:
                _print_inline("> ", style="user")
                line = await loop.run_in_executor(None, input)
            except (EOFError, KeyboardInterrupt):
                print()
                _print("Bye!", style="info")
                break

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                should_quit = await self._handle_command(line)
                if should_quit:
                    break
                continue

            await self._send_message(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the TUI."""
    try:
        asyncio.run(TUI().run())
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
