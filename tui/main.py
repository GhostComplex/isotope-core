"""isotope-core TUI — interactive chat against a local proxy.

Usage:
    python tui/main.py              # from repo root
    python -m tui.main              # from repo root

Commands (between messages):
    /tools          Toggle tools (read_file, write_file, edit_file, terminal)
    /model <name>   Switch model
    /system <text>  Change system prompt
    /clear          Clear conversation history
    /history        Show message count & token usage
    /debug          Toggle debug mode (shows event types)
    /help           Show available commands
    /quit           Exit

Commands (during streaming):
    Any text       Steering — cancels stream and queues your message (Claude Code style)
    /follow <msg>  Queue a follow-up for after completion
    /abort         Abort the current response
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncGenerator
from typing import Any

# Bypass system HTTP proxies (e.g. Clash) for localhost
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

# ---------------------------------------------------------------------------
# Output helpers (plain stdout)
# ---------------------------------------------------------------------------

def _print(text: str, style: str | None = None, **kw: Any) -> None:
    del style
    print(text, end=kw.get("end", "\n"))


def _print_inline(text: str, style: str | None = None) -> None:
    del style
    print(text, end="", flush=True)


# ---------------------------------------------------------------------------
# Optional prompt_toolkit support (Claude Code style input during streaming)
# ---------------------------------------------------------------------------

try:
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.application import Application as _Application
    from prompt_toolkit.buffer import Buffer as _Buffer
    from prompt_toolkit.document import Document as _Document
    from prompt_toolkit.formatted_text import HTML as _HTML
    from prompt_toolkit.key_binding import KeyBindings as _KeyBindings
    from prompt_toolkit.layout.containers import HSplit as _HSplit
    from prompt_toolkit.layout.containers import Window as _Window
    from prompt_toolkit.layout.controls import BufferControl as _BufferControl
    from prompt_toolkit.layout.controls import FormattedTextControl as _FormattedTextControl
    from prompt_toolkit.layout.layout import Layout as _Layout
    from prompt_toolkit.layout.processors import BeforeInput as _BeforeInput
    from prompt_toolkit.patch_stdout import patch_stdout as _patch_stdout

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

# ---------------------------------------------------------------------------
# Imports from isotope-core
# ---------------------------------------------------------------------------

from isotope_core import Agent  # noqa: E402
from isotope_core.providers.proxy import ProxyProvider  # noqa: E402
from isotope_core.tools import Tool, ToolResult  # noqa: E402
from isotope_core.types import AgentEvent, AssistantMessage  # noqa: E402

PROXY_BASE_URL = "http://localhost:4141/v1"
DEFAULT_MODEL = "claude-opus-4.6"

# Workspace directory — all relative file paths are resolved against this.
WORKSPACE = os.getcwd()


def _resolve_path(path: str) -> str:
    """Resolve a tool path to an absolute path.

    - Paths starting with ``~`` are expanded via :func:`os.path.expanduser`.
    - Absolute paths are returned as-is.
    - Relative paths are joined with :data:`WORKSPACE` so that the LLM's
      relative references always land in the correct directory regardless
      of the process's *cwd*.
    """
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE, path)
    return path

_SYSTEM_PROMPT_TEMPLATE = """\
You are isotope, an expert software engineer assistant. You help users with \
coding tasks including writing, reading, editing, and debugging code.

Your workspace directory is {cwd} — use this as the base for all \
file operations unless the user specifies an absolute path.

You have access to the following tools:

1. **read_file** — Read a file's contents.
   Parameters: path (string, required) — absolute or relative file path.

2. **write_file** — Create or overwrite a file. Creates parent directories \
if needed.
   Parameters: path (string, required), content (string, required).

3. **edit_file** — Edit a file by replacing an exact text match. The \
old_text must appear exactly once in the file.
   Parameters: path (string, required), old_text (string, required), \
new_text (string, required).

4. **terminal** — Execute a shell command and return stdout/stderr. \
Timeout defaults to 30s, max 120s.
   Parameters: command (string, required), timeout (number, optional).

5. **get_current_time** — Get the current date and time in UTC.

Guidelines:
- Read files before editing to understand context.
- Use edit_file for surgical changes; use write_file only for new files or \
full rewrites.
- Prefer relative paths from the workspace directory when possible.
- Keep responses concise and actionable.
"""

DEFAULT_SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(cwd=WORKSPACE)

BETWEEN_MESSAGE_COMMANDS = (
    "/tools          Toggle tools",
    "/model <name>   Switch model",
    "/system <text>  Change system prompt",
    "/clear          Clear conversation",
    "/history        Show usage stats",
    "/debug          Toggle debug mode",
    "/help           Show available commands",
    "/quit           Exit",
)

DURING_STREAMING_COMMANDS = (
    "Any text       Steering — cancels stream, queues your message",
    "/follow <msg>  Queue follow-up for after completion",
    "/abort         Abort current response",
)

# When prompt_toolkit is active, streaming output must be printed as complete
# lines for patch_stdout to render them correctly above the prompt.
# This helper buffers partial deltas and flushes only on newline boundaries.


class _StreamBuffer:
    """Buffer streaming text, print complete lines only."""

    def __init__(self) -> None:
        self._pending = ""

    def write(self, text: str) -> None:
        """Buffer text. Immediately print each complete line."""
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            print(line)

    def flush(self) -> None:
        """Print any remaining buffered text."""
        if self._pending:
            print(self._pending)
            self._pending = ""

    def drain(self) -> str:
        """Return buffered text and clear it without printing."""
        pending = self._pending
        self._pending = ""
        return pending

    def discard(self) -> None:
        """Discard buffered text without printing (used on cancellation)."""
        self._pending = ""


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


def _make_tools() -> list[Tool]:
    """Create the built-in tools: read_file, write_file, edit_file, terminal, get_current_time."""

    async def _read_file(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        path = params.get("path", "")
        if not path:
            return ToolResult.error("Missing required parameter: path")
        path = _resolve_path(path)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Truncate very large files
            if len(content) > 100_000:
                content = content[:100_000] + f"\n\n... [truncated, {len(content)} chars total]"
            return ToolResult.text(content)
        except FileNotFoundError:
            return ToolResult.error(f"File not found: {path}")
        except PermissionError:
            return ToolResult.error(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error(f"Error reading file: {e}")

    async def _write_file(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return ToolResult.error("Missing required parameter: path")
        path = _resolve_path(path)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult.text(f"Written {len(content)} chars to {path}")
        except PermissionError:
            return ToolResult.error(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error(f"Error writing file: {e}")

    async def _edit_file(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        path = params.get("path", "")
        old_text = params.get("old_text", "")
        new_text = params.get("new_text", "")
        if not path:
            return ToolResult.error("Missing required parameter: path")
        if not old_text:
            return ToolResult.error("Missing required parameter: old_text")
        path = _resolve_path(path)
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            count = content.count(old_text)
            if count == 0:
                return ToolResult.error(
                    f"old_text not found in {path}. Make sure it matches exactly."
                )
            if count > 1:
                return ToolResult.error(
                    f"old_text found {count} times in {path}. Must match exactly once."
                )
            content = content.replace(old_text, new_text, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult.text(f"Edited {path}: replaced 1 occurrence")
        except FileNotFoundError:
            return ToolResult.error(f"File not found: {path}")
        except PermissionError:
            return ToolResult.error(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error(f"Error editing file: {e}")

    async def _terminal(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        command = params.get("command", "")
        if not command:
            return ToolResult.error("Missing required parameter: command")
        timeout = min(params.get("timeout", 30), 120)  # cap at 120s
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=WORKSPACE,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                proc.kill()
                return ToolResult.error(f"Command timed out after {timeout}s")
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            if len(output) > 50_000:
                output = output[:50_000] + f"\n\n... [truncated, {len(output)} chars total]"
            exit_code = proc.returncode
            result = f"Exit code: {exit_code}\n{output}" if output else f"Exit code: {exit_code}"
            if exit_code != 0:
                return ToolResult.error(result)
            return ToolResult.text(result)
        except Exception as e:
            return ToolResult.error(f"Error running command: {e}")

    async def _get_current_time(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        import datetime

        now = datetime.datetime.now(tz=datetime.UTC).isoformat()
        return ToolResult.text(f"Current UTC time: {now}")

    return [
        Tool(
            name="read_file",
            description="Read the contents of a file at the given path",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path to read",
                    },
                },
                "required": ["path"],
            },
            execute=_read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "Create or overwrite a file with the given content."
                " Creates parent directories if needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
            execute=_write_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "Edit a file by replacing an exact text match."
                " old_text must match exactly once in the file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find (must match exactly once)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace old_text with",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
            execute=_edit_file,
        ),
        Tool(
            name="terminal",
            description=(
                "Execute a shell command and return stdout/stderr."
                " Timeout defaults to 30s, max 120s."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default 30, max 120)",
                    },
                },
                "required": ["command"],
            },
            execute=_terminal,
        ),
        Tool(
            name="get_current_time",
            description="Get the current date and time in UTC",
            parameters={"type": "object", "properties": {}},
            execute=_get_current_time,
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
    """Interactive TUI for isotope-core."""

    def __init__(self) -> None:
        self.model = DEFAULT_MODEL
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.tools_enabled = True
        self.debug = False
        self.tools = _make_tools()
        self.agent: Agent | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._is_streaming = False
        self._stream_task: asyncio.Task[None] | None = None
        self._steer_text: str | None = None  # set by input reader on steer
        self._prefill_text = ""  # carry partially typed input between prompts
        self._prompt_session: Any = _PromptSession() if HAS_PROMPT_TOOLKIT else None
        self._stream_prompt_app: Any = None
        self._stream_prompt_buffer: Any = None

    def _create_stream_prompt_app(self) -> tuple[Any, Any]:
        """Create the in-stream footer input application."""
        done = asyncio.Event()
        app: Any = None

        def _accept(buf: Any) -> bool:
            if app is not None and not done.is_set():
                done.set()
                app.exit(result=buf.text)
            return True

        buffer = _Buffer(
            document=_Document(
                text=self._prefill_text,
                cursor_position=len(self._prefill_text),
            ),
            multiline=False,
            accept_handler=_accept,
        )
        bindings = _KeyBindings()

        @bindings.add("c-c")
        def _abort(_event: Any) -> None:
            if self.agent is not None:
                self.agent.abort()
            if app is not None and not done.is_set():
                done.set()
                app.exit(result="/abort")

        app = _Application(
            layout=_Layout(
                _HSplit(
                    [
                        _Window(
                            content=_FormattedTextControl(
                                [("fg:#555555", "─" * 50)],
                            ),
                            height=1,
                            dont_extend_height=True,
                        ),
                        _Window(
                            content=_BufferControl(
                                buffer=buffer,
                                input_processors=[
                                    _BeforeInput([("fg:#5599ff bold", "› ")]),
                                ],
                            ),
                            height=1,
                            dont_extend_height=True,
                        ),
                    ]
                )
            ),
            key_bindings=bindings,
            erase_when_done=True,
            full_screen=False,
            mouse_support=False,
        )
        return app, buffer

    def _close_stream_prompt(self, *, preserve_buffer: bool) -> None:
        """Close the active in-stream prompt application."""
        if preserve_buffer and self._stream_prompt_buffer is not None:
            self._prefill_text = self._stream_prompt_buffer.text
        if self._stream_prompt_app is not None:
            with contextlib.suppress(Exception):
                self._stream_prompt_app.exit(result=None)

    @staticmethod
    def _print_command_group(title: str, commands: tuple[str, ...]) -> None:
        """Print a formatted command list."""
        _print(title, style="info")
        for command in commands:
            _print(f"  {command}", style="dim")

    def _print_help(self) -> None:
        """Print interactive help."""
        self._print_command_group("Commands (between messages):", BETWEEN_MESSAGE_COMMANDS)
        _print("\nCommands (during streaming):", style="info")
        for command in DURING_STREAMING_COMMANDS:
            _print(f"  {command}", style="dim")

    @staticmethod
    def _print_known_commands() -> None:
        """Print the short known-command summary."""
        _print(
            "Commands: /tools /model /system /clear /history /debug /help /quit",
            style="dim",
        )

    def _print_stream_notice(
        self,
        message: str,
        *,
        prompt_toolkit: bool,
        style: str,
    ) -> None:
        """Print a status line while the model is streaming."""
        if prompt_toolkit:
            print(f"  [{message}]", flush=True)
        else:
            _print(f"\n  [{message}]", style=style)

    def _handle_stream_input_line(self, line: str, *, prompt_toolkit: bool) -> bool:
        """Handle one line of user input while streaming.

        Returns True when the caller should stop reading more input.
        """
        line = line.strip()
        if not line:
            return False

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/follow" and arg and self.agent:
                self.agent.follow_up(arg)
                self._print_stream_notice(
                    f"follow-up queued: {arg}",
                    prompt_toolkit=prompt_toolkit,
                    style="tool",
                )
            elif cmd == "/abort" and self.agent:
                self.agent.abort()
                self._print_stream_notice(
                    "aborting...",
                    prompt_toolkit=prompt_toolkit,
                    style="warn",
                )
                return True
            elif cmd in ("/follow", "/steer") and not arg:
                self._print_stream_notice(
                    f"usage: {cmd} <message>",
                    prompt_toolkit=prompt_toolkit,
                    style="warn",
                )
            return False

        if self.agent:
            self._steer_text = line
            self._cancel_stream()
            return True
        return False

    async def _consume_stream_events(
        self,
        gen: AsyncGenerator[AgentEvent, None],
        *,
        prompt_toolkit: bool,
        buf: _StreamBuffer | None,
    ) -> None:
        """Consume agent events for a single streamed response."""
        if prompt_toolkit:
            await asyncio.sleep(0)
        try:
            async for event in gen:
                if self.debug:
                    if buf:
                        buf.flush()
                        print(f"  [{event.type}]")
                    else:
                        _print(f"  [{event.type}]", style="dim")

                if event.type == "message_update":
                    delta = getattr(event, "delta", None)
                    if delta:
                        if buf:
                            buf.write(delta)
                        else:
                            _print_inline(delta, style="model")

                elif event.type == "tool_start":
                    tool_name = getattr(event, "tool_name", "?")
                    if buf:
                        buf.flush()
                        print(f"  [calling {tool_name}]")
                    else:
                        _print(f"\n  [calling {tool_name}]", style="tool")

                elif event.type == "tool_end":
                    is_error = getattr(event, "is_error", False)
                    if is_error:
                        if buf:
                            buf.flush()
                            print("  [tool error]")
                        else:
                            _print("  [tool error]", style="err")

                elif event.type == "turn_end":
                    msg = getattr(event, "message", None)
                    if isinstance(msg, AssistantMessage):
                        self.total_input_tokens += msg.usage.input_tokens
                        self.total_output_tokens += msg.usage.output_tokens

                elif event.type == "steer":
                    if self.debug:
                        turn = getattr(event, "turn_number", "?")
                        if buf:
                            buf.flush()
                            print(f"  [steer applied, turn {turn}]")
                        else:
                            _print(f"\n  [steer applied, turn {turn}]", style="tool")

                elif event.type == "follow_up":
                    if self.debug:
                        turn = getattr(event, "turn_number", "?")
                        if buf:
                            buf.flush()
                            print(f"  [follow-up applied, turn {turn}]")
                        else:
                            _print(f"\n  [follow-up applied, turn {turn}]", style="tool")

                elif event.type == "agent_end":
                    reason = getattr(event, "reason", "completed")
                    if reason != "completed" and self.debug:
                        if buf:
                            buf.flush()
                            print(f"  [ended: {reason}]")
                        else:
                            _print(f"\n  [ended: {reason}]", style="dim")

        except asyncio.CancelledError:
            # On cancellation (steering), discard the buffer.
            # The partial response is saved to history via partial_msg
            # so the LLM has context. Don't flush here because the
            # prompt_toolkit Application may already be torn down.
            if buf:
                buf.discard()
        except Exception as exc:
            if buf:
                buf.flush()
                print(f"Error: {exc}")
            else:
                _print(f"\nError: {exc}", style="err")

    async def _finish_stream_iteration(
        self,
        *,
        gen: AsyncGenerator[AgentEvent, None],
        buf: _StreamBuffer | None,
        done: set[asyncio.Task[None]],
        pending: set[asyncio.Task[None]],
        input_task: asyncio.Task[None],
    ) -> tuple[str, str | None, AssistantMessage | None]:
        """Finalize one streamed response iteration."""
        if input_task in pending:
            self._close_stream_prompt(preserve_buffer=True)

        for task in pending:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        trailing_text = buf.drain() if buf else ""

        partial_msg = self.agent.state.stream_message if self.agent is not None else None

        # Explicitly close the generator so _run_loop's finally block runs
        # synchronously, resetting agent.state.is_streaming to False.
        await gen.aclose()

        stream_task = self._stream_task
        self._is_streaming = False
        self._stream_task = None

        steer_text = self._steer_text
        stream_completed_naturally = (
            stream_task is not None
            and stream_task in done
            and not stream_task.cancelled()
            and stream_task.exception() is None
        )
        if steer_text and stream_completed_naturally:
            self._prefill_text = steer_text
            steer_text = None

        assistant_partial = partial_msg if isinstance(partial_msg, AssistantMessage) else None
        return trailing_text, steer_text, assistant_partial

    def _apply_steering_redirect(
        self,
        steer_text: str,
        partial_msg: AssistantMessage | None,
    ) -> str:
        """Apply a steering redirect after the current stream is interrupted."""
        print(f"  [→ {steer_text}]")

        # Drain the steering queue — we handle steering at the TUI level
        # by calling prompt() directly, so stale queue entries would
        # cause a duplicate redirect on the next agent_loop run.
        while not self.agent._steering_queue.empty():
            try:
                self.agent._steering_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Preserve the partial assistant response so the LLM knows what
        # it had already said before the user interrupted.
        if partial_msg is not None:
            self.agent.append_message(partial_msg)

        return steer_text

    def _cancel_stream(self) -> None:
        """Cancel the active stream task immediately (Claude Code style).

        Sends asyncio.CancelledError into the provider's stream generator,
        bypassing signal-based abort which has inherent polling latency.
        """
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()

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

        if cmd == "/help":
            self._print_help()
            return False

        _print(f"Unknown command: {cmd}", style="warn")
        self._print_known_commands()
        return False

    async def _read_input_during_stream(self) -> None:
        """Read input concurrently during streaming.

        Uses prompt_toolkit when available for a visible input prompt at the
        bottom of the terminal (like Claude Code). Falls back to readline.

        Any text input (no '/' prefix) is treated as steering.
        Only /follow and /abort are explicit commands.
        """
        if HAS_PROMPT_TOOLKIT and self._prompt_session is not None:
            return await self._read_input_prompt_toolkit()
        return await self._read_input_readline()

    async def _read_input_prompt_toolkit(self) -> None:
        """Read input using prompt_toolkit (visible prompt at bottom)."""
        app, buffer = self._create_stream_prompt_app()
        self._stream_prompt_app = app
        self._stream_prompt_buffer = buffer
        try:
            while self._is_streaming:
                line = await app.run_async()
                if line is None:
                    return
                if self._handle_stream_input_line(line, prompt_toolkit=True):
                    return
        except asyncio.CancelledError:
            if self._stream_prompt_buffer is not None:
                self._prefill_text = self._stream_prompt_buffer.text
            self._close_stream_prompt(preserve_buffer=False)
            raise
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            self._stream_prompt_app = None
            self._stream_prompt_buffer = None

    async def _read_input_readline(self) -> None:
        """Read input using readline (fallback when prompt_toolkit unavailable)."""
        loop = asyncio.get_event_loop()
        while self._is_streaming:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, OSError):
                break

            if not self._is_streaming:
                break

            if self._handle_stream_input_line(line, prompt_toolkit=False):
                return

    async def _send_message(self, text: str) -> None:
        """Send a user message and stream the response with concurrent input.

        Claude Code style steering: any text typed during streaming cancels the
        current response immediately and starts a new turn with that text.
        The partial assistant response is preserved in history so the LLM has
        context of what it already said.
        """
        if self.agent is None:
            self._rebuild_agent()
        assert self.agent is not None

        current_text = text

        # patch_stdout wraps the entire streaming loop so that print() output
        # renders above prompt_toolkit's input prompt (Claude Code style).
        ctx = _patch_stdout() if HAS_PROMPT_TOOLKIT else contextlib.nullcontext()
        with ctx:
            while True:
                self._is_streaming = True
                self._steer_text = None
                trailing_text = ""

                # Hold a reference to the generator for explicit lifecycle control.
                # Just creating the generator doesn't execute code — it starts
                # running only when _consume iterates it.
                gen = self.agent.prompt(current_text)  # type: ignore[arg-type]

                # When prompt_toolkit is active, use print() for output so that
                # patch_stdout can route it above the input prompt (Rich Console
                # bypasses patch_stdout because it holds the original sys.stdout).
                _pt = HAS_PROMPT_TOOLKIT
                buf = _StreamBuffer() if _pt else None

                # Create input task FIRST so prompt_toolkit's Application starts
                # before streaming output arrives (patch_stdout needs the Application
                # running to route output above the prompt).
                input_task = asyncio.create_task(self._read_input_during_stream())
                self._stream_task = asyncio.create_task(
                    self._consume_stream_events(gen, prompt_toolkit=_pt, buf=buf)
                )

                # Wait for whichever finishes first.
                done, pending = await asyncio.wait(
                    {self._stream_task, input_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                trailing_text, steer_text, partial_msg = await self._finish_stream_iteration(
                    gen=gen,
                    buf=buf,
                    done=done,
                    pending=pending,
                    input_task=input_task,
                )

                if steer_text:
                    current_text = self._apply_steering_redirect(steer_text, partial_msg)
                    continue

                if trailing_text:
                    print(trailing_text)

                # Normal completion — exit loop
                break

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
        _print("isotope-core TUI v0.1", style="info")
        _print(f"Proxy: {PROXY_BASE_URL}", style="dim")
        _print(f"Workspace: {WORKSPACE}", style="dim")

        # Fetch and select model
        models = await _fetch_models(PROXY_BASE_URL)
        self.model = await self._select_model(models)
        _print(f"Model: {self.model}", style="model")

        # System prompt
        custom_prompt = await self._get_system_prompt()
        if custom_prompt:
            self.system_prompt = custom_prompt
            _print(f"System prompt: {self.system_prompt}", style="dim")
        else:
            _print("Using default system prompt (isotope agent)", style="dim")

        # Create agent
        self.agent = self._create_agent()

        _print("\nType your message (or /help for commands). Ctrl+C to quit.\n", style="dim")

        loop = asyncio.get_event_loop()
        while True:
            try:
                if HAS_PROMPT_TOOLKIT and self._prompt_session is not None:
                    _print("─" * 50, style="white")
                    line = await self._prompt_session.prompt_async(
                        _HTML("<style fg='#5599ff'><b>› </b></style>"),
                        default=self._prefill_text,
                    )
                    self._prefill_text = ""
                else:
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
