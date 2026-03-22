"""Regression tests for the TUI streaming prompt behavior."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from tui import main as tui_main


class _FakeAgent:
    def __init__(self) -> None:
        self.messages: list[object] = []
        self.state = SimpleNamespace(stream_message=None)
        self._steering_queue: asyncio.Queue[object] = asyncio.Queue()

    async def _events(self):
        yield SimpleNamespace(type="message_update", delta="Hello")
        yield SimpleNamespace(type="turn_end", message=None)

    def prompt(self, text: str):
        return self._events()

    def append_message(self, message: object) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_send_message_flushes_trailing_text_after_input_cancel(monkeypatch) -> None:
    tui = tui_main.TUI()
    tui.agent = _FakeAgent()  # type: ignore[assignment]

    monkeypatch.setattr(tui_main, "HAS_PROMPT_TOOLKIT", True)
    monkeypatch.setattr(tui_main, "_patch_stdout", contextlib.nullcontext)

    input_cancelled = asyncio.Event()

    async def fake_read_input_during_stream() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            input_cancelled.set()
            raise

    monkeypatch.setattr(tui, "_read_input_during_stream", fake_read_input_during_stream)

    printed: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        text = "" if not args else str(args[0])
        if text == "Hello":
            assert input_cancelled.is_set()
        printed.append(text)

    monkeypatch.setattr("builtins.print", fake_print)

    await tui._send_message("hi")

    assert "Hello" in printed


@pytest.mark.asyncio
async def test_send_message_late_input_prefills_after_natural_completion(monkeypatch) -> None:
    tui = tui_main.TUI()

    class _FastAgent:
        def __init__(self) -> None:
            self.messages: list[object] = []
            self.state = SimpleNamespace(stream_message=None)
            self._steering_queue: asyncio.Queue[object] = asyncio.Queue()
            self.prompt_calls: list[str] = []

        async def _events(self):
            yield SimpleNamespace(type="message_update", delta="done")
            yield SimpleNamespace(type="turn_end", message=None)

        def prompt(self, text: str):
            self.prompt_calls.append(text)
            return self._events()

        def append_message(self, message: object) -> None:
            self.messages.append(message)

    agent = _FastAgent()
    tui.agent = agent  # type: ignore[assignment]

    monkeypatch.setattr(tui_main, "HAS_PROMPT_TOOLKIT", True)
    monkeypatch.setattr(tui_main, "_patch_stdout", contextlib.nullcontext)
    tui._stream_prompt_buffer = SimpleNamespace(text="can you help me too")

    async def fake_read_input_during_stream() -> None:
        await asyncio.Future()

    monkeypatch.setattr(tui, "_read_input_during_stream", fake_read_input_during_stream)

    printed: list[str] = []

    def fake_print(*args, **_kwargs) -> None:
        printed.append("" if not args else str(args[0]))

    monkeypatch.setattr("builtins.print", fake_print)

    await tui._send_message("hi")

    assert agent.prompt_calls == ["hi"]
    assert tui._prefill_text == "can you help me too"
    assert all(not line.startswith("  [→ ") for line in printed)


@pytest.mark.asyncio
async def test_prompt_toolkit_cancel_resets_buffer(monkeypatch) -> None:
    tui = tui_main.TUI()

    class _App:
        async def run_async(self):
            raise asyncio.CancelledError

        def exit(self, **_kwargs) -> None:
            pass

    tui._prefill_text = ""
    tui._is_streaming = True
    buffer = SimpleNamespace(text="are you ok for")

    def fake_create_stream_prompt_app():
        return _App(), buffer

    monkeypatch.setattr(tui, "_create_stream_prompt_app", fake_create_stream_prompt_app)

    with pytest.raises(asyncio.CancelledError):
        await tui._read_input_prompt_toolkit()

    assert tui._prefill_text == "are you ok for"
    assert tui._stream_prompt_app is None
    assert tui._stream_prompt_buffer is None


@pytest.mark.asyncio
async def test_prompt_toolkit_cancel_stores_empty_prefill_when_no_text(monkeypatch) -> None:
    tui = tui_main.TUI()

    class _App:
        async def run_async(self):
            raise asyncio.CancelledError

        def exit(self, **_kwargs) -> None:
            pass

    def fake_create_stream_prompt_app():
        return _App(), SimpleNamespace(text="")

    monkeypatch.setattr(tui, "_create_stream_prompt_app", fake_create_stream_prompt_app)
    tui._is_streaming = True
    tui._prefill_text = "stale"

    with pytest.raises(asyncio.CancelledError):
        await tui._read_input_prompt_toolkit()

    assert tui._prefill_text == ""


def test_prompt_toolkit_uses_application_erase_when_done() -> None:
    tui = tui_main.TUI()
    tui._prefill_text = "and 5 more"

    app, buffer = tui._create_stream_prompt_app()

    assert app.erase_when_done is True
    assert buffer.text == "and 5 more"


@pytest.mark.asyncio
async def test_prompt_toolkit_programmatic_close_does_not_steer(monkeypatch) -> None:
    tui = tui_main.TUI()

    class _App:
        async def run_async(self):
            return None

    def fake_create_stream_prompt_app():
        return _App(), SimpleNamespace(text="half typed")

    monkeypatch.setattr(tui, "_create_stream_prompt_app", fake_create_stream_prompt_app)
    tui._is_streaming = True

    await tui._read_input_prompt_toolkit()

    assert tui._steer_text is None
    assert tui._stream_prompt_app is None
    assert tui._stream_prompt_buffer is None
