"""Tests for lifecycle hooks on Agent.

Covers on_agent_start, on_agent_end, on_turn_start, on_turn_end, on_error.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from isotope_core.agent import Agent
from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
)
from isotope_core.tools import Tool, ToolResult
from isotope_core.types import (
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    ToolCallContent,
    Usage,
)

# =============================================================================
# Mock Provider
# =============================================================================


class MockProvider:
    """A mock provider for testing."""

    def __init__(self, responses: list[AssistantMessage] | None = None) -> None:
        self.responses = responses or []
        self.call_count = 0

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self.call_count >= len(self.responses):
            msg = AssistantMessage(
                content=[TextContent(text="Default response")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            )
        else:
            msg = self.responses[self.call_count]

        self.call_count += 1

        if signal and signal.is_set():
            error_msg = AssistantMessage(
                content=[],
                stop_reason=StopReason.ABORTED,
                error_message="Aborted",
                timestamp=int(time.time() * 1000),
            )
            yield StreamStartEvent(partial=error_msg)
            yield StreamDoneEvent(message=error_msg)
            return

        yield StreamStartEvent(partial=msg)
        yield StreamDoneEvent(message=msg)


class ErrorProvider:
    """A provider that returns an error message."""

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        msg = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason=StopReason.ERROR,
            error_message="Provider error occurred",
            timestamp=int(time.time() * 1000),
        )
        yield StreamStartEvent(partial=msg)
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Tests
# =============================================================================


class TestOnAgentStart:
    """Tests for on_agent_start hook."""

    @pytest.mark.asyncio
    async def test_on_agent_start_called(self) -> None:
        called = False

        async def hook() -> None:
            nonlocal called
            called = True

        agent = Agent(
            provider=MockProvider(
                [
                    AssistantMessage(
                        content=[TextContent(text="Hello")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    )
                ]
            ),
            on_agent_start=hook,
        )

        async for _ in agent.prompt("Hi"):
            pass

        assert called is True

    @pytest.mark.asyncio
    async def test_on_agent_start_none_is_fine(self) -> None:
        """Agent works when on_agent_start is None."""
        agent = Agent(
            provider=MockProvider(
                [
                    AssistantMessage(
                        content=[TextContent(text="Hello")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    )
                ]
            ),
        )

        events = []
        async for event in agent.prompt("Hi"):
            events.append(event)

        assert len(events) > 0


class TestOnAgentEnd:
    """Tests for on_agent_end hook."""

    @pytest.mark.asyncio
    async def test_on_agent_end_called_with_reason(self) -> None:
        reasons: list[str] = []

        async def hook(reason: str) -> None:
            reasons.append(reason)

        agent = Agent(
            provider=MockProvider(
                [
                    AssistantMessage(
                        content=[TextContent(text="Hello")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    )
                ]
            ),
            on_agent_end=hook,
        )

        async for _ in agent.prompt("Hi"):
            pass

        assert reasons == ["completed"]

    @pytest.mark.asyncio
    async def test_on_agent_end_error_reason(self) -> None:
        reasons: list[str] = []

        async def hook(reason: str) -> None:
            reasons.append(reason)

        agent = Agent(
            provider=ErrorProvider(),
            on_agent_end=hook,
        )

        async for _ in agent.prompt("Hi"):
            pass

        assert "error" in reasons


class TestOnTurnStart:
    """Tests for on_turn_start hook."""

    @pytest.mark.asyncio
    async def test_on_turn_start_called_with_turn_number(self) -> None:
        turns: list[int] = []

        async def hook(turn_number: int) -> None:
            turns.append(turn_number)

        # Create a tool that triggers a second turn
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("Done")

        test_tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute,
        )

        tool_response = AssistantMessage(
            content=[ToolCallContent(id="call_1", name="test_tool", arguments={})],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )
        final_response = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        agent = Agent(
            provider=MockProvider([tool_response, final_response]),
            tools=[test_tool],
            on_turn_start=hook,
        )

        async for _ in agent.prompt("Hi"):
            pass

        # Should have turn 1 and turn 2
        assert 1 in turns
        assert 2 in turns


class TestOnTurnEnd:
    """Tests for on_turn_end hook."""

    @pytest.mark.asyncio
    async def test_on_turn_end_called_with_correct_args(self) -> None:
        turn_data: list[tuple[int, AssistantMessage]] = []

        async def hook(turn_number: int, message: AssistantMessage) -> None:
            turn_data.append((turn_number, message))

        response = AssistantMessage(
            content=[TextContent(text="Hello!")],
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=10, output_tokens=20),
            timestamp=int(time.time() * 1000),
        )

        agent = Agent(
            provider=MockProvider([response]),
            on_turn_end=hook,
        )

        async for _ in agent.prompt("Hi"):
            pass

        assert len(turn_data) == 1
        assert turn_data[0][0] == 1  # turn_number
        assert isinstance(turn_data[0][1], AssistantMessage)
        assert turn_data[0][1].content[0].text == "Hello!"  # type: ignore[union-attr]


class TestOnError:
    """Tests for on_error hook."""

    @pytest.mark.asyncio
    async def test_on_error_called_on_provider_error(self) -> None:
        errors: list[Exception] = []

        async def hook(exc: Exception) -> None:
            errors.append(exc)

        agent = Agent(
            provider=ErrorProvider(),
            on_error=hook,
        )

        async for _ in agent.prompt("Hi"):
            pass

        assert len(errors) == 1
        assert "Provider error occurred" in str(errors[0])


class TestHooksOptional:
    """Tests that all hooks are optional (no errors when None)."""

    @pytest.mark.asyncio
    async def test_all_hooks_none(self) -> None:
        """Agent works with all hooks set to None."""
        agent = Agent(
            provider=MockProvider(
                [
                    AssistantMessage(
                        content=[TextContent(text="Hello")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    )
                ]
            ),
            on_agent_start=None,
            on_agent_end=None,
            on_turn_start=None,
            on_turn_end=None,
            on_error=None,
        )

        events = []
        async for event in agent.prompt("Hi"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types


class TestHookErrorHandling:
    """Tests that hook errors don't break the loop."""

    @pytest.mark.asyncio
    async def test_hook_error_suppressed(self) -> None:
        """A hook that raises should not crash the agent."""

        async def bad_hook() -> None:
            raise RuntimeError("Hook exploded")

        agent = Agent(
            provider=MockProvider(
                [
                    AssistantMessage(
                        content=[TextContent(text="Hello")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    )
                ]
            ),
            on_agent_start=bad_hook,
        )

        events = []
        async for event in agent.prompt("Hi"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_turn_end_hook_error_suppressed(self) -> None:
        """A turn_end hook error should not crash the agent."""

        async def bad_hook(turn: int, msg: AssistantMessage) -> None:
            raise RuntimeError("turn_end hook exploded")

        agent = Agent(
            provider=MockProvider(
                [
                    AssistantMessage(
                        content=[TextContent(text="Hello")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    )
                ]
            ),
            on_turn_end=bad_hook,
        )

        events = []
        async for event in agent.prompt("Hi"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_end" in event_types
