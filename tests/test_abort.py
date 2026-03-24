"""Tests for abort improvements (M4 deliverable 6)."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from isotope_core.agent import Agent
from isotope_core.loop import AgentLoopConfig, agent_loop
from isotope_core.providers.base import StreamDoneEvent, StreamEvent, StreamStartEvent
from isotope_core.tools import Tool, ToolResult
from isotope_core.types import (
    AgentEvent,
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    ToolCallContent,
    UserMessage,
)

# =============================================================================
# Mock Provider
# =============================================================================


class MockProvider:
    """A mock provider for testing."""

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
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
                content=[TextContent(text="Done")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            )
        else:
            msg = self.responses[self.call_count]

        self.call_count += 1

        if signal and signal.is_set():
            error_msg = AssistantMessage(
                content=[TextContent(text="")],
                stop_reason=StopReason.ABORTED,
                error_message="Aborted",
                timestamp=int(time.time() * 1000),
            )
            yield StreamStartEvent(partial=error_msg)
            yield StreamDoneEvent(message=error_msg)
            return

        yield StreamStartEvent(partial=msg)
        yield StreamDoneEvent(message=msg)


class SlowMockProvider(MockProvider):
    """A mock provider that delays, allowing abort to be tested."""

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        # Small delay to allow abort signal to be set
        await asyncio.sleep(0.01)

        if signal and signal.is_set():
            error_msg = AssistantMessage(
                content=[TextContent(text="")],
                stop_reason=StopReason.ABORTED,
                error_message="Aborted",
                timestamp=int(time.time() * 1000),
            )
            yield StreamStartEvent(partial=error_msg)
            yield StreamDoneEvent(message=error_msg)
            return

        async for event in super().stream(
            context,
            temperature=temperature,
            max_tokens=max_tokens,
            signal=signal,
        ):
            yield event


# =============================================================================
# Tests
# =============================================================================


class TestAbortDuringToolExecution:
    """Test abort during tool execution."""

    @pytest.mark.asyncio
    async def test_abort_during_tool_sets_signal(self) -> None:
        """Abort during tool execution produces proper events."""

        async def slow_execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            # Simulate slow tool
            await asyncio.sleep(0.01)
            if signal and signal.is_set():
                return ToolResult(content=[TextContent(text="Aborted")], is_error=True)
            return ToolResult.text("Done")

        tool = Tool(
            name="slow_tool",
            description="A slow tool",
            parameters={"type": "object"},
            execute=slow_execute,
        )

        tool_response = AssistantMessage(
            content=[ToolCallContent(id="call_1", name="slow_tool", arguments={})],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_response])
        agent = Agent(provider=provider, tools=[tool])

        events: list[AgentEvent] = []
        async for event in agent.prompt("Use the tool"):
            events.append(event)
            if event.type == "tool_start":
                agent.abort()

        event_types = [e.type for e in events]
        assert "agent_end" in event_types


class TestAbortDuringLLMStreaming:
    """Test abort during LLM streaming."""

    @pytest.mark.asyncio
    async def test_abort_signal_stops_streaming(self) -> None:
        """Abort signal set before loop starts results in immediate stop."""
        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )

        signal = asyncio.Event()
        signal.set()

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config, signal=signal):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "aborted"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_abort_during_streaming(self) -> None:
        """Agent.abort() during streaming produces agent_end event."""
        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = SlowMockProvider([response])
        agent = Agent(provider=provider)

        events: list[AgentEvent] = []
        async for event in agent.prompt("Hello"):
            events.append(event)
            if event.type == "message_start":
                agent.abort()

        event_types = [e.type for e in events]
        assert "agent_end" in event_types


class TestMultipleAbortCalls:
    """Test that multiple abort calls are safe."""

    @pytest.mark.asyncio
    async def test_abort_idempotent(self) -> None:
        """Calling abort() multiple times is safe."""
        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        agent = Agent(provider=provider)

        # Abort without streaming — should not raise
        agent.abort()
        agent.abort()
        agent.abort()

    @pytest.mark.asyncio
    async def test_abort_multiple_during_streaming(self) -> None:
        """Multiple abort calls during streaming don't cause errors."""
        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = SlowMockProvider([response])
        agent = Agent(provider=provider)

        events: list[AgentEvent] = []
        async for event in agent.prompt("Hello"):
            events.append(event)
            agent.abort()
            agent.abort()  # Second call should be safe

        event_types = [e.type for e in events]
        assert "agent_end" in event_types


class TestEventSequenceOnAbort:
    """Test event sequence on abort."""

    @pytest.mark.asyncio
    async def test_abort_event_sequence(self) -> None:
        """Abort produces proper event sequence: agent_start, ..., turn_end, agent_end."""
        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )

        signal = asyncio.Event()
        signal.set()

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config, signal=signal):
            events.append(event)

        event_types = [e.type for e in events]

        # Should have: agent_start, turn_start, message_start/end (for prompt),
        # message_start/end (for abort), turn_end, agent_end
        assert event_types[0] == "agent_start"
        assert event_types[1] == "turn_start"
        assert "turn_end" in event_types
        assert event_types[-1] == "agent_end"

        # Turn end and agent end should be at the end
        turn_end_idx = event_types.index("turn_end")
        agent_end_idx = event_types.index("agent_end")
        assert agent_end_idx > turn_end_idx

    @pytest.mark.asyncio
    async def test_abort_clears_queues(self) -> None:
        """Abort clears steering and follow-up queues."""
        provider = MockProvider([])
        agent = Agent(provider=provider)

        # Add messages to queues
        agent.steer("Steer this")
        agent.follow_up("Follow up")

        # Abort should clear them
        agent.abort()

        assert agent._steering_queue.empty()
        assert agent._follow_up_queue.empty()
