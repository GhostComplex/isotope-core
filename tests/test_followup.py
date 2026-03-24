"""Tests for follow-up queue (M4 deliverable 2)."""

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

        yield StreamStartEvent(partial=msg)
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Tests
# =============================================================================


class TestFollowUpTriggersNewTurn:
    """Test that follow-up triggers a new turn after end_turn."""

    @pytest.mark.asyncio
    async def test_follow_up_after_end_turn(self) -> None:
        """Follow-up message triggers a new turn after agent would normally stop."""
        first_response = AssistantMessage(
            content=[TextContent(text="First response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        followup_response = AssistantMessage(
            content=[TextContent(text="Follow-up response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([first_response, followup_response])
        follow_up_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        follow_up_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Now summarize what you did")],
                timestamp=int(time.time() * 1000),
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            follow_up_queue=follow_up_queue,  # type: ignore[arg-type]
        )

        prompt = UserMessage(
            content=[TextContent(text="Do something")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "follow_up" in event_types
        assert "agent_end" in event_types
        assert provider.call_count == 2


class TestMultipleFollowUpsInSequence:
    """Test multiple follow-ups in sequence."""

    @pytest.mark.asyncio
    async def test_multiple_follow_ups(self) -> None:
        """Multiple follow-up messages are processed one after another."""
        responses = [
            AssistantMessage(
                content=[TextContent(text=f"Response {i}")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            )
            for i in range(3)
        ]

        provider = MockProvider(responses)
        follow_up_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        follow_up_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Follow-up 1")],
                timestamp=int(time.time() * 1000),
            )
        )
        follow_up_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Follow-up 2")],
                timestamp=int(time.time() * 1000),
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            follow_up_queue=follow_up_queue,  # type: ignore[arg-type]
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        followup_events = [e for e in events if e.type == "follow_up"]
        # Both follow-ups are consumed in one batch after the first end_turn
        assert len(followup_events) == 2
        # Provider called: initial + after follow-ups (both consumed, one new turn)
        assert provider.call_count == 2


class TestFollowUpWithEmptyQueue:
    """Test follow-up with empty queue (normal end)."""

    @pytest.mark.asyncio
    async def test_empty_follow_up_queue_ends_normally(self) -> None:
        """Empty follow-up queue results in normal agent end."""
        response = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        follow_up_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        config = AgentLoopConfig(
            provider=provider,
            follow_up_queue=follow_up_queue,  # type: ignore[arg-type]
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "follow_up" not in event_types
        assert "agent_end" in event_types

        # Check reason is completed
        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "completed"  # type: ignore[union-attr]


class TestFollowUpCombinedWithSteering:
    """Test follow-up combined with steering."""

    @pytest.mark.asyncio
    async def test_steering_takes_precedence_over_followup(self) -> None:
        """Steering is checked first; follow-up triggers after steering is consumed."""

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("Done")

        tool = Tool(
            name="my_tool",
            description="Test",
            parameters={"type": "object"},
            execute=execute,
        )

        tool_response = AssistantMessage(
            content=[ToolCallContent(id="call_1", name="my_tool", arguments={})],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )
        steered_response = AssistantMessage(
            content=[TextContent(text="Steered")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        followup_response = AssistantMessage(
            content=[TextContent(text="Follow-up")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_response, steered_response, followup_response])
        steering_queue: asyncio.Queue[UserMessage] = asyncio.Queue()
        follow_up_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        steering_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Steer this")],
                timestamp=int(time.time() * 1000),
            )
        )
        follow_up_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Follow up on that")],
                timestamp=int(time.time() * 1000),
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            tools=[tool],
            steering_queue=steering_queue,  # type: ignore[arg-type]
            follow_up_queue=follow_up_queue,  # type: ignore[arg-type]
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "steer" in event_types
        assert "follow_up" in event_types

        # Steer comes before follow_up in event order
        steer_idx = event_types.index("steer")
        followup_idx = event_types.index("follow_up")
        assert steer_idx < followup_idx


class TestFollowUpViaAgent:
    """Test follow-up via the Agent class."""

    @pytest.mark.asyncio
    async def test_agent_follow_up_with_string(self) -> None:
        """Agent.follow_up() works with a string argument."""
        first_response = AssistantMessage(
            content=[TextContent(text="First response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        second_response = AssistantMessage(
            content=[TextContent(text="Follow-up response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([first_response, second_response])
        agent = Agent(provider=provider)

        agent.follow_up("Now summarize what you did")

        events: list[AgentEvent] = []
        async for event in agent.prompt("Hello"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "follow_up" in event_types

    @pytest.mark.asyncio
    async def test_agent_follow_up_with_message(self) -> None:
        """Agent.follow_up() works with a Message argument."""
        first_response = AssistantMessage(
            content=[TextContent(text="First")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        second_response = AssistantMessage(
            content=[TextContent(text="Second")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([first_response, second_response])
        agent = Agent(provider=provider)

        msg = UserMessage(
            content=[TextContent(text="Summarize")],
            timestamp=int(time.time() * 1000),
        )
        agent.follow_up(msg)

        events: list[AgentEvent] = []
        async for event in agent.prompt("Hello"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "follow_up" in event_types
