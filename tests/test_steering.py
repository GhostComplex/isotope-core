"""Tests for steering messages (M4 deliverable 1)."""

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
        self.contexts_seen: list[Context] = []

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.contexts_seen.append(context)

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


# =============================================================================
# Tests
# =============================================================================


class TestSteerMidRunAfterToolExecution:
    """Test steering mid-run after tool execution."""

    @pytest.mark.asyncio
    async def test_steer_after_tool_call(self) -> None:
        """Steering message is processed after tool execution completes."""

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("Tool done")

        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute,
        )

        tool_response = AssistantMessage(
            content=[ToolCallContent(id="call_1", name="my_tool", arguments={})],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )
        steered_response = AssistantMessage(
            content=[TextContent(text="Steered response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_response, steered_response])
        steering_queue: asyncio.Queue[AssistantMessage | UserMessage] = asyncio.Queue()

        # Queue a steering message before we start
        steer_msg = UserMessage(
            content=[TextContent(text="Actually use Python")],
            timestamp=int(time.time() * 1000),
        )
        steering_queue.put_nowait(steer_msg)

        config = AgentLoopConfig(
            provider=provider,
            tools=[tool],
            steering_queue=steering_queue,  # type: ignore[arg-type]
        )

        prompt = UserMessage(
            content=[TextContent(text="Do something")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "steer" in event_types
        assert "agent_end" in event_types

        # The steering message should have been included in the LLM context
        # Provider was called twice: once for tool call, once after steering
        assert provider.call_count == 2


class TestMultipleSteeringMessagesQueued:
    """Test multiple steering messages queued."""

    @pytest.mark.asyncio
    async def test_multiple_steers_processed(self) -> None:
        """Multiple steering messages are all processed in one batch."""

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("Done")

        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute,
        )

        tool_response = AssistantMessage(
            content=[ToolCallContent(id="call_1", name="my_tool", arguments={})],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )
        final_response = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_response, final_response])
        steering_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        # Queue two steering messages
        steering_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Steer 1")],
                timestamp=int(time.time() * 1000),
            )
        )
        steering_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Steer 2")],
                timestamp=int(time.time() * 1000),
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            tools=[tool],
            steering_queue=steering_queue,  # type: ignore[arg-type]
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        steer_events = [e for e in events if e.type == "steer"]
        assert len(steer_events) == 2


class TestSteerWithNoPendingToolCalls:
    """Test steer when there are no pending tool calls (text-only response)."""

    @pytest.mark.asyncio
    async def test_steer_during_text_only_response(self) -> None:
        """Steering during a text-only response (no tool calls)."""
        # When there are no tool calls, steering is checked after turn_end
        # but before the follow-up check. Since no tool calls exist, steering
        # still triggers.
        first_response = AssistantMessage(
            content=[TextContent(text="First response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        second_response = AssistantMessage(
            content=[TextContent(text="Steered response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([first_response, second_response])
        steering_queue: asyncio.Queue[UserMessage] = asyncio.Queue()

        # Queue a steering message — it'll be checked after the first turn
        steering_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Change direction")],
                timestamp=int(time.time() * 1000),
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            steering_queue=steering_queue,  # type: ignore[arg-type]
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
        assert provider.call_count == 2


class TestSteerViaAgent:
    """Test steering via the Agent class."""

    @pytest.mark.asyncio
    async def test_agent_steer_with_string(self) -> None:
        """Agent.steer() works with a string argument."""
        first_response = AssistantMessage(
            content=[TextContent(text="First response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        second_response = AssistantMessage(
            content=[TextContent(text="Steered")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([first_response, second_response])
        agent = Agent(provider=provider)

        events: list[AgentEvent] = []
        # Steer before prompting — it'll be picked up after the first turn
        agent.steer("Actually, use Python instead of TypeScript")

        async for event in agent.prompt("Hello"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "steer" in event_types

    @pytest.mark.asyncio
    async def test_agent_steer_with_message(self) -> None:
        """Agent.steer() works with a Message argument."""
        first_response = AssistantMessage(
            content=[TextContent(text="First response")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        second_response = AssistantMessage(
            content=[TextContent(text="Steered")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([first_response, second_response])
        agent = Agent(provider=provider)

        msg = UserMessage(
            content=[TextContent(text="Change plans")],
            timestamp=int(time.time() * 1000),
        )
        agent.steer(msg)

        events: list[AgentEvent] = []
        async for event in agent.prompt("Hello"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "steer" in event_types
