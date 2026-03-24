"""Tests for budget limits (M4 deliverable 4)."""

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
    Usage,
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


class TestMaxTurnsEnforcement:
    """Test max_turns budget enforcement."""

    @pytest.mark.asyncio
    async def test_max_turns_stops_loop(self) -> None:
        """Loop stops when max_turns is reached."""

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

        # Create responses that would loop forever with tool calls
        responses = [
            AssistantMessage(
                content=[ToolCallContent(id=f"call_{i}", name="my_tool", arguments={})],
                stop_reason=StopReason.TOOL_USE,
                timestamp=int(time.time() * 1000),
            )
            for i in range(10)
        ]

        provider = MockProvider(responses)
        config = AgentLoopConfig(
            provider=provider,
            tools=[tool],
            max_turns=3,
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "max_turns"  # type: ignore[union-attr]

        # Should have stopped after max_turns LLM calls
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_max_turns_one(self) -> None:
        """max_turns=1 allows exactly one LLM call."""
        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        config = AgentLoopConfig(
            provider=provider,
            max_turns=1,
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "completed"  # type: ignore[union-attr]
        assert provider.call_count == 1


class TestMaxTokensEnforcement:
    """Test max_total_tokens budget enforcement."""

    @pytest.mark.asyncio
    async def test_max_tokens_stops_loop(self) -> None:
        """Loop stops when cumulative token usage exceeds max_total_tokens."""

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

        # Each response uses 100 tokens
        responses = [
            AssistantMessage(
                content=[ToolCallContent(id=f"call_{i}", name="my_tool", arguments={})],
                stop_reason=StopReason.TOOL_USE,
                usage=Usage(input_tokens=50, output_tokens=50),
                timestamp=int(time.time() * 1000),
            )
            for i in range(10)
        ]

        provider = MockProvider(responses)
        config = AgentLoopConfig(
            provider=provider,
            tools=[tool],
            max_total_tokens=250,  # Should stop after 3 turns (3 * 100 = 300 >= 250)
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "max_budget"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_max_tokens_under_limit_completes(self) -> None:
        """Loop completes normally when under the token budget."""
        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=10, output_tokens=10),
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        config = AgentLoopConfig(
            provider=provider,
            max_total_tokens=1000,
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "completed"  # type: ignore[union-attr]


class TestBudgetEventsAndReasons:
    """Test budget-related event reasons."""

    @pytest.mark.asyncio
    async def test_completed_reason(self) -> None:
        """Normal completion has reason='completed'."""
        response = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "completed"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_aborted_reason(self) -> None:
        """Aborted loop has reason='aborted'."""
        response = AssistantMessage(
            content=[TextContent(text="Done")],
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
        signal.set()  # Pre-set abort

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config, signal=signal):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "aborted"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_error_reason(self) -> None:
        """Error stop has reason='error'."""
        response = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason=StopReason.ERROR,
            error_message="Something failed",
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "error"  # type: ignore[union-attr]


class TestBudgetWithSteeringAndFollowUp:
    """Test budget enforcement with steering and follow-up."""

    @pytest.mark.asyncio
    async def test_max_turns_counts_steered_turns(self) -> None:
        """Turn count includes steered turns."""
        responses = [
            AssistantMessage(
                content=[TextContent(text="First")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            ),
            AssistantMessage(
                content=[TextContent(text="Steered")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            ),
        ]

        provider = MockProvider(responses)
        steering_queue: asyncio.Queue[UserMessage] = asyncio.Queue()
        steering_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Steer")],
                timestamp=int(time.time() * 1000),
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            steering_queue=steering_queue,  # type: ignore[arg-type]
            max_turns=2,
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "completed"  # type: ignore[union-attr]
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_max_turns_stops_during_follow_up(self) -> None:
        """Max turns enforcement includes follow-up turns."""
        responses = [
            AssistantMessage(
                content=[TextContent(text="First")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            ),
            AssistantMessage(
                content=[TextContent(text="Follow-up")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            ),
        ]

        provider = MockProvider(responses)
        follow_up_queue: asyncio.Queue[UserMessage] = asyncio.Queue()
        follow_up_queue.put_nowait(
            UserMessage(
                content=[TextContent(text="Follow up")],
                timestamp=int(time.time() * 1000),
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            follow_up_queue=follow_up_queue,  # type: ignore[arg-type]
            max_turns=1,
        )

        prompt = UserMessage(
            content=[TextContent(text="Start")],
            timestamp=int(time.time() * 1000),
        )

        events: list[AgentEvent] = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        end_event = next(e for e in events if e.type == "agent_end")
        # The follow-up was consumed (emitting follow_up event), but the next
        # turn (turn 2) hits max_turns before the LLM is called.
        assert "follow_up" in event_types
        assert end_event.reason == "max_turns"  # type: ignore[union-attr]
        assert provider.call_count == 1


class TestBudgetViaAgent:
    """Test budget enforcement via the Agent class."""

    @pytest.mark.asyncio
    async def test_agent_max_turns(self) -> None:
        """Agent respects max_turns."""

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

        responses = [
            AssistantMessage(
                content=[ToolCallContent(id=f"call_{i}", name="my_tool", arguments={})],
                stop_reason=StopReason.TOOL_USE,
                timestamp=int(time.time() * 1000),
            )
            for i in range(10)
        ]

        provider = MockProvider(responses)
        agent = Agent(provider=provider, tools=[tool], max_turns=2)

        events: list[AgentEvent] = []
        async for event in agent.prompt("Start"):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "max_turns"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_agent_max_total_tokens(self) -> None:
        """Agent respects max_total_tokens."""

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

        responses = [
            AssistantMessage(
                content=[ToolCallContent(id=f"call_{i}", name="my_tool", arguments={})],
                stop_reason=StopReason.TOOL_USE,
                usage=Usage(input_tokens=100, output_tokens=100),
                timestamp=int(time.time() * 1000),
            )
            for i in range(10)
        ]

        provider = MockProvider(responses)
        agent = Agent(provider=provider, tools=[tool], max_total_tokens=500)

        events: list[AgentEvent] = []
        async for event in agent.prompt("Start"):
            events.append(event)

        end_event = next(e for e in events if e.type == "agent_end")
        assert end_event.reason == "max_budget"  # type: ignore[union-attr]
