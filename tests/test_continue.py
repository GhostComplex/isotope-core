"""Tests for continue/resume (M4 deliverable 3)."""

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest

from isotope_core.agent import Agent
from isotope_core.providers.base import StreamDoneEvent, StreamEvent, StreamStartEvent
from isotope_core.types import (
    AgentEvent,
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    ToolResultMessage,
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

        yield StreamStartEvent(partial=msg)
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Tests
# =============================================================================


class TestContinueAfterMaxTokens:
    """Test continue after max_tokens stop reason."""

    @pytest.mark.asyncio
    async def test_continue_after_max_tokens(self) -> None:
        """Continue works after a max_tokens stop to let the model finish."""
        # First response stops at max_tokens
        max_tokens_response = AssistantMessage(
            content=[TextContent(text="The answer is")],
            stop_reason=StopReason.MAX_TOKENS,
            usage=Usage(input_tokens=10, output_tokens=10),
            timestamp=int(time.time() * 1000),
        )
        # Second response completes
        completion_response = AssistantMessage(
            content=[TextContent(text=" 42.")],
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=20, output_tokens=5),
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([max_tokens_response, completion_response])
        agent = Agent(provider=provider)

        # First prompt
        events1: list[AgentEvent] = []
        async for event in agent.prompt("What is the answer?"):
            events1.append(event)

        # The agent should have the max_tokens response in history
        assert len(agent.messages) == 2  # user + assistant
        assert agent.messages[-1].role == "assistant"  # type: ignore[union-attr]

        # Continue from where we left off
        events2: list[AgentEvent] = []
        async for event in agent.continue_():
            events2.append(event)

        event_types = [e.type for e in events2]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

        # Provider should have been called twice total
        assert provider.call_count == 2

        # The second call should have the full history
        assert len(provider.contexts_seen) == 2
        # Second context should contain the original messages + max_tokens response
        assert len(provider.contexts_seen[1].messages) > len(provider.contexts_seen[0].messages)


class TestContinueAfterError:
    """Test continue after error stop reason."""

    @pytest.mark.asyncio
    async def test_continue_after_error(self) -> None:
        """Continue works after an error to retry."""
        error_response = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason=StopReason.ERROR,
            error_message="Rate limited",
            timestamp=int(time.time() * 1000),
        )
        success_response = AssistantMessage(
            content=[TextContent(text="Success")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([error_response, success_response])
        agent = Agent(provider=provider)

        # First prompt — will error
        events1: list[AgentEvent] = []
        async for event in agent.prompt("Hello"):
            events1.append(event)

        end_event1 = next(e for e in events1 if e.type == "agent_end")
        assert end_event1.reason == "error"  # type: ignore[union-attr]

        # Continue — should retry
        events2: list[AgentEvent] = []
        async for event in agent.continue_():
            events2.append(event)

        end_event2 = next(e for e in events2 if e.type == "agent_end")
        assert end_event2.reason == "completed"  # type: ignore[union-attr]


class TestContinueWithEmptyHistory:
    """Test continue with empty history (error case)."""

    @pytest.mark.asyncio
    async def test_continue_with_no_messages_raises(self) -> None:
        """Continue raises RuntimeError when there are no messages."""
        provider = MockProvider([])
        agent = Agent(provider=provider)

        with pytest.raises(RuntimeError, match="No messages to continue"):
            async for _ in agent.continue_():
                pass


class TestContinueWithToolResult:
    """Test continue after a tool result message."""

    @pytest.mark.asyncio
    async def test_continue_from_tool_result(self) -> None:
        """Continue works when last message is a tool result."""
        response = AssistantMessage(
            content=[TextContent(text="I processed the tool result")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response])
        agent = Agent(provider=provider)

        # Manually set up state as if we stopped mid-tool-execution
        agent.append_message(
            UserMessage(
                content=[TextContent(text="Use the tool")],
                timestamp=int(time.time() * 1000),
            )
        )
        agent.append_message(
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="my_tool",
                content=[TextContent(text="Tool output")],
                is_error=False,
                timestamp=int(time.time() * 1000),
            )
        )

        events: list[AgentEvent] = []
        async for event in agent.continue_():
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types


class TestContinuePreservesContext:
    """Test that continue preserves existing context."""

    @pytest.mark.asyncio
    async def test_continue_sends_full_history(self) -> None:
        """Continue sends the full message history to the provider."""
        first_response = AssistantMessage(
            content=[TextContent(text="First")],
            stop_reason=StopReason.MAX_TOKENS,
            timestamp=int(time.time() * 1000),
        )
        second_response = AssistantMessage(
            content=[TextContent(text="Second")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([first_response, second_response])
        agent = Agent(provider=provider, system_prompt="Be helpful")

        # First prompt
        async for _ in agent.prompt("Hello"):
            pass

        # Continue
        async for _ in agent.continue_():
            pass

        # Second call should have seen the full history
        assert provider.call_count == 2
        second_context = provider.contexts_seen[1]
        assert second_context.system_prompt == "Be helpful"
        # Should have user message + first assistant response
        assert len(second_context.messages) == 2
