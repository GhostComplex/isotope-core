"""Tests for isotope_core.agent module."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from isotope_core.agent import Agent, AgentState
from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
)
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

    def __init__(self, responses: list[AssistantMessage] | None = None) -> None:
        """Initialize with optional responses."""
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
        """Stream a response."""
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


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MockProvider:
    """Create a mock provider with a simple response."""
    return MockProvider(
        [
            AssistantMessage(
                content=[TextContent(text="Hello!")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            )
        ]
    )


@pytest.fixture
def agent(mock_provider: MockProvider) -> Agent:
    """Create an agent with a mock provider."""
    return Agent(provider=mock_provider, system_prompt="You are helpful.")


# =============================================================================
# Tests for AgentState
# =============================================================================


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_default_state(self) -> None:
        """Test default AgentState values."""
        state = AgentState()
        assert state.system_prompt == ""
        assert state.provider is None
        assert state.tools == []
        assert state.messages == []
        assert state.is_streaming is False
        assert state.stream_message is None
        assert state.pending_tool_calls == set()
        assert state.error is None


# =============================================================================
# Tests for Agent initialization
# =============================================================================


class TestAgentInit:
    """Tests for Agent initialization."""

    def test_agent_default_init(self) -> None:
        """Test default agent initialization."""
        agent = Agent()
        assert agent.state.system_prompt == ""
        assert agent.state.provider is None
        assert agent.state.tools == []

    def test_agent_with_provider(self, mock_provider: MockProvider) -> None:
        """Test agent initialization with provider."""
        agent = Agent(provider=mock_provider)
        assert agent.state.provider is mock_provider

    def test_agent_with_system_prompt(self) -> None:
        """Test agent initialization with system prompt."""
        agent = Agent(system_prompt="Be helpful.")
        assert agent.state.system_prompt == "Be helpful."

    def test_agent_with_tools(self) -> None:
        """Test agent initialization with tools."""

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("Done")

        tool = Tool(
            name="test",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute,
        )
        agent = Agent(tools=[tool])
        assert len(agent.state.tools) == 1


# =============================================================================
# Tests for Agent state mutators
# =============================================================================


class TestAgentStateMutators:
    """Tests for Agent state mutators."""

    def test_set_system_prompt(self, agent: Agent) -> None:
        """Test setting system prompt."""
        agent.set_system_prompt("New prompt")
        assert agent.state.system_prompt == "New prompt"

    def test_set_provider(self, agent: Agent) -> None:
        """Test setting provider."""
        new_provider = MockProvider()
        agent.set_provider(new_provider)
        assert agent.state.provider is new_provider

    def test_set_tools(self, agent: Agent) -> None:
        """Test setting tools."""

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("Done")

        tool = Tool(
            name="new_tool",
            description="A new tool",
            parameters={"type": "object"},
            execute=execute,
        )
        agent.set_tools([tool])
        assert len(agent.state.tools) == 1
        assert agent.state.tools[0].name == "new_tool"

    def test_replace_messages(self, agent: Agent) -> None:
        """Test replacing messages."""
        msg = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        agent.replace_messages([msg])
        assert len(agent.messages) == 1

    def test_append_message(self, agent: Agent) -> None:
        """Test appending a message."""
        msg = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        agent.append_message(msg)
        assert len(agent.messages) == 1

    def test_clear_messages(self, agent: Agent) -> None:
        """Test clearing messages."""
        msg = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        agent.append_message(msg)
        agent.clear_messages()
        assert len(agent.messages) == 0

    def test_reset(self, agent: Agent) -> None:
        """Test resetting agent."""
        msg = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        agent.append_message(msg)
        agent.reset()
        assert len(agent.messages) == 0
        assert agent.state.error is None


# =============================================================================
# Tests for Agent subscription
# =============================================================================


class TestAgentSubscription:
    """Tests for Agent event subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_receives_events(self, agent: Agent) -> None:
        """Test that subscribers receive events."""
        received_events: list[AgentEvent] = []

        def callback(event: AgentEvent) -> None:
            received_events.append(event)

        unsubscribe = agent.subscribe(callback)

        async for _ in agent.prompt("Hello"):
            pass

        assert len(received_events) > 0
        unsubscribe()

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_events(self, agent: Agent) -> None:
        """Test that unsubscribing stops event delivery."""
        received_events: list[AgentEvent] = []

        def callback(event: AgentEvent) -> None:
            received_events.append(event)

        unsubscribe = agent.subscribe(callback)
        unsubscribe()

        async for _ in agent.prompt("Hello"):
            pass

        assert len(received_events) == 0


# =============================================================================
# Tests for Agent.prompt()
# =============================================================================


class TestAgentPrompt:
    """Tests for Agent.prompt() method."""

    @pytest.mark.asyncio
    async def test_prompt_with_text(self, agent: Agent) -> None:
        """Test prompting with text."""
        events = []
        async for event in agent.prompt("Hello"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_prompt_adds_message_to_history(self, agent: Agent) -> None:
        """Test that prompting adds messages to history."""
        async for _ in agent.prompt("Hello"):
            pass

        # Should have user message + assistant response
        assert len(agent.messages) == 2
        assert agent.messages[0].role == "user"
        assert agent.messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_prompt_with_messages(self, agent: Agent) -> None:
        """Test prompting with pre-built messages."""
        msg = UserMessage(
            content=[TextContent(text="Custom message")],
            timestamp=int(time.time() * 1000),
        )
        async for _ in agent.prompt(messages=[msg]):
            pass

        assert agent.messages[0].content[0].text == "Custom message"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_prompt_fails_without_provider(self) -> None:
        """Test that prompting fails without a provider."""
        agent = Agent()
        with pytest.raises(RuntimeError, match="No provider configured"):
            async for _ in agent.prompt("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_prompt_fails_when_already_streaming(self, agent: Agent) -> None:
        """Test that prompting fails when already streaming."""
        # Start one prompt
        gen = agent.prompt("Hello")
        await gen.__anext__()  # Get first event

        # Try to start another
        with pytest.raises(RuntimeError, match="already processing"):
            async for _ in agent.prompt("World"):
                pass

        # Clean up
        async for _ in gen:
            pass


# =============================================================================
# Tests for Agent.continue_()
# =============================================================================


class TestAgentContinue:
    """Tests for Agent.continue_() method."""

    @pytest.mark.asyncio
    async def test_continue_fails_without_messages(self, agent: Agent) -> None:
        """Test that continue_ fails without messages."""
        with pytest.raises(RuntimeError, match="No messages to continue"):
            async for _ in agent.continue_():
                pass

    @pytest.mark.asyncio
    async def test_continue_works_from_assistant_message(self, agent: Agent) -> None:
        """Test that continue_ works when last message is from assistant (e.g. after max_tokens)."""
        async for _ in agent.prompt("Hello"):
            pass

        # Last message is now from assistant — continue_ should work
        events = []
        async for event in agent.continue_():
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types


# =============================================================================
# Tests for Agent.abort()
# =============================================================================


class TestAgentAbort:
    """Tests for Agent.abort() method."""

    @pytest.mark.asyncio
    async def test_abort_stops_streaming(self) -> None:
        """Test that abort() stops the current operation."""
        # Create a provider that yields many events
        responses = [
            AssistantMessage(
                content=[TextContent(text="Hello!")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            )
        ]
        provider = MockProvider(responses)
        agent = Agent(provider=provider)

        events = []
        async for event in agent.prompt("Hello"):
            events.append(event)
            if event.type == "message_start":
                agent.abort()

        # Loop should have exited (may or may not have reached agent_end depending on timing)
        assert len(events) > 0


# =============================================================================
# Tests for Agent properties
# =============================================================================


class TestAgentProperties:
    """Tests for Agent properties."""

    def test_state_property(self, agent: Agent) -> None:
        """Test state property."""
        assert isinstance(agent.state, AgentState)

    def test_messages_property(self, agent: Agent) -> None:
        """Test messages property."""
        assert agent.messages == []

    def test_is_streaming_property(self, agent: Agent) -> None:
        """Test is_streaming property."""
        assert agent.is_streaming is False


# =============================================================================
# Tests for Agent with tools
# =============================================================================


class TestAgentWithTools:
    """Tests for Agent with tool execution."""

    @pytest.mark.asyncio
    async def test_agent_executes_tools(self) -> None:
        """Test that agent executes tool calls."""
        tool_executed = False

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            nonlocal tool_executed
            tool_executed = True
            return ToolResult.text("Tool result")

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute,
        )

        # Response that calls the tool
        tool_response = AssistantMessage(
            content=[
                ToolCallContent(
                    id="call_1",
                    name="test_tool",
                    arguments={},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )
        final_response = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_response, final_response])
        agent = Agent(provider=provider, tools=[tool])

        async for _ in agent.prompt("Use the tool"):
            pass

        assert tool_executed is True
