"""Tests for isotope_core.loop module."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from isotope_core.loop import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
    agent_loop,
)
from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDeltaEvent,
)
from isotope_core.tools import Tool, ToolResult
from isotope_core.types import (
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
        """Initialize with a list of responses to return."""
        self.responses = responses
        self.call_count = 0
        self.last_context: Context | None = None

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from the mock provider."""
        self.last_context = context

        if self.call_count >= len(self.responses):
            # Return an end_turn response if we run out
            msg = AssistantMessage(
                content=[TextContent(text="Done")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            )
            yield StreamStartEvent(partial=msg)
            yield StreamDoneEvent(message=msg)
            return

        msg = self.responses[self.call_count]
        self.call_count += 1

        # Check for abort
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

        # Yield start event
        yield StreamStartEvent(partial=msg)

        # Yield deltas for text content
        for content in msg.content:
            if isinstance(content, TextContent):
                yield StreamTextDeltaEvent(
                    content_index=0,
                    delta=content.text,
                    partial=msg,
                )

        # Yield done event
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_response() -> AssistantMessage:
    """Create a simple assistant response."""
    return AssistantMessage(
        content=[TextContent(text="Hello, world!")],
        stop_reason=StopReason.END_TURN,
        timestamp=int(time.time() * 1000),
    )


@pytest.fixture
def tool_call_response() -> AssistantMessage:
    """Create an assistant response with a tool call."""
    return AssistantMessage(
        content=[
            TextContent(text="Let me check the weather."),
            ToolCallContent(
                id="call_1",
                name="get_weather",
                arguments={"location": "NYC"},
            ),
        ],
        stop_reason=StopReason.TOOL_USE,
        timestamp=int(time.time() * 1000),
    )


@pytest.fixture
def weather_tool() -> Tool:
    """Create a weather tool for testing."""

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        location = params.get("location", "unknown")
        return ToolResult.text(f"Weather in {location}: Sunny, 72°F")

    return Tool(
        name="get_weather",
        description="Get the current weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        execute=execute,
    )


# =============================================================================
# Tests
# =============================================================================


class TestAgentLoopBasic:
    """Basic tests for the agent loop."""

    @pytest.mark.asyncio
    async def test_simple_prompt_response(self, simple_response: AssistantMessage) -> None:
        """Test a simple prompt/response cycle."""
        provider = MockProvider([simple_response])
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        context = Context(system_prompt="You are helpful.")

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        # Check event sequence
        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "turn_start" in event_types
        assert "message_start" in event_types
        assert "message_end" in event_types
        assert "turn_end" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_context_passed_to_provider(self, simple_response: AssistantMessage) -> None:
        """Test that context is properly passed to the provider."""
        provider = MockProvider([simple_response])
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        context = Context(system_prompt="Be helpful.")

        async for _ in agent_loop([prompt], context, config):
            pass

        assert provider.last_context is not None
        assert provider.last_context.system_prompt == "Be helpful."
        assert len(provider.last_context.messages) == 1


class TestAgentLoopWithTools:
    """Tests for the agent loop with tool execution."""

    @pytest.mark.asyncio
    async def test_tool_execution(
        self,
        tool_call_response: AssistantMessage,
        simple_response: AssistantMessage,
        weather_tool: Tool,
    ) -> None:
        """Test that tools are executed when called."""
        provider = MockProvider([tool_call_response, simple_response])
        config = AgentLoopConfig(provider=provider, tools=[weather_tool])

        prompt = UserMessage(
            content=[TextContent(text="What's the weather in NYC?")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        # Check that tool events were emitted
        event_types = [e.type for e in events]
        assert "tool_start" in event_types
        assert "tool_end" in event_types

        # Find the tool_end event and check the result
        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.tool_name == "get_weather"  # type: ignore[union-attr]
        assert tool_end.is_error is False  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_tool_not_found(
        self, tool_call_response: AssistantMessage, simple_response: AssistantMessage
    ) -> None:
        """Test handling of unknown tool calls."""
        provider = MockProvider([tool_call_response, simple_response])
        config = AgentLoopConfig(provider=provider, tools=[])  # No tools

        prompt = UserMessage(
            content=[TextContent(text="What's the weather?")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        # Find the tool_end event and check it's an error
        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.is_error is True  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_sequential_tool_execution(
        self,
        weather_tool: Tool,
    ) -> None:
        """Test sequential tool execution mode."""
        # Create a response with multiple tool calls
        response_with_tools = AssistantMessage(
            content=[
                ToolCallContent(id="call_1", name="get_weather", arguments={"location": "NYC"}),
                ToolCallContent(id="call_2", name="get_weather", arguments={"location": "LA"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )
        final_response = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([response_with_tools, final_response])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            tool_execution="sequential",
        )

        prompt = UserMessage(
            content=[TextContent(text="Weather?")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        # Check that both tool calls were made
        tool_ends = [e for e in events if e.type == "tool_end"]
        assert len(tool_ends) == 2


class TestAgentLoopHooks:
    """Tests for agent loop hooks."""

    @pytest.mark.asyncio
    async def test_before_tool_call_hook_allows(
        self,
        tool_call_response: AssistantMessage,
        simple_response: AssistantMessage,
        weather_tool: Tool,
    ) -> None:
        """Test that before_tool_call hook can allow execution."""
        hook_called = False

        async def before_hook(
            ctx: BeforeToolCallContext, signal: asyncio.Event | None
        ) -> BeforeToolCallResult | None:
            nonlocal hook_called
            hook_called = True
            return BeforeToolCallResult(block=False)

        provider = MockProvider([tool_call_response, simple_response])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            before_tool_call=before_hook,
        )

        prompt = UserMessage(
            content=[TextContent(text="Weather?")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        async for _ in agent_loop([prompt], context, config):
            pass

        assert hook_called is True

    @pytest.mark.asyncio
    async def test_before_tool_call_hook_blocks(
        self,
        tool_call_response: AssistantMessage,
        simple_response: AssistantMessage,
        weather_tool: Tool,
    ) -> None:
        """Test that before_tool_call hook can block execution."""

        async def before_hook(
            ctx: BeforeToolCallContext, signal: asyncio.Event | None
        ) -> BeforeToolCallResult:
            return BeforeToolCallResult(block=True, reason="Not allowed")

        provider = MockProvider([tool_call_response, simple_response])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            before_tool_call=before_hook,
        )

        prompt = UserMessage(
            content=[TextContent(text="Weather?")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        # Find the tool_end event and check it's an error
        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.is_error is True  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_after_tool_call_hook_modifies_result(
        self,
        tool_call_response: AssistantMessage,
        simple_response: AssistantMessage,
        weather_tool: Tool,
    ) -> None:
        """Test that after_tool_call hook can modify result."""

        async def after_hook(
            ctx: AfterToolCallContext, signal: asyncio.Event | None
        ) -> AfterToolCallResult:
            return AfterToolCallResult(
                content=[TextContent(text="Modified result")],
                is_error=False,
            )

        provider = MockProvider([tool_call_response, simple_response])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            after_tool_call=after_hook,
        )

        prompt = UserMessage(
            content=[TextContent(text="Weather?")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        # Find the message_end for the tool result
        message_ends = [
            e
            for e in events
            if e.type == "message_end" and hasattr(e.message, "tool_call_id")  # type: ignore[union-attr]
        ]
        assert len(message_ends) == 1
        tool_result_msg = message_ends[0].message  # type: ignore[union-attr]
        assert tool_result_msg.content[0].text == "Modified result"  # type: ignore[union-attr]


class TestAgentLoopAbort:
    """Tests for abort handling in the agent loop."""

    @pytest.mark.asyncio
    async def test_abort_signal_stops_loop(self) -> None:
        """Test that setting abort signal stops the loop."""
        # Create a response that would trigger tool calls
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
        context = Context()

        # Set abort signal before starting
        abort_signal = asyncio.Event()
        abort_signal.set()

        events = []
        async for event in agent_loop([prompt], context, config, signal=abort_signal):
            events.append(event)

        # Check that we got an agent_end event (loop should have exited)
        event_types = [e.type for e in events]
        assert "agent_end" in event_types


class TestAgentLoopTransformContext:
    """Tests for context transformation."""

    @pytest.mark.asyncio
    async def test_transform_context_hook(self, simple_response: AssistantMessage) -> None:
        """Test that transform_context hook is called."""
        transform_called = False

        async def transform(messages: list[Any], signal: asyncio.Event | None) -> list[Any]:
            nonlocal transform_called
            transform_called = True
            return messages

        provider = MockProvider([simple_response])
        config = AgentLoopConfig(
            provider=provider,
            transform_context=transform,
        )

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        async for _ in agent_loop([prompt], context, config):
            pass

        assert transform_called is True

    @pytest.mark.asyncio
    async def test_transform_context_can_modify_messages(
        self, simple_response: AssistantMessage
    ) -> None:
        """Test that transform_context can modify the message list."""

        async def transform(messages: list[Any], signal: asyncio.Event | None) -> list[Any]:
            # Only keep the last message (simulating context pruning)
            return messages[-1:] if messages else messages

        provider = MockProvider([simple_response])
        config = AgentLoopConfig(
            provider=provider,
            transform_context=transform,
        )

        # Add multiple messages
        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        context = Context(
            messages=[
                UserMessage(
                    content=[TextContent(text="Old message")],
                    timestamp=int(time.time() * 1000),
                ),
            ]
        )

        async for _ in agent_loop([prompt], context, config):
            pass

        # The provider should have received only 1 message (the transformed result)
        assert provider.last_context is not None
        assert len(provider.last_context.messages) == 1


# =============================================================================
# Additional coverage tests
# =============================================================================


class _EmptyProvider:
    """Provider that yields no events at all."""

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        return
        yield  # pragma: no cover — makes it an async generator


class _DoneWithoutStartProvider:
    """Provider that yields StreamDoneEvent without a prior StreamStartEvent."""

    def __init__(self, msg: AssistantMessage) -> None:
        self.msg = msg

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        yield StreamDoneEvent(message=self.msg)


class _ErrorStreamProvider:
    """Provider that yields a StreamErrorEvent (error in stream)."""

    from isotope_core.providers.base import StreamErrorEvent

    def __init__(self, msg: AssistantMessage) -> None:
        self.msg = msg

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        from isotope_core.providers.base import StreamErrorEvent

        yield StreamErrorEvent(error=self.msg)


class _ExplodingProvider:
    """Provider whose stream() raises an exception."""

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        raise RuntimeError("Provider exploded")
        yield  # pragma: no cover


class TestAgentLoopBudget:
    """Tests for budget limits (max_turns, max_total_tokens)."""

    @pytest.mark.asyncio
    async def test_max_turns_with_on_agent_end_hook(self) -> None:
        """Test that on_agent_end hook is called with 'max_turns' reason."""
        from isotope_core.middleware import LifecycleHooks

        hook_reason: list[str] = []

        async def on_agent_end(reason: str) -> None:
            hook_reason.append(reason)

        # Provider keeps returning tool calls forever
        tool_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="get_weather", arguments={"location": "NYC"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        async def dummy_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None, on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("ok")

        weather_tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            execute=dummy_execute,
        )

        provider = MockProvider([tool_response, tool_response, tool_response])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            max_turns=1,
            lifecycle_hooks=LifecycleHooks(on_agent_end=on_agent_end),
        )

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_end" in event_types
        agent_end = next(e for e in events if e.type == "agent_end")
        assert agent_end.reason == "max_turns"  # type: ignore[union-attr]
        assert hook_reason == ["max_turns"]

    @pytest.mark.asyncio
    async def test_max_budget_with_on_agent_end_hook(self) -> None:
        """Test that on_agent_end is called with 'max_budget' when token budget exceeded."""
        from isotope_core.middleware import LifecycleHooks
        from isotope_core.types import Usage

        hook_reason: list[str] = []

        async def on_agent_end(reason: str) -> None:
            hook_reason.append(reason)

        # First response uses many tokens (defined inline in mock below)

        async def dummy_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None, on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("ok")

        tool_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="get_weather", arguments={"location": "NYC"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            usage=Usage(input_tokens=100, output_tokens=100),
            timestamp=int(time.time() * 1000),
        )

        weather_tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            execute=dummy_execute,
        )

        provider = MockProvider([tool_response, tool_response, tool_response])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            max_total_tokens=50,  # Very low budget
            lifecycle_hooks=LifecycleHooks(on_agent_end=on_agent_end),
        )

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        agent_end = next(e for e in events if e.type == "agent_end")
        assert agent_end.reason == "max_budget"  # type: ignore[union-attr]
        assert hook_reason == ["max_budget"]


class TestAgentLoopAbortDetailed:
    """Detailed abort signal tests for the agent loop."""

    @pytest.mark.asyncio
    async def test_abort_signal_with_on_agent_end_hook(self) -> None:
        """Test that on_agent_end is called with 'aborted' on abort signal."""
        from isotope_core.middleware import LifecycleHooks

        hook_reason: list[str] = []

        async def on_agent_end(reason: str) -> None:
            hook_reason.append(reason)

        response = AssistantMessage(
            content=[TextContent(text="Hello")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        provider = MockProvider([response])
        config = AgentLoopConfig(
            provider=provider,
            lifecycle_hooks=LifecycleHooks(on_agent_end=on_agent_end),
        )

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))
        abort_signal = asyncio.Event()
        abort_signal.set()

        events = []
        async for event in agent_loop([prompt], Context(), config, signal=abort_signal):
            events.append(event)

        assert hook_reason == ["aborted"]


class TestAgentLoopProviderEdgeCases:
    """Tests for edge cases in provider responses."""

    @pytest.mark.asyncio
    async def test_done_without_start_emits_message_start(self) -> None:
        """Test that done event without prior start emits message_start."""
        msg = AssistantMessage(
            content=[TextContent(text="Quick")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        provider = _DoneWithoutStartProvider(msg)
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        # Should see message_start even though provider only yielded done
        assert event_types.count("message_start") >= 2  # prompt + assistant
        assert "message_end" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_error_event_from_provider(self) -> None:
        """Test that provider error event is handled correctly."""
        msg = AssistantMessage(
            content=[TextContent(text="Error occurred")],
            stop_reason=StopReason.ERROR,
            error_message="API error",
            timestamp=int(time.time() * 1000),
        )
        provider = _ErrorStreamProvider(msg)
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "message_start" in event_types
        assert "message_end" in event_types

    @pytest.mark.asyncio
    async def test_empty_provider_creates_error_message(self) -> None:
        """Test that empty provider (no events) creates error message."""
        provider = _EmptyProvider()
        config = AgentLoopConfig(provider=provider)

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "message_start" in event_types
        assert "message_end" in event_types
        assert "agent_end" in event_types
        # Should end with error reason
        agent_end = next(e for e in events if e.type == "agent_end")
        assert agent_end.reason == "error"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_provider_exception_with_on_error_hook(self) -> None:
        """Test that provider exceptions call on_error hook and re-raise."""
        from isotope_core.middleware import LifecycleHooks

        errors_caught: list[Exception] = []

        async def on_error(exc: Exception) -> None:
            errors_caught.append(exc)

        provider = _ExplodingProvider()
        config = AgentLoopConfig(
            provider=provider,
            lifecycle_hooks=LifecycleHooks(on_error=on_error),
        )

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        with pytest.raises(RuntimeError, match="Provider exploded"):
            async for _ in agent_loop([prompt], Context(), config):
                pass

        assert len(errors_caught) == 1
        assert "Provider exploded" in str(errors_caught[0])


class TestAgentLoopStopReasons:
    """Tests for assistant message stop reason handling."""

    @pytest.mark.asyncio
    async def test_error_stop_reason_with_hooks(self) -> None:
        """Test ERROR stop reason triggers on_turn_end and on_error hooks."""
        from isotope_core.middleware import LifecycleHooks

        hook_calls: list[str] = []

        async def on_turn_end(turn: int, msg: AssistantMessage) -> None:
            hook_calls.append(f"turn_end:{turn}")

        async def on_error(exc: Exception) -> None:
            hook_calls.append("on_error")

        async def on_agent_end(reason: str) -> None:
            hook_calls.append(f"agent_end:{reason}")

        error_response = AssistantMessage(
            content=[TextContent(text="Error")],
            stop_reason=StopReason.ERROR,
            error_message="Something failed",
            timestamp=int(time.time() * 1000),
        )
        provider = MockProvider([error_response])
        config = AgentLoopConfig(
            provider=provider,
            lifecycle_hooks=LifecycleHooks(
                on_turn_end=on_turn_end,
                on_error=on_error,
                on_agent_end=on_agent_end,
            ),
        )

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        assert "turn_end:1" in hook_calls
        assert "on_error" in hook_calls
        assert "agent_end:error" in hook_calls

    @pytest.mark.asyncio
    async def test_aborted_stop_reason_with_hooks(self) -> None:
        """Test ABORTED stop reason triggers on_turn_end and on_agent_end hooks."""
        from isotope_core.middleware import LifecycleHooks

        hook_calls: list[str] = []

        async def on_turn_end(turn: int, msg: AssistantMessage) -> None:
            hook_calls.append(f"turn_end:{turn}")

        async def on_agent_end(reason: str) -> None:
            hook_calls.append(f"agent_end:{reason}")

        aborted_response = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason=StopReason.ABORTED,
            error_message="Aborted by user",
            timestamp=int(time.time() * 1000),
        )
        provider = MockProvider([aborted_response])
        config = AgentLoopConfig(
            provider=provider,
            lifecycle_hooks=LifecycleHooks(
                on_turn_end=on_turn_end,
                on_agent_end=on_agent_end,
            ),
        )

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        assert "turn_end:1" in hook_calls
        assert "agent_end:aborted" in hook_calls


class TestAgentLoopToolEdgeCases:
    """Tests for tool execution edge cases."""

    @pytest.mark.asyncio
    async def test_tool_validation_failure(self) -> None:
        """Test that invalid tool arguments are handled gracefully."""

        async def strict_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None, on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("ok")

        strict_tool = Tool(
            name="strict_tool",
            description="A tool with strict params",
            parameters={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            execute=strict_execute,
        )

        # Call with missing required argument
        tool_call_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="strict_tool", arguments={}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        final = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_call_response, final])
        config = AgentLoopConfig(provider=provider, tools=[strict_tool])

        prompt = UserMessage(content=[TextContent(text="Go")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.is_error is True  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_before_tool_call_hook_raises_exception(self) -> None:
        """Test that exceptions in before_tool_call hook are handled."""

        async def exploding_before_hook(
            ctx: BeforeToolCallContext, signal: asyncio.Event | None
        ) -> BeforeToolCallResult:
            raise RuntimeError("Hook exploded")

        async def dummy_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None, on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("ok")

        weather_tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            execute=dummy_execute,
        )

        tool_call_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="get_weather", arguments={"location": "NYC"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        final = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_call_response, final])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            before_tool_call=exploding_before_hook,
        )

        prompt = UserMessage(content=[TextContent(text="Go")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.is_error is True  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_tool_update_callback(self) -> None:
        """Test that on_update callback is invoked during tool execution."""

        async def streaming_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            if on_update:
                on_update(ToolResult.text("partial result..."))
            return ToolResult.text("final result")

        streaming_tool = Tool(
            name="streaming_tool",
            description="A tool that streams updates",
            parameters={"type": "object", "properties": {}},
            execute=streaming_execute,
        )

        tool_call_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="streaming_tool", arguments={}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        final = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_call_response, final])
        config = AgentLoopConfig(
            provider=provider,
            tools=[streaming_tool],
            tool_execution="sequential",
        )

        prompt = UserMessage(content=[TextContent(text="Go")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "tool_update" in event_types

    @pytest.mark.asyncio
    async def test_after_tool_call_hook_modifies_is_error(self) -> None:
        """Test that after_tool_call hook can change is_error flag."""

        async def mark_not_error_hook(
            ctx: AfterToolCallContext, signal: asyncio.Event | None
        ) -> AfterToolCallResult:
            # Even if tool returned an error, mark it as not an error
            return AfterToolCallResult(is_error=False)

        async def failing_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None, on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.error("Something went wrong")

        failing_tool = Tool(
            name="failing_tool",
            description="A tool that fails",
            parameters={"type": "object", "properties": {}},
            execute=failing_execute,
        )

        tool_call_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="failing_tool", arguments={}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        final = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_call_response, final])
        config = AgentLoopConfig(
            provider=provider,
            tools=[failing_tool],
            after_tool_call=mark_not_error_hook,
        )

        prompt = UserMessage(content=[TextContent(text="Go")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        tool_end = next(e for e in events if e.type == "tool_end")
        # The hook changed is_error to False
        assert tool_end.is_error is False  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_after_tool_call_hook_exception_silenced(self) -> None:
        """Test that exceptions in after_tool_call hook are silenced."""

        async def exploding_after_hook(
            ctx: AfterToolCallContext, signal: asyncio.Event | None
        ) -> AfterToolCallResult:
            raise RuntimeError("After hook exploded")

        async def dummy_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None, on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("ok")

        weather_tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            execute=dummy_execute,
        )

        tool_call_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="get_weather", arguments={"location": "NYC"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        final = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        provider = MockProvider([tool_call_response, final])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            after_tool_call=exploding_after_hook,
        )

        prompt = UserMessage(content=[TextContent(text="Go")], timestamp=int(time.time() * 1000))

        # Should not raise — the exception is silently caught
        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        assert any(e.type == "agent_end" for e in events)


class TestAgentLoopSteeringFollowUp:
    """Tests for steering and follow-up queue edge cases."""

    @pytest.mark.asyncio
    async def test_steering_queue_empty_race(self) -> None:
        """Test steering queue handling when QueueEmpty is raised during drain."""
        # Use a tool call to trigger the steering check path
        async def dummy_execute(
            tool_call_id: str, params: dict[str, Any],
            signal: asyncio.Event | None = None, on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text("ok")

        weather_tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            execute=dummy_execute,
        )

        tool_call_response = AssistantMessage(
            content=[
                ToolCallContent(id="c1", name="get_weather", arguments={"location": "NYC"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=int(time.time() * 1000),
        )

        final = AssistantMessage(
            content=[TextContent(text="Done")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        steering_queue: asyncio.Queue[Any] = asyncio.Queue()
        # Put a message in the steering queue so it gets processed
        steer_msg = UserMessage(
            content=[TextContent(text="Go faster")],
            timestamp=int(time.time() * 1000),
        )
        steering_queue.put_nowait(steer_msg)

        provider = MockProvider([tool_call_response, final])
        config = AgentLoopConfig(
            provider=provider,
            tools=[weather_tool],
            steering_queue=steering_queue,
        )

        prompt = UserMessage(content=[TextContent(text="Go")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "steer" in event_types

    @pytest.mark.asyncio
    async def test_follow_up_queue_triggers_new_turn(self) -> None:
        """Test that follow-up queue messages trigger new turns."""
        response1 = AssistantMessage(
            content=[TextContent(text="First answer")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        response2 = AssistantMessage(
            content=[TextContent(text="Second answer")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )

        follow_up_queue: asyncio.Queue[Any] = asyncio.Queue()
        followup_msg = UserMessage(
            content=[TextContent(text="Follow up question")],
            timestamp=int(time.time() * 1000),
        )
        follow_up_queue.put_nowait(followup_msg)

        provider = MockProvider([response1, response2])
        config = AgentLoopConfig(
            provider=provider,
            follow_up_queue=follow_up_queue,
        )

        prompt = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

        events = []
        async for event in agent_loop([prompt], Context(), config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "follow_up" in event_types
        # Should have at least 2 turn_start events (original + follow-up)
        assert event_types.count("turn_start") >= 2
