"""Tests for isotope_core.types module."""

from isotope_core.types import (
    AgentEndEvent,
    AgentStartEvent,
    AssistantMessage,
    Context,
    ImageContent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolEndEvent,
    ToolResultMessage,
    ToolSchema,
    ToolStartEvent,
    ToolUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
    UserMessage,
)


class TestTextContent:
    """Tests for TextContent."""

    def test_create_text_content(self) -> None:
        """Test creating a text content block."""
        content = TextContent(text="Hello, world!")
        assert content.type == "text"
        assert content.text == "Hello, world!"

    def test_text_content_serialization(self) -> None:
        """Test serializing text content to dict."""
        content = TextContent(text="Hello")
        data = content.model_dump()
        assert data == {"type": "text", "text": "Hello"}

    def test_text_content_from_dict(self) -> None:
        """Test creating text content from dict."""
        content = TextContent.model_validate({"type": "text", "text": "Hello"})
        assert content.text == "Hello"


class TestImageContent:
    """Tests for ImageContent."""

    def test_create_image_content(self) -> None:
        """Test creating an image content block."""
        content = ImageContent(data="base64data", mime_type="image/png")
        assert content.type == "image"
        assert content.data == "base64data"
        assert content.mime_type == "image/png"

    def test_image_content_serialization(self) -> None:
        """Test serializing image content to dict."""
        content = ImageContent(data="abc", mime_type="image/jpeg")
        data = content.model_dump()
        assert data == {"type": "image", "data": "abc", "mime_type": "image/jpeg"}


class TestThinkingContent:
    """Tests for ThinkingContent."""

    def test_create_thinking_content(self) -> None:
        """Test creating a thinking content block."""
        content = ThinkingContent(thinking="Let me think...")
        assert content.type == "thinking"
        assert content.thinking == "Let me think..."
        assert content.thinking_signature is None
        assert content.redacted is False

    def test_thinking_content_with_signature(self) -> None:
        """Test thinking content with signature."""
        content = ThinkingContent(
            thinking="Hidden",
            thinking_signature="sig123",
            redacted=True,
        )
        assert content.thinking_signature == "sig123"
        assert content.redacted is True


class TestToolCallContent:
    """Tests for ToolCallContent."""

    def test_create_tool_call_content(self) -> None:
        """Test creating a tool call content block."""
        content = ToolCallContent(
            id="call_123",
            name="get_weather",
            arguments={"location": "NYC"},
        )
        assert content.type == "tool_call"
        assert content.id == "call_123"
        assert content.name == "get_weather"
        assert content.arguments == {"location": "NYC"}


class TestUsage:
    """Tests for Usage."""

    def test_default_usage(self) -> None:
        """Test default usage values."""
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_with_values(self) -> None:
        """Test usage with specific values."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_write_tokens=10,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150


class TestUserMessage:
    """Tests for UserMessage."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = UserMessage(
            content=[TextContent(text="Hello")],
            timestamp=1234567890000,
        )
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hello"
        assert msg.timestamp == 1234567890000

    def test_user_message_with_image(self) -> None:
        """Test user message with image."""
        msg = UserMessage(
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(data="base64", mime_type="image/png"),
            ],
            timestamp=1234567890000,
        )
        assert len(msg.content) == 2


class TestAssistantMessage:
    """Tests for AssistantMessage."""

    def test_create_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = AssistantMessage(
            content=[TextContent(text="Hello!")],
            stop_reason=StopReason.END_TURN,
            timestamp=1234567890000,
        )
        assert msg.role == "assistant"
        assert msg.stop_reason == StopReason.END_TURN
        assert msg.error_message is None

    def test_assistant_message_with_tool_call(self) -> None:
        """Test assistant message with tool call."""
        msg = AssistantMessage(
            content=[
                TextContent(text="Let me check the weather."),
                ToolCallContent(
                    id="call_1",
                    name="get_weather",
                    arguments={"location": "NYC"},
                ),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=1234567890000,
        )
        assert msg.stop_reason == StopReason.TOOL_USE
        assert len(msg.content) == 2

    def test_assistant_message_with_error(self) -> None:
        """Test assistant message with error."""
        msg = AssistantMessage(
            content=[],
            stop_reason=StopReason.ERROR,
            error_message="API error",
            timestamp=1234567890000,
        )
        assert msg.stop_reason == StopReason.ERROR
        assert msg.error_message == "API error"


class TestToolResultMessage:
    """Tests for ToolResultMessage."""

    def test_create_tool_result_message(self) -> None:
        """Test creating a tool result message."""
        msg = ToolResultMessage(
            tool_call_id="call_123",
            tool_name="get_weather",
            content=[TextContent(text="Sunny, 72°F")],
            timestamp=1234567890000,
        )
        assert msg.role == "tool_result"
        assert msg.tool_call_id == "call_123"
        assert msg.tool_name == "get_weather"
        assert msg.is_error is False

    def test_tool_result_message_with_error(self) -> None:
        """Test tool result message with error."""
        msg = ToolResultMessage(
            tool_call_id="call_123",
            tool_name="get_weather",
            content=[TextContent(text="Location not found")],
            is_error=True,
            timestamp=1234567890000,
        )
        assert msg.is_error is True


class TestContext:
    """Tests for Context."""

    def test_empty_context(self) -> None:
        """Test creating an empty context."""
        ctx = Context()
        assert ctx.system_prompt == ""
        assert ctx.messages == []
        assert ctx.tools == []

    def test_context_with_data(self) -> None:
        """Test context with data."""
        ctx = Context(
            system_prompt="You are helpful.",
            messages=[
                UserMessage(
                    content=[TextContent(text="Hi")],
                    timestamp=1234567890000,
                ),
            ],
            tools=[
                ToolSchema(
                    name="test",
                    description="A test tool",
                    parameters={"type": "object"},
                ),
            ],
        )
        assert ctx.system_prompt == "You are helpful."
        assert len(ctx.messages) == 1
        assert len(ctx.tools) == 1


class TestStopReason:
    """Tests for StopReason enum."""

    def test_stop_reason_values(self) -> None:
        """Test stop reason enum values."""
        assert StopReason.END_TURN.value == "end_turn"
        assert StopReason.TOOL_USE.value == "tool_use"
        assert StopReason.MAX_TOKENS.value == "max_tokens"
        assert StopReason.ERROR.value == "error"
        assert StopReason.ABORTED.value == "aborted"


class TestAgentEvents:
    """Tests for agent event types."""

    def test_agent_start_event(self) -> None:
        """Test AgentStartEvent."""
        event = AgentStartEvent()
        assert event.type == "agent_start"

    def test_agent_end_event(self) -> None:
        """Test AgentEndEvent."""
        msg = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=1234567890000,
        )
        event = AgentEndEvent(messages=[msg])
        assert event.type == "agent_end"
        assert len(event.messages) == 1

    def test_turn_start_event(self) -> None:
        """Test TurnStartEvent."""
        event = TurnStartEvent()
        assert event.type == "turn_start"

    def test_turn_end_event(self) -> None:
        """Test TurnEndEvent."""
        msg = AssistantMessage(
            content=[TextContent(text="Hello")],
            timestamp=1234567890000,
        )
        event = TurnEndEvent(message=msg)
        assert event.type == "turn_end"
        assert event.tool_results == []

    def test_message_start_event(self) -> None:
        """Test MessageStartEvent."""
        msg = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=1234567890000,
        )
        event = MessageStartEvent(message=msg)
        assert event.type == "message_start"

    def test_message_update_event(self) -> None:
        """Test MessageUpdateEvent."""
        msg = AssistantMessage(
            content=[TextContent(text="Hel")],
            timestamp=1234567890000,
        )
        event = MessageUpdateEvent(message=msg, delta="lo")
        assert event.type == "message_update"
        assert event.delta == "lo"

    def test_message_end_event(self) -> None:
        """Test MessageEndEvent."""
        msg = AssistantMessage(
            content=[TextContent(text="Hello")],
            timestamp=1234567890000,
        )
        event = MessageEndEvent(message=msg)
        assert event.type == "message_end"

    def test_tool_start_event(self) -> None:
        """Test ToolStartEvent."""
        event = ToolStartEvent(
            tool_call_id="call_1",
            tool_name="get_weather",
            args={"location": "NYC"},
        )
        assert event.type == "tool_start"
        assert event.tool_call_id == "call_1"

    def test_tool_update_event(self) -> None:
        """Test ToolUpdateEvent."""
        event = ToolUpdateEvent(
            tool_call_id="call_1",
            tool_name="get_weather",
            args={"location": "NYC"},
            partial_result={"progress": 50},
        )
        assert event.type == "tool_update"

    def test_tool_end_event(self) -> None:
        """Test ToolEndEvent."""
        event = ToolEndEvent(
            tool_call_id="call_1",
            tool_name="get_weather",
            result={"weather": "sunny"},
            is_error=False,
        )
        assert event.type == "tool_end"
        assert event.is_error is False
