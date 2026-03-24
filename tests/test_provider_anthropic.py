"""Tests for Anthropic provider."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from isotope_core.providers.anthropic import (
    AnthropicProvider,
    ThinkingConfig,
    _convert_context_to_anthropic,
    _convert_tools,
    _map_stop_reason,
)
from isotope_core.types import (
    AssistantMessage,
    Context,
    ImageContent,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultMessage,
    ToolSchema,
    Usage,
    UserMessage,
)

# =============================================================================
# Helper to build mock Anthropic streaming events
# =============================================================================


def _mock_event(event_type: str, **kwargs: Any) -> MagicMock:
    """Build a mock Anthropic streaming event."""
    event = MagicMock()
    event.type = event_type

    if event_type == "message_start":
        msg = MagicMock()
        msg.id = kwargs.get("id", "msg_123")
        msg.usage = MagicMock()
        msg.usage.input_tokens = kwargs.get("input_tokens", 10)
        msg.usage.output_tokens = kwargs.get("output_tokens", 0)
        msg.usage.cache_read_input_tokens = kwargs.get("cache_read", 0)
        msg.usage.cache_creation_input_tokens = kwargs.get("cache_write", 0)
        event.message = msg

    elif event_type == "content_block_start":
        event.index = kwargs.get("index", 0)
        cb = MagicMock()
        cb.type = kwargs.get("block_type", "text")
        if cb.type == "tool_use":
            cb.id = kwargs.get("tool_id", "toolu_1")
            cb.name = kwargs.get("tool_name", "read_file")
            cb.input = kwargs.get("tool_input", {})
        elif cb.type == "redacted_thinking":
            cb.data = kwargs.get("data", "redacted_sig")
        event.content_block = cb

    elif event_type == "content_block_delta":
        event.index = kwargs.get("index", 0)
        delta = MagicMock()
        delta.type = kwargs.get("delta_type", "text_delta")
        if delta.type == "text_delta":
            delta.text = kwargs.get("text", "")
        elif delta.type == "thinking_delta":
            delta.thinking = kwargs.get("thinking", "")
        elif delta.type == "input_json_delta":
            delta.partial_json = kwargs.get("partial_json", "")
        elif delta.type == "signature_delta":
            delta.signature = kwargs.get("signature", "")
        event.delta = delta

    elif event_type == "content_block_stop":
        event.index = kwargs.get("index", 0)

    elif event_type == "message_delta":
        delta = MagicMock()
        delta.stop_reason = kwargs.get("stop_reason")
        event.delta = delta
        usage = MagicMock()
        usage.output_tokens = kwargs.get("output_tokens", 0)
        event.usage = usage

    return event


class TestConvertContextToAnthropic:
    """Tests for context to Anthropic message conversion."""

    def test_system_prompt(self) -> None:
        ctx = Context(system_prompt="You are helpful", messages=[])
        system, messages, tools = _convert_context_to_anthropic(ctx)
        assert system == "You are helpful"
        assert messages == []
        assert tools is None

    def test_user_message_text(self) -> None:
        ctx = Context(messages=[UserMessage(content=[TextContent(text="hello")], timestamp=0)])
        _, messages, _ = _convert_context_to_anthropic(ctx)
        assert messages[0] == {"role": "user", "content": "hello"}

    def test_user_message_with_image(self) -> None:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="describe"),
                        ImageContent(data="imgdata", mime_type="image/jpeg"),
                    ],
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["media_type"] == "image/jpeg"

    def test_assistant_message_with_tool_use(self) -> None:
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[
                        TextContent(text="I'll read that file."),
                        ToolCallContent(id="toolu_1", name="read_file", arguments={"path": "/a"}),
                    ],
                    stop_reason=StopReason.TOOL_USE,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        msg = messages[0]
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "toolu_1"
        assert content[1]["name"] == "read_file"

    def test_tool_result_in_user_message(self) -> None:
        ctx = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="toolu_1",
                    tool_name="read_file",
                    content=[TextContent(text="file contents")],
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        msg = messages[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "toolu_1"

    def test_consecutive_tool_results_merged(self) -> None:
        ctx = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="t1",
                    tool_name="t1_fn",
                    content=[TextContent(text="r1")],
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="t2",
                    tool_name="t2_fn",
                    content=[TextContent(text="r2")],
                    timestamp=0,
                ),
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        # Both should be in the same user message
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2

    def test_thinking_with_signature(self) -> None:
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="deep thoughts", thinking_signature="sig123"),
                        TextContent(text="answer"),
                    ],
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        content = messages[0]["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["signature"] == "sig123"

    def test_redacted_thinking(self) -> None:
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[
                        ThinkingContent(
                            thinking="[Reasoning redacted]",
                            thinking_signature="redacted_data",
                            redacted=True,
                        ),
                        TextContent(text="result"),
                    ],
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        content = messages[0]["content"]
        assert content[0]["type"] == "redacted_thinking"
        assert content[0]["data"] == "redacted_data"


class TestConvertToolsAnthropic:
    """Tests for tool schema conversion to Anthropic format."""

    def test_tool_conversion(self) -> None:
        tools = [
            ToolSchema(
                name="read_file",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]
        result = _convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["input_schema"]["type"] == "object"
        assert "path" in result[0]["input_schema"]["properties"]


class TestMapStopReasonAnthropic:
    """Tests for Anthropic stop reason mapping."""

    def test_end_turn(self) -> None:
        assert _map_stop_reason("end_turn") == StopReason.END_TURN

    def test_max_tokens(self) -> None:
        assert _map_stop_reason("max_tokens") == StopReason.MAX_TOKENS

    def test_tool_use(self) -> None:
        assert _map_stop_reason("tool_use") == StopReason.TOOL_USE

    def test_stop_sequence(self) -> None:
        assert _map_stop_reason("stop_sequence") == StopReason.END_TURN


class TestAnthropicProviderInit:
    """Tests for AnthropicProvider initialization."""

    def test_default_init(self) -> None:
        provider = AnthropicProvider()
        assert provider.model == "claude-sonnet-4-20250514"

    def test_custom_model(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-20250514")
        assert provider.model == "claude-opus-4-20250514"

    def test_thinking_config(self) -> None:
        thinking = ThinkingConfig(enabled=True, budget_tokens=4096)
        provider = AnthropicProvider(thinking=thinking)
        assert provider._thinking is not None
        assert provider._thinking.enabled is True
        assert provider._thinking.budget_tokens == 4096

    def test_import_error_when_no_sdk(self) -> None:
        provider = AnthropicProvider()
        with (
            patch.dict("sys.modules", {"anthropic": None}),
            pytest.raises(ImportError, match="anthropic"),
        ):
            provider._get_client()


class TestAnthropicProviderStream:
    """Tests for Anthropic streaming."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """Test streaming a simple text response."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        events_list = [
            _mock_event("message_start", input_tokens=10, output_tokens=0),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="Hello"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text=" world"),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=5),
        ]

        # Build async context manager + async iterator for stream
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(
                system_prompt="be helpful",
                messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
            )
        ):
            collected.append(event)

        event_types = [e.type for e in collected]
        assert "start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

        done_event = [e for e in collected if e.type == "done"][0]
        assert done_event.message.stop_reason == StopReason.END_TURN

    @pytest.mark.asyncio
    async def test_tool_call_response(self) -> None:
        """Test streaming a tool call response."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event(
                "content_block_start",
                index=0,
                block_type="tool_use",
                tool_id="toolu_1",
                tool_name="read_file",
            ),
            _mock_event(
                "content_block_delta",
                index=0,
                delta_type="input_json_delta",
                partial_json='{"path":',
            ),
            _mock_event(
                "content_block_delta",
                index=0,
                delta_type="input_json_delta",
                partial_json='"/tmp/a"}',
            ),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="tool_use", output_tokens=20),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(
                messages=[UserMessage(content=[TextContent(text="read /tmp/a")], timestamp=0)],
                tools=[
                    ToolSchema(
                        name="read_file",
                        description="Read a file",
                        parameters={
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    )
                ],
            )
        ):
            collected.append(event)

        event_types = [e.type for e in collected]
        assert "tool_call_start" in event_types
        assert "tool_call_delta" in event_types
        assert "tool_call_end" in event_types
        assert "done" in event_types

        tc_end = [e for e in collected if e.type == "tool_call_end"][0]
        assert tc_end.tool_name == "read_file"
        assert tc_end.arguments == {"path": "/tmp/a"}

    @pytest.mark.asyncio
    async def test_thinking_response(self) -> None:
        """Test streaming with extended thinking."""
        provider = AnthropicProvider(
            model="claude-sonnet-4-20250514",
            thinking=ThinkingConfig(enabled=True, budget_tokens=4096),
        )

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event("content_block_start", index=0, block_type="thinking"),
            _mock_event(
                "content_block_delta",
                index=0,
                delta_type="thinking_delta",
                thinking="Let me think...",
            ),
            _mock_event(
                "content_block_delta", index=0, delta_type="signature_delta", signature="sig123"
            ),
            _mock_event("content_block_stop", index=0),
            _mock_event("content_block_start", index=1, block_type="text"),
            _mock_event("content_block_delta", index=1, delta_type="text_delta", text="The answer"),
            _mock_event("content_block_stop", index=1),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=30),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="think")], timestamp=0)])
        ):
            collected.append(event)

        event_types = [e.type for e in collected]
        assert "thinking_delta" in event_types
        assert "thinking_end" in event_types
        assert "text_delta" in event_types

    @pytest.mark.asyncio
    async def test_abort_signal(self) -> None:
        """Test that abort signal stops streaming."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        signal = asyncio.Event()

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="Hello"),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e
            signal.set()
            yield _mock_event("content_block_delta", index=0, delta_type="text_delta", text=" more")

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)]),
            signal=signal,
        ):
            collected.append(event)

        last_event = collected[-1]
        assert last_event.type == "error"

    @pytest.mark.asyncio
    async def test_api_error_yields_error_event(self) -> None:
        """Test that API errors yield StreamErrorEvent."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(side_effect=Exception("Invalid API key"))
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            collected.append(event)

        assert len(collected) == 1
        assert collected[0].type == "error"
        assert "Invalid API key" in (collected[0].error.error_message or "")

    @pytest.mark.asyncio
    async def test_usage_extraction(self) -> None:
        """Test that usage is correctly extracted."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        events_list = [
            _mock_event(
                "message_start", input_tokens=100, output_tokens=0, cache_read=20, cache_write=5
            ),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="ok"),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=50),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            collected.append(event)

        done_event = [e for e in collected if e.type == "done"][0]
        assert done_event.message.usage.input_tokens == 100
        assert done_event.message.usage.output_tokens == 50
        assert done_event.message.usage.cache_read_tokens == 20
        assert done_event.message.usage.cache_write_tokens == 5

    @pytest.mark.asyncio
    async def test_redacted_thinking(self) -> None:
        """Test handling of redacted thinking blocks."""
        provider = AnthropicProvider(
            model="claude-sonnet-4-20250514",
            thinking=ThinkingConfig(enabled=True, budget_tokens=4096),
        )

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event(
                "content_block_start",
                index=0,
                block_type="redacted_thinking",
                data="redacted_sig_data",
            ),
            _mock_event("content_block_stop", index=0),
            _mock_event("content_block_start", index=1, block_type="text"),
            _mock_event("content_block_delta", index=1, delta_type="text_delta", text="answer"),
            _mock_event("content_block_stop", index=1),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=10),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="think")], timestamp=0)])
        ):
            collected.append(event)

        event_types = [e.type for e in collected]
        assert "thinking_delta" in event_types
        assert "done" in event_types


class TestAnthropicProviderProperties:
    """Tests for provider properties."""

    def test_model_name_property(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-20250514")
        assert provider.model_name == "claude-opus-4-20250514"

    def test_provider_name_property(self) -> None:
        provider = AnthropicProvider()
        assert provider.provider_name == "anthropic"


class TestParseStreamingJson:
    """Tests for _parse_streaming_json."""

    def test_empty_string(self) -> None:
        from isotope_core.providers.anthropic import _parse_streaming_json

        assert _parse_streaming_json("") == {}

    def test_valid_json(self) -> None:
        from isotope_core.providers.anthropic import _parse_streaming_json

        assert _parse_streaming_json('{"key": "value"}') == {"key": "value"}

    def test_incomplete_json_repair(self) -> None:
        from isotope_core.providers.anthropic import _parse_streaming_json

        result = _parse_streaming_json('{"key": "value"')
        assert result == {"key": "value"}

    def test_unfixable_json(self) -> None:
        from isotope_core.providers.anthropic import _parse_streaming_json

        assert _parse_streaming_json("{broken: json: bad") == {}


class TestMapStopReasonUnknown:
    """Test unknown stop_reason mapping."""

    def test_unknown_stop_reason(self) -> None:
        assert _map_stop_reason("unknown_reason") == StopReason.ERROR


class TestConvertAssistantMessageEdgeCases:
    """Edge cases for assistant message conversion."""

    def test_empty_assistant_message_returns_none(self) -> None:
        """Empty text in assistant message should return None."""
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[TextContent(text="   ")],
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        # The text is all whitespace, so content list is empty => returns None
        assert len(messages) == 0

    def test_thinking_without_signature_falls_back_to_text(self) -> None:
        """Thinking content without signature should be converted to text."""
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="my thoughts", thinking_signature=None),
                        TextContent(text="answer"),
                    ],
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        content = messages[0]["content"]
        # thinking without signature -> text block
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "my thoughts"

    def test_tool_result_with_image_content(self) -> None:
        """Test tool result containing image content."""
        ctx = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="t1",
                    tool_name="screenshot",
                    content=[ImageContent(data="base64data", mime_type="image/png")],
                    timestamp=0,
                )
            ]
        )
        _, messages, _ = _convert_context_to_anthropic(ctx)
        tool_result = messages[0]["content"][0]
        assert tool_result["type"] == "tool_result"
        # Content should contain an image block
        content = tool_result["content"]
        assert isinstance(content, list)
        assert any(b.get("type") == "image" for b in content)


class TestAnthropicProviderStreamEdgeCases:
    """Edge-case streaming tests."""

    @pytest.mark.asyncio
    async def test_no_stop_reason_defaults_to_end_turn(self) -> None:
        """Test that missing stop_reason defaults to END_TURN."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="Hi"),
            _mock_event("content_block_stop", index=0),
            # message_delta with no stop_reason
            _mock_event("message_delta", stop_reason=None, output_tokens=5),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            collected.append(event)

        done_event = [e for e in collected if e.type == "done"][0]
        assert done_event.message.stop_reason == StopReason.END_TURN

    @pytest.mark.asyncio
    async def test_content_block_delta_unknown_index_skipped(self) -> None:
        """Test that deltas for unknown block indices are skipped."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        events_list = [
            _mock_event("message_start", input_tokens=10),
            # Delta for index 99 that was never started
            _mock_event("content_block_delta", index=99, delta_type="text_delta", text="ghost"),
            _mock_event("content_block_stop", index=99),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="real"),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=5),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            collected.append(event)

        done_event = [e for e in collected if e.type == "done"][0]
        # Only real text should be captured
        text_blocks = [b for b in done_event.message.content if isinstance(b, TextContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "real"

    @pytest.mark.asyncio
    async def test_thinking_params_not_set_when_disabled(self) -> None:
        """Test that thinking params are not included when thinking is disabled."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="ok"),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=5),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)]),
            temperature=0.5,
        ):
            collected.append(event)

        # Check temperature was passed (no thinking)
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs.get("temperature") == 0.5

    @pytest.mark.asyncio
    async def test_temperature_ignored_with_thinking(self) -> None:
        """Test that temperature is not passed when thinking is enabled."""
        provider = AnthropicProvider(
            model="claude-sonnet-4-20250514",
            thinking=ThinkingConfig(enabled=True, budget_tokens=1024),
        )

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="ok"),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=5),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        async for _ in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)]),
            temperature=0.5,
        ):
            pass

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert "temperature" not in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"

    @pytest.mark.asyncio
    async def test_api_key_resolver(self) -> None:
        """Test that api_key_resolver is called before streaming."""

        async def resolver() -> str:
            return "resolved-key-123"

        provider = AnthropicProvider(
            model="claude-sonnet-4-20250514",
            api_key_resolver=resolver,
        )

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="ok"),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=5),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            collected.append(event)

        # The client's api_key should have been set to the resolved key
        assert mock_client.api_key == "resolved-key-123"

    @pytest.mark.asyncio
    async def test_exception_with_abort_signal(self) -> None:
        """Test exception during streaming when abort signal is set."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        signal = asyncio.Event()
        signal.set()  # Already aborted

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(side_effect=Exception("Connection lost"))
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)]),
            signal=signal,
        ):
            collected.append(event)

        assert len(collected) == 1
        assert collected[0].type == "error"
        assert collected[0].error.stop_reason == StopReason.ABORTED

    @pytest.mark.asyncio
    async def test_message_delta_with_no_usage_output_tokens(self) -> None:
        """Test message_delta where usage.output_tokens is 0/None."""
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        events_list = [
            _mock_event("message_start", input_tokens=10),
            _mock_event("content_block_start", index=0, block_type="text"),
            _mock_event("content_block_delta", index=0, delta_type="text_delta", text="ok"),
            _mock_event("content_block_stop", index=0),
            _mock_event("message_delta", stop_reason="end_turn", output_tokens=0),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_aiter(self):
            for e in events_list:
                yield e

        mock_stream.__aiter__ = mock_aiter

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        collected = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            collected.append(event)

        done = [e for e in collected if e.type == "done"][0]
        assert done.message.usage.output_tokens == 0
