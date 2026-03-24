"""Tests for OpenAI provider."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from isotope_core.providers.openai import (
    OpenAIProvider,
    _convert_context_to_openai,
    _convert_tools,
    _map_finish_reason,
    _parse_streaming_json,
)
from isotope_core.types import (
    AssistantMessage,
    Context,
    ImageContent,
    StopReason,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    ToolSchema,
    Usage,
    UserMessage,
)

# =============================================================================
# Helpers to build mock streaming chunks
# =============================================================================


def _mock_chunk(
    *,
    content: str | None = None,
    tool_call_id: str | None = None,
    tool_call_name: str | None = None,
    tool_call_args: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    reasoning_content: str | None = None,
) -> MagicMock:
    """Build a mock OpenAI streaming chunk."""
    chunk = MagicMock()
    chunk.id = "chatcmpl-123"
    chunk.usage = None

    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None

    # Reasoning content for o-series models
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    else:
        delta.reasoning_content = None
    delta.reasoning = None

    if tool_call_id or tool_call_name or tool_call_args:
        tc = MagicMock()
        tc.index = 0
        tc.id = tool_call_id
        tc.function = MagicMock()
        tc.function.name = tool_call_name
        tc.function.arguments = tool_call_args
        delta.tool_calls = [tc]

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]

    if usage:
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = usage.get("prompt_tokens", 0)
        mock_usage.completion_tokens = usage.get("completion_tokens", 0)
        mock_usage.prompt_tokens_details = MagicMock()
        mock_usage.prompt_tokens_details.cached_tokens = usage.get("cached_tokens", 0)
        chunk.usage = mock_usage

    return chunk


class TestParseStreamingJson:
    """Tests for _parse_streaming_json."""

    def test_complete_json(self) -> None:
        assert _parse_streaming_json('{"key": "value"}') == {"key": "value"}

    def test_incomplete_json(self) -> None:
        result = _parse_streaming_json('{"key": "val')
        # Should attempt repair or return empty
        assert isinstance(result, dict)

    def test_empty_string(self) -> None:
        assert _parse_streaming_json("") == {}


class TestConvertContextToOpenAI:
    """Tests for context to OpenAI message conversion."""

    def test_system_prompt(self) -> None:
        ctx = Context(system_prompt="You are helpful", messages=[])
        messages, tools = _convert_context_to_openai(ctx)
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert tools is None

    def test_user_message_text(self) -> None:
        ctx = Context(messages=[UserMessage(content=[TextContent(text="hello")], timestamp=0)])
        messages, _ = _convert_context_to_openai(ctx)
        assert messages[0] == {"role": "user", "content": "hello"}

    def test_user_message_with_image(self) -> None:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="describe this"),
                        ImageContent(data="base64data", mime_type="image/png"),
                    ],
                    timestamp=0,
                )
            ]
        )
        messages, _ = _convert_context_to_openai(ctx)
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64,base64data" in content[1]["image_url"]["url"]

    def test_assistant_message_text(self) -> None:
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[TextContent(text="hi there")],
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        messages, _ = _convert_context_to_openai(ctx)
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "hi there"

    def test_assistant_message_with_tool_call(self) -> None:
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[
                        ToolCallContent(id="tc_1", name="read_file", arguments={"path": "/tmp/a"})
                    ],
                    stop_reason=StopReason.TOOL_USE,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        messages, _ = _convert_context_to_openai(ctx)
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["function"]["name"] == "read_file"

    def test_tool_result_message(self) -> None:
        ctx = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="tc_1",
                    tool_name="read_file",
                    content=[TextContent(text="file contents")],
                    timestamp=0,
                )
            ]
        )
        messages, _ = _convert_context_to_openai(ctx)
        msg = messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc_1"
        assert msg["content"] == "file contents"

    def test_empty_assistant_skipped(self) -> None:
        ctx = Context(
            messages=[
                AssistantMessage(
                    content=[],
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(),
                    timestamp=0,
                )
            ]
        )
        messages, _ = _convert_context_to_openai(ctx)
        assert len(messages) == 0


class TestConvertTools:
    """Tests for tool schema conversion."""

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
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"


class TestMapFinishReason:
    """Tests for finish reason mapping."""

    def test_stop(self) -> None:
        assert _map_finish_reason("stop") == StopReason.END_TURN

    def test_length(self) -> None:
        assert _map_finish_reason("length") == StopReason.MAX_TOKENS

    def test_tool_calls(self) -> None:
        assert _map_finish_reason("tool_calls") == StopReason.TOOL_USE

    def test_content_filter(self) -> None:
        assert _map_finish_reason("content_filter") == StopReason.ERROR

    def test_none(self) -> None:
        assert _map_finish_reason(None) == StopReason.END_TURN


class TestOpenAIProviderInit:
    """Tests for OpenAIProvider initialization."""

    def test_default_init(self) -> None:
        provider = OpenAIProvider()
        assert provider.model == "gpt-4o"

    def test_custom_model(self) -> None:
        provider = OpenAIProvider(model="gpt-4o-mini")
        assert provider.model == "gpt-4o-mini"

    def test_import_error_when_no_sdk(self) -> None:
        provider = OpenAIProvider()
        with (
            patch.dict("sys.modules", {"openai": None}),
            pytest.raises(ImportError, match="openai"),
        ):
            provider._get_client()


class TestOpenAIProviderStream:
    """Tests for OpenAI streaming."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """Test streaming a simple text response."""
        provider = OpenAIProvider(model="gpt-4o")

        chunks = [
            _mock_chunk(content="Hello"),
            _mock_chunk(content=" world"),
            _mock_chunk(finish_reason="stop"),
            _mock_chunk(usage={"prompt_tokens": 10, "completion_tokens": 5}),
        ]

        async def mock_chunks():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        events = []
        async for event in provider.stream(
            Context(
                system_prompt="test",
                messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
            )
        ):
            events.append(event)

        # Should have: start, text_delta (Hello), text_delta ( world), text_end, done
        event_types = [e.type for e in events]
        assert "start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

        # Check final message
        done_event = [e for e in events if e.type == "done"][0]
        assert done_event.message.stop_reason == StopReason.END_TURN

    @pytest.mark.asyncio
    async def test_tool_call_response(self) -> None:
        """Test streaming a tool call response."""
        provider = OpenAIProvider(model="gpt-4o")

        chunks = [
            _mock_chunk(
                tool_call_id="call_1",
                tool_call_name="read_file",
                tool_call_args='{"pa',
            ),
            _mock_chunk(tool_call_args='th": "/tmp/a"}'),
            _mock_chunk(finish_reason="tool_calls"),
            _mock_chunk(usage={"prompt_tokens": 10, "completion_tokens": 15}),
        ]

        async def mock_chunks():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        events = []
        ctx = Context(
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
        async for event in provider.stream(ctx):
            events.append(event)

        event_types = [e.type for e in events]
        assert "tool_call_start" in event_types
        assert "tool_call_delta" in event_types
        assert "tool_call_end" in event_types
        assert "done" in event_types

        # Check tool call end event
        tc_end = [e for e in events if e.type == "tool_call_end"][0]
        assert tc_end.tool_name == "read_file"
        assert tc_end.tool_call_id == "call_1"

    @pytest.mark.asyncio
    async def test_abort_signal(self) -> None:
        """Test that abort signal stops streaming."""
        provider = OpenAIProvider(model="gpt-4o")
        signal = asyncio.Event()

        chunks = [
            _mock_chunk(content="Hello"),
        ]

        async def mock_chunks():
            for c in chunks:
                yield c
            # Set signal after first chunk
            signal.set()
            yield _mock_chunk(content=" more")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        events = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)]),
            signal=signal,
        ):
            events.append(event)

        # Should end with error event (aborted)
        last_event = events[-1]
        assert last_event.type == "error"

    @pytest.mark.asyncio
    async def test_api_error_yields_error_event(self) -> None:
        """Test that API errors yield StreamErrorEvent."""
        provider = OpenAIProvider(model="gpt-4o")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API key invalid"))
        provider._client = mock_client

        events = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "API key invalid" in (events[0].error.error_message or "")

    @pytest.mark.asyncio
    async def test_usage_extraction(self) -> None:
        """Test that usage is extracted from the final chunk."""
        provider = OpenAIProvider(model="gpt-4o")

        chunks = [
            _mock_chunk(content="hi"),
            _mock_chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 20},
            ),
        ]

        async def mock_chunks():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        events = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)])
        ):
            events.append(event)

        done_event = [e for e in events if e.type == "done"][0]
        assert done_event.message.usage.input_tokens == 100
        assert done_event.message.usage.output_tokens == 50
        assert done_event.message.usage.cache_read_tokens == 20

    @pytest.mark.asyncio
    async def test_reasoning_content(self) -> None:
        """Test handling of reasoning content from o-series models."""
        provider = OpenAIProvider(model="o3")

        chunks = [
            _mock_chunk(reasoning_content="Let me think..."),
            _mock_chunk(content="The answer is 42"),
            _mock_chunk(finish_reason="stop"),
            _mock_chunk(usage={"prompt_tokens": 10, "completion_tokens": 20}),
        ]

        async def mock_chunks():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        events = []
        async for event in provider.stream(
            Context(messages=[UserMessage(content=[TextContent(text="think")], timestamp=0)])
        ):
            events.append(event)

        event_types = [e.type for e in events]
        assert "thinking_delta" in event_types
        assert "text_delta" in event_types
