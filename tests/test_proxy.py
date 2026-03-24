"""Tests for ProxyProvider."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from isotope_core.providers.proxy import ProxyProvider
from isotope_core.types import (
    Context,
    StopReason,
    TextContent,
    ToolSchema,
    UserMessage,
)

# =============================================================================
# Helpers
# =============================================================================


def _mock_chunk(
    *,
    content: str | None = None,
    tool_call_id: str | None = None,
    tool_call_name: str | None = None,
    tool_call_args: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock OpenAI streaming chunk."""
    chunk = MagicMock()
    chunk.id = "chatcmpl-proxy"
    chunk.usage = None

    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
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


def _simple_context() -> Context:
    return Context(
        system_prompt="test",
        messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
    )


# =============================================================================
# Tests
# =============================================================================


class TestProxyProviderInit:
    """Tests for ProxyProvider initialization."""

    def test_init_with_base_url(self) -> None:
        provider = ProxyProvider(model="llama3", base_url="http://localhost:11434/v1")
        assert provider.model == "llama3"
        assert provider._base_url == "http://localhost:11434/v1"

    def test_provider_name(self) -> None:
        provider = ProxyProvider(model="test", base_url="http://localhost/v1")
        assert provider.provider_name == "proxy"

    def test_model_name(self) -> None:
        provider = ProxyProvider(model="llama3", base_url="http://localhost/v1")
        assert provider.model_name == "llama3"

    def test_custom_headers(self) -> None:
        headers = {"X-Custom": "value"}
        provider = ProxyProvider(
            model="test",
            base_url="http://localhost/v1",
            default_headers=headers,
        )
        assert provider._default_headers == headers

    def test_auth_token(self) -> None:
        provider = ProxyProvider(
            model="test",
            base_url="http://localhost/v1",
            api_key="secret-token",
        )
        assert provider._api_key == "secret-token"


class TestProxyProviderStream:
    """Tests for ProxyProvider streaming."""

    @pytest.mark.asyncio
    async def test_basic_text_streaming(self) -> None:
        """Test basic text streaming through proxy."""
        provider = ProxyProvider(model="llama3", base_url="http://localhost:11434/v1")

        chunks = [
            _mock_chunk(content="Hello"),
            _mock_chunk(content=" from proxy"),
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
        async for event in provider.stream(_simple_context()):
            events.append(event)

        event_types = [e.type for e in events]
        assert "start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

        done_event = [e for e in events if e.type == "done"][0]
        assert done_event.message.stop_reason == StopReason.END_TURN

    @pytest.mark.asyncio
    async def test_tool_call_streaming(self) -> None:
        """Test tool call streaming through proxy."""
        provider = ProxyProvider(model="test", base_url="http://localhost/v1")

        chunks = [
            _mock_chunk(
                tool_call_id="call_1",
                tool_call_name="search",
                tool_call_args='{"query":',
            ),
            _mock_chunk(tool_call_args=' "test"}'),
            _mock_chunk(finish_reason="tool_calls"),
            _mock_chunk(usage={"prompt_tokens": 15, "completion_tokens": 10}),
        ]

        async def mock_chunks():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        ctx = Context(
            messages=[UserMessage(content=[TextContent(text="search")], timestamp=0)],
            tools=[
                ToolSchema(
                    name="search",
                    description="Search",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
            ],
        )

        events = []
        async for event in provider.stream(ctx):
            events.append(event)

        event_types = [e.type for e in events]
        assert "tool_call_start" in event_types
        assert "tool_call_end" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_usage_extraction(self) -> None:
        """Test usage is extracted from proxy response."""
        provider = ProxyProvider(model="test", base_url="http://localhost/v1")

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
        async for event in provider.stream(_simple_context()):
            events.append(event)

        done_event = [e for e in events if e.type == "done"][0]
        assert done_event.message.usage.input_tokens == 100
        assert done_event.message.usage.output_tokens == 50
        assert done_event.message.usage.cache_read_tokens == 20

    @pytest.mark.asyncio
    async def test_api_error_yields_error_event(self) -> None:
        """Test that API errors yield StreamErrorEvent."""
        provider = ProxyProvider(model="test", base_url="http://localhost/v1")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        provider._client = mock_client

        events = []
        async for event in provider.stream(_simple_context()):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "Connection refused" in (events[0].error.error_message or "")

    @pytest.mark.asyncio
    async def test_abort_signal(self) -> None:
        """Test that abort signal stops streaming."""
        provider = ProxyProvider(model="test", base_url="http://localhost/v1")
        signal = asyncio.Event()

        async def mock_chunks():
            yield _mock_chunk(content="Hello")
            signal.set()
            yield _mock_chunk(content=" more")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        events = []
        async for event in provider.stream(_simple_context(), signal=signal):
            events.append(event)

        last_event = events[-1]
        assert last_event.type == "error"


class TestProxyProviderApiKeyResolver:
    """Tests for API key resolver in ProxyProvider."""

    @pytest.mark.asyncio
    async def test_api_key_resolver_called(self) -> None:
        """Test that api_key_resolver is called on each stream."""
        call_count = 0

        async def resolver() -> str:
            nonlocal call_count
            call_count += 1
            return f"key-{call_count}"

        provider = ProxyProvider(
            model="test",
            base_url="http://localhost/v1",
            api_key_resolver=resolver,
        )

        chunks = [
            _mock_chunk(content="ok"),
            _mock_chunk(finish_reason="stop"),
            _mock_chunk(usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ]

        async def mock_chunks():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        async for _ in provider.stream(_simple_context()):
            pass

        assert call_count == 1
        assert mock_client.api_key == "key-1"
