"""Tests for dynamic API key resolution across providers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from isotope_core.providers.openai import OpenAIProvider
from isotope_core.types import (
    Context,
    TextContent,
    UserMessage,
)

# =============================================================================
# Helpers
# =============================================================================


def _mock_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock OpenAI streaming chunk."""
    chunk = MagicMock()
    chunk.id = "chatcmpl-resolver"
    chunk.usage = None

    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    delta.reasoning_content = None
    delta.reasoning = None

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


def _mock_stream_chunks() -> list[MagicMock]:
    return [
        _mock_chunk(content="ok"),
        _mock_chunk(finish_reason="stop"),
        _mock_chunk(usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ]


# =============================================================================
# Tests
# =============================================================================


class TestApiKeyResolverOpenAI:
    """Tests for dynamic API key resolution in OpenAIProvider."""

    @pytest.mark.asyncio
    async def test_resolver_called_each_stream(self) -> None:
        """Resolver is called before each stream() invocation."""
        call_count = 0

        async def resolver() -> str:
            nonlocal call_count
            call_count += 1
            return f"key-{call_count}"

        provider = OpenAIProvider(model="gpt-4o", api_key_resolver=resolver)

        async def mock_chunks():
            for c in _mock_stream_chunks():
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        # First call
        async for _ in provider.stream(_simple_context()):
            pass
        assert call_count == 1
        assert mock_client.api_key == "key-1"

        # Second call — resolver should be called again
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        async for _ in provider.stream(_simple_context()):
            pass
        assert call_count == 2
        assert mock_client.api_key == "key-2"

    @pytest.mark.asyncio
    async def test_resolver_error_yields_error_event(self) -> None:
        """If the resolver raises, an error event is yielded."""

        async def bad_resolver() -> str:
            raise RuntimeError("Token refresh failed")

        provider = OpenAIProvider(model="gpt-4o", api_key_resolver=bad_resolver)
        mock_client = AsyncMock()
        provider._client = mock_client

        events = []
        async for event in provider.stream(_simple_context()):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "Token refresh failed" in (events[0].error.error_message or "")

    @pytest.mark.asyncio
    async def test_key_rotation_during_session(self) -> None:
        """Keys rotate correctly across multiple stream calls."""
        keys = ["key-alpha", "key-beta", "key-gamma"]
        index = 0

        async def rotating_resolver() -> str:
            nonlocal index
            key = keys[index % len(keys)]
            index += 1
            return key

        provider = OpenAIProvider(model="gpt-4o", api_key_resolver=rotating_resolver)

        mock_client = AsyncMock()
        provider._client = mock_client

        resolved_keys: list[str] = []

        for _ in range(3):

            async def mock_chunks():
                for c in _mock_stream_chunks():
                    yield c

            mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())

            async for _ in provider.stream(_simple_context()):
                pass

            resolved_keys.append(mock_client.api_key)

        assert resolved_keys == ["key-alpha", "key-beta", "key-gamma"]

    @pytest.mark.asyncio
    async def test_no_resolver_uses_static_key(self) -> None:
        """Without resolver, static api_key is used (backward compatible)."""
        provider = OpenAIProvider(model="gpt-4o", api_key="static-key")

        async def mock_chunks():
            for c in _mock_stream_chunks():
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        async for _ in provider.stream(_simple_context()):
            pass

        # api_key should NOT have been overwritten
        # The static key is passed at client construction time
        # and the resolver path should not be entered
        assert provider._api_key == "static-key"

    @pytest.mark.asyncio
    async def test_resolver_with_none_api_key(self) -> None:
        """Resolver works even when no static api_key is provided."""

        async def resolver() -> str:
            return "dynamic-key"

        provider = OpenAIProvider(model="gpt-4o", api_key_resolver=resolver)

        async def mock_chunks():
            for c in _mock_stream_chunks():
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chunks())
        provider._client = mock_client

        async for _ in provider.stream(_simple_context()):
            pass

        assert mock_client.api_key == "dynamic-key"
