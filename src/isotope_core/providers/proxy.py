"""Proxy provider for OpenAI-compatible endpoints.

This module provides a provider for any API that follows the OpenAI
Chat Completions format (LiteLLM, vLLM, TGI, Ollama, Azure OpenAI, etc.).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING

from isotope_core.providers.base import StreamEvent
from isotope_core.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from isotope_core.types import Context


class ProxyProvider(OpenAIProvider):
    """OpenAI-compatible proxy provider.

    Wraps OpenAIProvider with a required base_url for connecting to any
    endpoint that implements the OpenAI Chat Completions API format.

    Use cases:
    - LiteLLM proxy
    - vLLM / TGI endpoints
    - Ollama (with OpenAI-compat mode)
    - Azure OpenAI
    - Any OpenAI-compatible endpoint
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
        api_key_resolver: Callable[[], Awaitable[str]] | None = None,
    ):
        """Initialize the proxy provider.

        Args:
            model: The model to use.
            base_url: Base URL for the OpenAI-compatible endpoint
                (e.g., "http://localhost:11434/v1").
            api_key: API key for authentication (some proxies require it).
            default_headers: Additional headers to include in requests.
            timeout: Request timeout in seconds.
            api_key_resolver: Optional async callable that returns a fresh API key.
                If provided, called before each stream() to get the current key.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            api_key_resolver=api_key_resolver,
        )
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "proxy"

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from the proxy endpoint.

        Delegates to OpenAIProvider.stream() since the proxy uses the same
        OpenAI Chat Completions format.

        Args:
            context: The conversation context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            signal: An asyncio.Event that, when set, signals abortion.

        Yields:
            StreamEvent: Events describing the streaming response.
        """
        async for event in super().stream(
            context,
            temperature=temperature,
            max_tokens=max_tokens,
            signal=signal,
        ):
            yield event
