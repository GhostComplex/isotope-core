"""Router provider with fallback chains and circuit breaker.

This module provides a provider that wraps multiple providers with routing
logic including fallback chains and circuit breaker patterns.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import Any

from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
)
from isotope_core.providers.utils import is_retryable_error
from isotope_core.types import AggregatedUsage, Context, Usage


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Healthy — requests pass through
    OPEN = "open"  # Tripped — requests are skipped
    HALF_OPEN = "half_open"  # Testing — one request allowed through


class _CircuitBreaker:
    """Per-provider circuit breaker.

    Tracks consecutive failures and transitions between states:
    - CLOSED: normal operation, requests pass through
    - OPEN: after threshold consecutive failures, skip this provider
    - HALF_OPEN: after timeout expires, allow one test request
    """

    def __init__(self, threshold: int, timeout: float) -> None:
        self.threshold = threshold
        self.timeout = timeout
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self._opened_at: float = 0.0

    def is_available(self) -> bool:
        """Check if the provider is available for requests."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if time.monotonic() - self._opened_at >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN — allow one request
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1
        if self.state == CircuitState.HALF_OPEN:
            # Test request failed — reopen
            self.state = CircuitState.OPEN
            self._opened_at = time.monotonic()
        elif self.consecutive_failures >= self.threshold:
            self.state = CircuitState.OPEN
            self._opened_at = time.monotonic()


class RouterProvider:
    """Provider that routes requests across multiple providers with fallback.

    Features:
    - Fallback chains: if primary fails with retryable error, try fallbacks in order
    - Circuit breaker: after N consecutive failures, skip a provider for a timeout period
    - Usage aggregation: track usage across all providers
    - Dynamic switching: change primary provider mid-session

    Non-retryable errors are re-raised immediately without trying fallbacks.
    """

    def __init__(
        self,
        primary: Any,  # Provider protocol
        fallbacks: list[Any] | None = None,
        health_check_interval: float = 60.0,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_timeout: float = 120.0,
    ) -> None:
        """Initialize the router provider.

        Args:
            primary: The primary provider to use.
            fallbacks: Ordered list of fallback providers.
            health_check_interval: Interval between health checks in seconds.
            circuit_breaker_threshold: Consecutive failures before tripping circuit.
            circuit_breaker_timeout: Seconds to wait before retrying a tripped provider.
        """
        self._primary = primary
        self._fallbacks: list[Any] = fallbacks or []
        self._health_check_interval = health_check_interval
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_timeout = circuit_breaker_timeout

        # Circuit breakers keyed by id(provider)
        self._breakers: dict[int, _CircuitBreaker] = {}
        self._ensure_breaker(primary)
        for fb in self._fallbacks:
            self._ensure_breaker(fb)

        # Aggregated usage
        self._usage = AggregatedUsage()

        # Lock for thread-safe circuit breaker updates
        self._lock = asyncio.Lock()

    def _ensure_breaker(self, provider: Any) -> _CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        pid = id(provider)
        if pid not in self._breakers:
            self._breakers[pid] = _CircuitBreaker(
                threshold=self._circuit_breaker_threshold,
                timeout=self._circuit_breaker_timeout,
            )
        return self._breakers[pid]

    @property
    def model_name(self) -> str:
        """Return the model of the current primary provider."""
        return str(self._primary.model_name)

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "router"

    def set_primary(self, provider: Any) -> None:
        """Switch the primary provider.

        Args:
            provider: The new primary provider.
        """
        self._primary = provider
        self._ensure_breaker(provider)

    def get_usage(self) -> AggregatedUsage:
        """Return aggregated usage across all providers."""
        return self._usage.model_copy(deep=True)

    def _accumulate_usage(self, provider: Any, usage: Usage) -> None:
        """Add usage from a provider to the aggregated totals."""
        self._usage.total_input_tokens += usage.input_tokens
        self._usage.total_output_tokens += usage.output_tokens
        self._usage.total_cache_read_tokens += usage.cache_read_tokens
        self._usage.total_cache_write_tokens += usage.cache_write_tokens

        # Per-provider usage
        pname = str(getattr(provider, "provider_name", "unknown"))
        if pname not in self._usage.provider_usage:
            self._usage.provider_usage[pname] = Usage()
        pu = self._usage.provider_usage[pname]
        pu.input_tokens += usage.input_tokens
        pu.output_tokens += usage.output_tokens
        pu.cache_read_tokens += usage.cache_read_tokens
        pu.cache_write_tokens += usage.cache_write_tokens

        # Per-model usage
        mname = str(getattr(provider, "model_name", "unknown"))
        if mname not in self._usage.model_usage:
            self._usage.model_usage[mname] = Usage()
        mu = self._usage.model_usage[mname]
        mu.input_tokens += usage.input_tokens
        mu.output_tokens += usage.output_tokens
        mu.cache_read_tokens += usage.cache_read_tokens
        mu.cache_write_tokens += usage.cache_write_tokens

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response, using fallbacks on retryable errors.

        The router tries the primary provider first. If it fails with a
        retryable error, it tries each fallback in order. Non-retryable
        errors are re-raised immediately.

        Args:
            context: The conversation context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            signal: An asyncio.Event that, when set, signals abortion.

        Yields:
            StreamEvent: Events describing the streaming response.
        """
        providers = [self._primary, *self._fallbacks]
        last_error: Exception | None = None

        for provider in providers:
            breaker = self._ensure_breaker(provider)

            async with self._lock:
                available = breaker.is_available()

            if not available:
                continue

            try:
                async for event in self._stream_with_tracking(
                    provider,
                    context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    signal=signal,
                ):
                    yield event
                # If we get here, stream completed (done or error event emitted)
                return
            except Exception as e:
                last_error = e

                async with self._lock:
                    breaker.record_failure()

                # Non-retryable errors must not fall through to fallbacks
                if not is_retryable_error(e):
                    raise

                # Retryable error — try next provider
                continue

        # All providers exhausted
        if last_error is not None:
            raise last_error

    async def _stream_with_tracking(
        self,
        provider: Any,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream from a provider with circuit breaker and usage tracking.

        This wraps the provider's stream and intercepts the final event
        to track usage and update circuit breaker state. If the stream
        yields a StreamErrorEvent as its final event with a retryable error,
        it raises an exception so the router can try the next provider.
        """
        events: list[StreamEvent] = []

        async for event in provider.stream(
            context,
            temperature=temperature,
            max_tokens=max_tokens,
            signal=signal,
        ):
            events.append(event)
            yield event

        # Process the final event for circuit breaker and usage
        if events:
            last_event = events[-1]

            if isinstance(last_event, StreamDoneEvent):
                async with self._lock:
                    self._ensure_breaker(provider).record_success()
                # Accumulate usage from the done message
                self._accumulate_usage(provider, last_event.message.usage)

            elif isinstance(last_event, StreamErrorEvent):
                # Check if the error is retryable
                error_msg = last_event.error.error_message or ""
                # Create a synthetic exception to check retryability
                synthetic_error = Exception(error_msg)
                if is_retryable_error(synthetic_error):
                    async with self._lock:
                        self._ensure_breaker(provider).record_failure()
                    # Raise so router tries next provider
                    raise Exception(error_msg)  # noqa: TRY002
                else:
                    # Non-retryable stream error — record failure but don't retry
                    async with self._lock:
                        self._ensure_breaker(provider).record_failure()
