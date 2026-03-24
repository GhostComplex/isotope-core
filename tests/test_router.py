"""Tests for RouterProvider with fallback chains and circuit breaker."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest

from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
)
from isotope_core.providers.router import CircuitState, RouterProvider, _CircuitBreaker
from isotope_core.types import (
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    Usage,
    UserMessage,
)

# =============================================================================
# Helpers — fake providers for testing
# =============================================================================


class FakeProvider:
    """A configurable fake provider for testing router behavior."""

    def __init__(
        self,
        name: str = "fake",
        model: str = "fake-model",
        responses: list[str] | None = None,
        error: Exception | None = None,
        usage: Usage | None = None,
    ) -> None:
        self._name = name
        self.model = model
        self._responses = responses or ["Hello"]
        self._error = error
        self._usage = usage or Usage(input_tokens=10, output_tokens=5)
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider_name(self) -> str:
        return self._name

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.call_count += 1

        if self._error is not None:
            raise self._error

        timestamp = int(time.time() * 1000)
        output = AssistantMessage(
            content=[],
            stop_reason=None,
            usage=self._usage,
            timestamp=timestamp,
        )

        yield StreamStartEvent(partial=output)

        for text in self._responses:
            block = TextContent(text=text)
            output.content.append(block)
            yield StreamTextDeltaEvent(
                content_index=len(output.content) - 1,
                delta=text,
                partial=output,
            )
            yield StreamTextEndEvent(
                content_index=len(output.content) - 1,
                content=text,
                partial=output,
            )

        output.stop_reason = StopReason.END_TURN
        yield StreamDoneEvent(message=output)


class FakeStreamErrorProvider:
    """Provider that yields a StreamErrorEvent instead of raising."""

    def __init__(
        self,
        name: str = "error-stream",
        model: str = "error-model",
        error_message: str = "something went wrong",
        retryable: bool = False,
    ) -> None:
        self._name = name
        self.model = model
        self._error_message = error_message
        self._retryable = retryable
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider_name(self) -> str:
        return self._name

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.call_count += 1
        timestamp = int(time.time() * 1000)

        # Make the error message contain retryable patterns if needed
        msg = self._error_message
        if self._retryable:
            msg = f"rate limit: {msg}"

        output = AssistantMessage(
            content=[TextContent(text=msg)],
            stop_reason=StopReason.ERROR,
            usage=Usage(),
            error_message=msg,
            timestamp=timestamp,
        )
        yield StreamErrorEvent(error=output)


def _simple_context() -> Context:
    """Create a simple context for testing."""
    return Context(
        system_prompt="test",
        messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
    )


# =============================================================================
# Circuit Breaker Unit Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for the _CircuitBreaker class."""

    def test_initial_state_is_closed(self) -> None:
        cb = _CircuitBreaker(threshold=3, timeout=60.0)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_stays_closed_under_threshold(self) -> None:
        cb = _CircuitBreaker(threshold=3, timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_trips_at_threshold(self) -> None:
        cb = _CircuitBreaker(threshold=3, timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_available() is False

    def test_success_resets_failures(self) -> None:
        cb = _CircuitBreaker(threshold=3, timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.consecutive_failures == 0
        assert cb.state == CircuitState.CLOSED

    def test_open_becomes_half_open_after_timeout(self) -> None:
        cb = _CircuitBreaker(threshold=1, timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.02)
        assert cb.is_available() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self) -> None:
        cb = _CircuitBreaker(threshold=1, timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.is_available()  # transitions to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_half_open_failure_reopens(self) -> None:
        cb = _CircuitBreaker(threshold=1, timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.is_available()  # transitions to HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN


# =============================================================================
# Router Provider Tests
# =============================================================================


class TestRouterProviderBasic:
    """Basic router provider tests."""

    def test_provider_info(self) -> None:
        primary = FakeProvider(name="openai", model="gpt-4o")
        router = RouterProvider(primary=primary)
        assert router.provider_name == "router"
        assert router.model_name == "gpt-4o"

    def test_set_primary(self) -> None:
        primary = FakeProvider(name="openai", model="gpt-4o")
        fallback = FakeProvider(name="anthropic", model="claude")
        router = RouterProvider(primary=primary, fallbacks=[fallback])

        router.set_primary(fallback)
        assert router.model_name == "claude"


class TestRouterProviderStream:
    """Tests for router streaming with fallbacks."""

    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self) -> None:
        """When primary succeeds, fallback is never called."""
        primary = FakeProvider(name="primary", responses=["Hello"])
        fallback = FakeProvider(name="fallback", responses=["Fallback"])
        router = RouterProvider(primary=primary, fallbacks=[fallback])

        events: list[StreamEvent] = []
        async for event in router.stream(_simple_context()):
            events.append(event)

        assert primary.call_count == 1
        assert fallback.call_count == 0

        event_types = [e.type for e in events]
        assert "start" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_primary_fails_retryable_uses_fallback(self) -> None:
        """When primary fails with retryable error, fallback is used."""

        class RetryableError(Exception):
            status_code = 429

        primary = FakeProvider(name="primary", error=RetryableError("rate limited"))
        fallback = FakeProvider(name="fallback", responses=["Fallback response"])
        router = RouterProvider(primary=primary, fallbacks=[fallback])

        events: list[StreamEvent] = []
        async for event in router.stream(_simple_context()):
            events.append(event)

        assert primary.call_count == 1
        assert fallback.call_count == 1

        event_types = [e.type for e in events]
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_primary_fails_non_retryable_no_fallback(self) -> None:
        """Non-retryable errors are re-raised immediately."""
        primary = FakeProvider(name="primary", error=ValueError("invalid input"))
        fallback = FakeProvider(name="fallback", responses=["Fallback"])
        router = RouterProvider(primary=primary, fallbacks=[fallback])

        with pytest.raises(ValueError, match="invalid input"):
            async for _ in router.stream(_simple_context()):
                pass

        assert primary.call_count == 1
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_multiple_fallbacks_in_order(self) -> None:
        """Fallbacks are tried in order."""

        class RetryableError(Exception):
            status_code = 500

        primary = FakeProvider(name="p", error=RetryableError("down"))
        fb1 = FakeProvider(name="fb1", error=RetryableError("also down"))
        fb2 = FakeProvider(name="fb2", responses=["Finally"])
        router = RouterProvider(primary=primary, fallbacks=[fb1, fb2])

        events: list[StreamEvent] = []
        async for event in router.stream(_simple_context()):
            events.append(event)

        assert primary.call_count == 1
        assert fb1.call_count == 1
        assert fb2.call_count == 1
        assert any(e.type == "done" for e in events)

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self) -> None:
        """When all providers fail, the last error is raised."""

        class RetryableError(Exception):
            status_code = 503

        primary = FakeProvider(name="p", error=RetryableError("down 1"))
        fb = FakeProvider(name="fb", error=RetryableError("down 2"))
        router = RouterProvider(primary=primary, fallbacks=[fb])

        with pytest.raises(RetryableError):
            async for _ in router.stream(_simple_context()):
                pass


class TestRouterCircuitBreaker:
    """Tests for circuit breaker integration in the router."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_after_threshold(self) -> None:
        """After threshold failures, provider is skipped."""

        class RetryableError(Exception):
            status_code = 429

        primary = FakeProvider(name="primary", error=RetryableError("rate limited"))
        fallback = FakeProvider(name="fallback", responses=["OK"])
        router = RouterProvider(
            primary=primary,
            fallbacks=[fallback],
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=60.0,
        )

        # First call: primary fails, fallback succeeds
        events = []
        async for event in router.stream(_simple_context()):
            events.append(event)
        assert primary.call_count == 1

        # Second call: primary fails again, trips the breaker
        events = []
        async for event in router.stream(_simple_context()):
            events.append(event)
        assert primary.call_count == 2

        # Third call: primary should be skipped (circuit open)
        events = []
        async for event in router.stream(_simple_context()):
            events.append(event)
        assert primary.call_count == 2  # Not called again
        assert fallback.call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_success_closes(self) -> None:
        """Half-open state: success closes the circuit."""
        call_count = 0

        class RetryableError(Exception):
            status_code = 500

        class RecoveringProvider:
            """Provider that fails twice then succeeds."""

            @property
            def model_name(self) -> str:
                return "recovering"

            @property
            def provider_name(self) -> str:
                return "recovering"

            async def stream(
                self,
                context: Context,
                *,
                temperature: float | None = None,
                max_tokens: int | None = None,
                signal: asyncio.Event | None = None,
            ) -> AsyncGenerator[StreamEvent, None]:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise RetryableError("down")

                timestamp = int(time.time() * 1000)
                output = AssistantMessage(
                    content=[TextContent(text="recovered")],
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(input_tokens=5, output_tokens=3),
                    timestamp=timestamp,
                )
                yield StreamStartEvent(partial=output)
                yield StreamDoneEvent(message=output)

        primary = RecoveringProvider()
        fallback = FakeProvider(name="fallback", responses=["fb"])
        router = RouterProvider(
            primary=primary,
            fallbacks=[fallback],
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=0.01,  # Very short for testing
        )

        # Trip the breaker
        for _ in range(2):
            async for _ in router.stream(_simple_context()):
                pass

        assert call_count == 2

        # Wait for timeout
        await asyncio.sleep(0.02)

        # Next call should test half_open (provider now succeeds)
        events = []
        async for event in router.stream(_simple_context()):
            events.append(event)

        assert call_count == 3
        assert any(e.type == "done" for e in events)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure_reopens(self) -> None:
        """Half-open state: failure reopens the circuit."""

        class RetryableError(Exception):
            status_code = 500

        primary = FakeProvider(name="primary", error=RetryableError("always down"))
        fallback = FakeProvider(name="fallback", responses=["fb"])
        router = RouterProvider(
            primary=primary,
            fallbacks=[fallback],
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.01,
        )

        # Trip the breaker
        async for _ in router.stream(_simple_context()):
            pass
        assert primary.call_count == 1

        # Wait for timeout (becomes half_open)
        await asyncio.sleep(0.02)

        # Test request in half_open — still fails, reopens
        async for _ in router.stream(_simple_context()):
            pass
        assert primary.call_count == 2  # Tried once more

        # Should be open again — won't be tried
        async for _ in router.stream(_simple_context()):
            pass
        assert primary.call_count == 2  # Skipped


class TestRouterDynamicSwitching:
    """Tests for dynamic primary switching."""

    @pytest.mark.asyncio
    async def test_set_primary_switches_provider(self) -> None:
        """set_primary changes which provider is used."""
        p1 = FakeProvider(name="p1", model="model-1", responses=["from p1"])
        p2 = FakeProvider(name="p2", model="model-2", responses=["from p2"])
        router = RouterProvider(primary=p1)

        # First call uses p1
        async for _ in router.stream(_simple_context()):
            pass
        assert p1.call_count == 1
        assert p2.call_count == 0

        # Switch to p2
        router.set_primary(p2)
        assert router.model_name == "model-2"

        async for _ in router.stream(_simple_context()):
            pass
        assert p1.call_count == 1
        assert p2.call_count == 1


class TestRouterUsageAggregation:
    """Tests for usage aggregation across providers."""

    @pytest.mark.asyncio
    async def test_usage_accumulated_from_single_provider(self) -> None:
        """Usage from a single provider is accumulated."""
        primary = FakeProvider(
            name="openai",
            model="gpt-4o",
            usage=Usage(input_tokens=100, output_tokens=50, cache_read_tokens=10),
        )
        router = RouterProvider(primary=primary)

        async for _ in router.stream(_simple_context()):
            pass

        agg = router.get_usage()
        assert agg.total_input_tokens == 100
        assert agg.total_output_tokens == 50
        assert agg.total_cache_read_tokens == 10

    @pytest.mark.asyncio
    async def test_usage_accumulated_across_providers(self) -> None:
        """Usage is accumulated when fallback is used."""

        class RetryableError(Exception):
            status_code = 429

        primary = FakeProvider(
            name="openai",
            model="gpt-4o",
            error=RetryableError("rate limited"),
        )
        fallback = FakeProvider(
            name="anthropic",
            model="claude",
            usage=Usage(input_tokens=80, output_tokens=30, cache_read_tokens=5),
        )
        router = RouterProvider(primary=primary, fallbacks=[fallback])

        async for _ in router.stream(_simple_context()):
            pass

        agg = router.get_usage()
        # Only fallback usage should be present (primary raised, no done event)
        assert agg.total_input_tokens == 80
        assert agg.total_output_tokens == 30

    @pytest.mark.asyncio
    async def test_usage_per_provider_and_model(self) -> None:
        """Per-provider and per-model usage are tracked."""
        primary = FakeProvider(
            name="openai",
            model="gpt-4o",
            usage=Usage(input_tokens=100, output_tokens=50),
        )
        router = RouterProvider(primary=primary)

        async for _ in router.stream(_simple_context()):
            pass

        agg = router.get_usage()
        assert "openai" in agg.provider_usage
        assert agg.provider_usage["openai"].input_tokens == 100
        assert "gpt-4o" in agg.model_usage
        assert agg.model_usage["gpt-4o"].output_tokens == 50

    @pytest.mark.asyncio
    async def test_usage_accumulates_over_multiple_calls(self) -> None:
        """Usage accumulates across multiple stream calls."""
        primary = FakeProvider(
            name="openai",
            model="gpt-4o",
            usage=Usage(input_tokens=50, output_tokens=25),
        )
        router = RouterProvider(primary=primary)

        # Two calls
        async for _ in router.stream(_simple_context()):
            pass
        async for _ in router.stream(_simple_context()):
            pass

        agg = router.get_usage()
        assert agg.total_input_tokens == 100
        assert agg.total_output_tokens == 50

    @pytest.mark.asyncio
    async def test_get_usage_returns_copy(self) -> None:
        """get_usage returns a copy, not a reference."""
        primary = FakeProvider(
            name="openai",
            model="gpt-4o",
            usage=Usage(input_tokens=50, output_tokens=25),
        )
        router = RouterProvider(primary=primary)

        async for _ in router.stream(_simple_context()):
            pass

        agg1 = router.get_usage()
        agg2 = router.get_usage()
        assert agg1 is not agg2
        assert agg1.total_input_tokens == agg2.total_input_tokens
