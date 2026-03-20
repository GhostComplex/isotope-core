"""Tests for provider utilities."""

from __future__ import annotations

import pytest

from isotopo_core.providers.utils import (
    RetryConfig,
    create_error_message,
    current_timestamp_ms,
    get_retry_after,
    is_retryable_error,
    map_error_to_stop_reason,
    retry_with_backoff,
)
from isotopo_core.types import StopReason, Usage


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self) -> None:
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert 429 in config.retryable_status_codes
        assert 500 in config.retryable_status_codes

    def test_custom_config(self) -> None:
        config = RetryConfig(max_retries=5, initial_delay=0.5, jitter=False)
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.jitter is False


class TestIsRetryableError:
    """Tests for is_retryable_error."""

    def test_retryable_status_code(self) -> None:
        class MockError(Exception):
            status_code = 429

        assert is_retryable_error(MockError()) is True

    def test_retryable_status_code_500(self) -> None:
        class MockError(Exception):
            status_code = 500

        assert is_retryable_error(MockError()) is True

    def test_non_retryable_status_code(self) -> None:
        class MockError(Exception):
            status_code = 400

        assert is_retryable_error(MockError()) is False

    def test_retryable_by_message(self) -> None:
        assert is_retryable_error(Exception("rate limit exceeded")) is True
        assert is_retryable_error(Exception("service unavailable")) is True
        assert is_retryable_error(Exception("connection timeout")) is True

    def test_non_retryable_error(self) -> None:
        assert is_retryable_error(Exception("invalid api key")) is False

    def test_retryable_from_response(self) -> None:
        class MockResponse:
            status_code = 503

        class MockError(Exception):
            response = MockResponse()

        assert is_retryable_error(MockError()) is True


class TestGetRetryAfter:
    """Tests for get_retry_after."""

    def test_retry_after_attribute(self) -> None:
        class MockError(Exception):
            retry_after = 30

        assert get_retry_after(MockError()) == 30.0

    def test_retry_after_header(self) -> None:
        class MockResponse:
            headers = {"Retry-After": "60"}

        class MockError(Exception):
            response = MockResponse()

        assert get_retry_after(MockError()) == 60.0

    def test_no_retry_after(self) -> None:
        assert get_retry_after(Exception("oops")) is None


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3))
        async def succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeeds()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        call_count = 0

        class RetryableError(Exception):
            status_code = 429

        @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.01, jitter=False))
        async def fails_then_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("rate limited")
            return "ok"

        result = await fails_then_succeeds()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.01))
        async def fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await fails()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        call_count = 0

        class RetryableError(Exception):
            status_code = 500

        @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=0.01, jitter=False))
        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise RetryableError("server error")

        with pytest.raises(RetryableError):
            await always_fails()
        assert call_count == 3  # initial + 2 retries


class TestMapErrorToStopReason:
    """Tests for map_error_to_stop_reason."""

    def test_abort_error(self) -> None:
        assert map_error_to_stop_reason(Exception("aborted")) == StopReason.ABORTED

    def test_cancel_error(self) -> None:
        assert map_error_to_stop_reason(Exception("cancelled")) == StopReason.ABORTED

    def test_generic_error(self) -> None:
        assert map_error_to_stop_reason(Exception("something broke")) == StopReason.ERROR


class TestCreateErrorMessage:
    """Tests for create_error_message."""

    def test_basic_error(self) -> None:
        msg = create_error_message(Exception("test error"), 1000)
        assert msg.error_message == "test error"
        assert msg.stop_reason == StopReason.ERROR
        assert msg.timestamp == 1000

    def test_error_with_status_code(self) -> None:
        class HttpError(Exception):
            status_code = 500

        msg = create_error_message(HttpError("internal server error"), 2000)
        assert "500" in (msg.error_message or "")

    def test_error_with_usage(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        msg = create_error_message(Exception("err"), 3000, usage=usage)
        assert msg.usage.input_tokens == 100


class TestCurrentTimestampMs:
    """Tests for current_timestamp_ms."""

    def test_returns_milliseconds(self) -> None:
        ts = current_timestamp_ms()
        assert isinstance(ts, int)
        assert ts > 1_000_000_000_000  # After year 2001 in ms
