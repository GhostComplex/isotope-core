"""Shared utilities for LLM providers.

This module provides retry logic, error mapping, and other utilities
shared across provider implementations.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from isotope_core.types import AssistantMessage, StopReason, TextContent, Usage

# =============================================================================
# Type Variables
# =============================================================================

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Retry Configuration
# =============================================================================


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts.
            initial_delay: Initial delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.
            exponential_base: Base for exponential backoff.
            jitter: Whether to add random jitter to delays.
            retryable_status_codes: HTTP status codes that trigger a retry.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes


DEFAULT_RETRY_CONFIG = RetryConfig()


# =============================================================================
# Error Classification
# =============================================================================


def is_retryable_error(error: Exception, config: RetryConfig = DEFAULT_RETRY_CONFIG) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception to check.
        config: Retry configuration.

    Returns:
        True if the error should trigger a retry.
    """
    # Check for status_code attribute (common in SDK errors)
    status_code = getattr(error, "status_code", None)
    if status_code is not None and status_code in config.retryable_status_codes:
        return True

    # Check for response attribute with status_code
    response = getattr(error, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status is not None and resp_status in config.retryable_status_codes:
            return True

    # Check for common transient error messages
    error_str = str(error).lower()
    transient_patterns = [
        "rate limit",
        "too many requests",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "connection reset",
        "connection refused",
        "timeout",
    ]

    return any(pattern in error_str for pattern in transient_patterns)


def get_retry_after(error: Exception) -> float | None:
    """Extract Retry-After value from error if present.

    Args:
        error: The exception to check.

    Returns:
        Retry-After value in seconds, or None if not present.
    """
    # Check for retry_after attribute (Anthropic SDK)
    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        return float(retry_after)

    # Check for response headers
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        retry_after_header = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after_header:
            try:
                return float(retry_after_header)
            except (ValueError, TypeError):
                pass

    return None


# =============================================================================
# Retry Decorator
# =============================================================================


def retry_with_backoff(
    config: RetryConfig = DEFAULT_RETRY_CONFIG,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for async functions with exponential backoff retry.

    Args:
        config: Retry configuration.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Don't retry on last attempt
                    if attempt >= config.max_retries:
                        raise

                    # Don't retry non-retryable errors
                    if not is_retryable_error(e, config):
                        raise

                    # Calculate delay
                    delay = min(
                        config.initial_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )

                    # Check for Retry-After header
                    retry_after = get_retry_after(e)
                    if retry_after is not None:
                        delay = min(retry_after, config.max_delay)

                    # Add jitter
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    await asyncio.sleep(delay)

            # Should never reach here, but satisfy type checker
            if last_error is not None:
                raise last_error
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


# =============================================================================
# Error Mapping
# =============================================================================


def map_error_to_stop_reason(error: Exception) -> StopReason:
    """Map an exception to an appropriate StopReason.

    Args:
        error: The exception to map.

    Returns:
        The appropriate StopReason for the error.
    """
    # Check for abort-related errors
    error_str = str(error).lower()
    if any(pattern in error_str for pattern in ["abort", "cancel", "cancelled", "interrupted"]):
        return StopReason.ABORTED

    return StopReason.ERROR


def get_error_status_code(error: Exception) -> int | None:
    """Extract HTTP status code from an error if available.

    Args:
        error: The exception to check.

    Returns:
        HTTP status code if available, None otherwise.
    """
    # Check direct status_code attribute
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return int(status_code)

    # Check response attribute
    response = getattr(error, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status is not None:
            return int(resp_status)

    return None


# =============================================================================
# Error Message Creation
# =============================================================================


def create_error_message(
    error: Exception,
    timestamp: int,
    usage: Usage | None = None,
) -> AssistantMessage:
    """Create an AssistantMessage representing an error.

    Args:
        error: The exception that occurred.
        timestamp: Unix timestamp in milliseconds.
        usage: Optional usage information if available.

    Returns:
        An AssistantMessage with error information.
    """
    error_text = str(error)
    status_code = get_error_status_code(error)

    if status_code is not None:
        error_text = f"HTTP {status_code}: {error_text}"

    return AssistantMessage(
        content=[TextContent(text=error_text)],
        stop_reason=map_error_to_stop_reason(error),
        usage=usage or Usage(),
        error_message=error_text,
        timestamp=timestamp,
    )


# =============================================================================
# Timestamp Utility
# =============================================================================


def current_timestamp_ms() -> int:
    """Get current Unix timestamp in milliseconds.

    Returns:
        Current timestamp in milliseconds.
    """
    import time

    return int(time.time() * 1000)
