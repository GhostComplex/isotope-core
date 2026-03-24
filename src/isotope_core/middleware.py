"""Middleware system for the agent loop.

This module provides a composable middleware/plugin system that intercepts
events flowing through the agent loop, enabling extensibility without
modifying core code.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from isotope_core.types import (
    AgentEvent,
    AssistantMessage,
    Message,
    MessageEndEvent,
    Usage,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Middleware Protocol & Context
# =============================================================================


@dataclass
class MiddlewareContext:
    """Context available to middleware."""

    messages: list[Message]
    turn_number: int
    cumulative_tokens: int
    agent_config: Any  # AgentLoopConfig (avoid circular import)


@runtime_checkable
class Middleware(Protocol):
    """Protocol for middleware that intercepts agent loop events.

    Middleware forms a chain — each calls ``next()`` to pass the event
    to the next middleware. Middleware can modify events before passing
    through, or suppress events by returning ``None`` (not calling ``next()``).

    Order matters: first added = outermost (sees events first, results last).
    """

    async def on_event(
        self,
        event: AgentEvent,
        context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None:
        """Process an event.

        Args:
            event: The agent event to process.
            context: Current middleware context with state info.
            next: Call to pass the event to the next middleware in the chain.

        Returns:
            The (possibly modified) event, or ``None`` to suppress it.
        """
        ...


# =============================================================================
# Middleware Chain Runner
# =============================================================================


async def run_middleware_chain(
    event: AgentEvent,
    context: MiddlewareContext,
    middleware: list[Any],
) -> AgentEvent | None:
    """Run an event through the middleware chain.

    The chain is built such that the first middleware in the list is the
    outermost (sees the event first). The innermost ``next()`` returns the
    event unchanged.

    If any middleware raises an exception, the error is logged and the event
    passes through unchanged.

    Args:
        event: The event to process.
        context: Current middleware context.
        middleware: Ordered list of middleware instances.

    Returns:
        The processed event, or ``None`` if suppressed by middleware.
    """
    if not middleware:
        return event

    # Build the chain from inside out.
    # The innermost next simply returns the event unchanged.
    async def innermost(evt: AgentEvent) -> AgentEvent | None:
        return evt

    # Wrap from the last middleware to the first, so that middleware[0]
    # is the outermost (called first).
    chain: Callable[[AgentEvent], Awaitable[AgentEvent | None]] = innermost
    for mw in reversed(middleware):

        # Capture mw and current chain via default args
        def _make_next(
            _mw: Any,
            _next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
        ) -> Callable[[AgentEvent], Awaitable[AgentEvent | None]]:
            async def _handler(evt: AgentEvent) -> AgentEvent | None:
                try:
                    result: AgentEvent | None = await _mw.on_event(evt, context, _next)
                    return result
                except Exception:
                    logger.exception(
                        "Middleware %s raised an error; passing event through",
                        type(_mw).__name__,
                    )
                    return evt

            return _handler

        chain = _make_next(mw, chain)

    return await chain(event)


# =============================================================================
# Built-in Middleware: LoggingMiddleware
# =============================================================================

# Event types per log level
_MINIMAL_EVENTS = frozenset({"agent_start", "agent_end", "turn_start", "turn_end"})
_NORMAL_EVENTS = _MINIMAL_EVENTS | frozenset(
    {"message_start", "message_end", "tool_start", "tool_end"}
)
_VERBOSE_EVENTS = _NORMAL_EVENTS | frozenset({"message_update", "tool_update"})

_LEVEL_MAP: dict[str, frozenset[str]] = {
    "minimal": _MINIMAL_EVENTS,
    "normal": _NORMAL_EVENTS,
    "verbose": _VERBOSE_EVENTS,
}


class LoggingMiddleware:
    """Logs all agent events with configurable verbosity.

    Args:
        logger: Optional custom logging callable. Defaults to ``print``.
        log_level: One of ``"minimal"``, ``"normal"``, or ``"verbose"``.
        include_content: Whether to include event content in log output.
    """

    def __init__(
        self,
        logger: Callable[[str], None] | None = None,
        log_level: Literal["minimal", "normal", "verbose"] = "normal",
        include_content: bool = False,
    ) -> None:
        self._logger = logger or print
        self._log_level = log_level
        self._include_content = include_content
        self._allowed_events = _LEVEL_MAP[log_level]

    async def on_event(
        self,
        event: AgentEvent,
        context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None:
        """Log the event if it matches the configured log level, then pass through."""
        if event.type in self._allowed_events:
            if self._include_content:
                self._logger(f"[{event.type}] {event.model_dump_json()}")
            else:
                self._logger(f"[{event.type}]")

        return await next(event)


# =============================================================================
# Built-in Middleware: TokenTrackingMiddleware
# =============================================================================


class TokenTrackingMiddleware:
    """Tracks detailed token usage across turns.

    Intercepts ``MessageEndEvent`` for assistant messages and accumulates
    token usage.
    """

    def __init__(self) -> None:
        self._total_usage = Usage()
        self._per_turn_usage: list[Usage] = []
        self._turn_count = 0

    @property
    def total_usage(self) -> Usage:
        """Total accumulated token usage."""
        return self._total_usage

    @property
    def per_turn_usage(self) -> list[Usage]:
        """Per-turn token usage list."""
        return list(self._per_turn_usage)

    @property
    def turn_count(self) -> int:
        """Number of turns tracked."""
        return self._turn_count

    async def on_event(
        self,
        event: AgentEvent,
        context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None:
        """Track token usage from MessageEndEvent for assistant messages."""
        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            usage = event.message.usage
            self._total_usage = Usage(
                input_tokens=self._total_usage.input_tokens + usage.input_tokens,
                output_tokens=self._total_usage.output_tokens + usage.output_tokens,
                cache_read_tokens=self._total_usage.cache_read_tokens + usage.cache_read_tokens,
                cache_write_tokens=self._total_usage.cache_write_tokens + usage.cache_write_tokens,
            )
            self._per_turn_usage.append(usage)
            self._turn_count += 1

        return await next(event)


# =============================================================================
# Built-in Middleware: EventFilterMiddleware
# =============================================================================


class EventFilterMiddleware:
    """Filters out specific event types.

    Events whose ``type`` is in the ``exclude`` set are suppressed
    (not passed to the next middleware or yielded to the caller).

    Args:
        exclude: Set of event type strings to filter out.
    """

    def __init__(self, exclude: set[str] | None = None) -> None:
        self._exclude: set[str] = exclude or set()

    async def on_event(
        self,
        event: AgentEvent,
        context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None:
        """Suppress excluded event types; pass all others through."""
        if event.type in self._exclude:
            return None
        return await next(event)


# =============================================================================
# Lifecycle Hook Types
# =============================================================================

OnAgentStartHook = Callable[[], Awaitable[None]]
OnAgentEndHook = Callable[[str], Awaitable[None]]
OnTurnStartHook = Callable[[int], Awaitable[None]]
OnTurnEndHook = Callable[[int, AssistantMessage], Awaitable[None]]
OnErrorHook = Callable[[Exception], Awaitable[None]]


@dataclass
class LifecycleHooks:
    """Collection of lifecycle hooks for the agent loop."""

    on_agent_start: OnAgentStartHook | None = None
    on_agent_end: OnAgentEndHook | None = None
    on_turn_start: OnTurnStartHook | None = None
    on_turn_end: OnTurnEndHook | None = None
    on_error: OnErrorHook | None = None


__all__ = [
    "EventFilterMiddleware",
    "LifecycleHooks",
    "LoggingMiddleware",
    "Middleware",
    "MiddlewareContext",
    "OnAgentEndHook",
    "OnAgentStartHook",
    "OnErrorHook",
    "OnTurnEndHook",
    "OnTurnStartHook",
    "TokenTrackingMiddleware",
    "run_middleware_chain",
]
