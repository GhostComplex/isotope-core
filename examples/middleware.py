"""Middleware example — custom middleware for the agent loop.

This example demonstrates:
- LoggingMiddleware for event logging
- TokenTrackingMiddleware for usage tracking
- EventFilterMiddleware for filtering events
- A custom middleware implementation

Uses a mock provider so no API key is needed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

from isotopo_core import (
    Agent,
    AgentEvent,
    EventFilterMiddleware,
    LoggingMiddleware,
    MiddlewareContext,
    TokenTrackingMiddleware,
)
from isotopo_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDeltaEvent,
)
from isotopo_core.types import (
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    Usage,
)

# =============================================================================
# Mock provider
# =============================================================================


class MockProvider:
    """Mock provider for middleware demos."""

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def provider_name(self) -> str:
        return "mock"

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        ts = int(time.time() * 1000)
        msg = AssistantMessage(
            content=[TextContent(text="Hello from the mock provider!")],
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=15, output_tokens=8),
            timestamp=ts,
        )
        yield StreamStartEvent(partial=msg)
        yield StreamTextDeltaEvent(
            content_index=0, delta="Hello from the mock provider!", partial=msg,
        )
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Custom middleware
# =============================================================================


class TimingMiddleware:
    """Custom middleware that measures event processing time."""

    def __init__(self) -> None:
        self.event_times: list[tuple[str, float]] = []
        self._start_time: float | None = None

    async def on_event(
        self,
        event: AgentEvent,
        context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None:
        if event.type == "agent_start":
            self._start_time = time.monotonic()

        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        self.event_times.append((event.type, elapsed))

        return await next(event)


class MessageCounterMiddleware:
    """Custom middleware that counts messages by role."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def on_event(
        self,
        event: AgentEvent,
        context: MiddlewareContext,
        next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
    ) -> AgentEvent | None:
        if event.type == "message_end":
            role = getattr(event.message, "role", "unknown")  # type: ignore[union-attr]
            self.counts[role] = self.counts.get(role, 0) + 1

        return await next(event)


# =============================================================================
# Demos
# =============================================================================


async def demo_logging_middleware() -> None:
    """Demonstrate LoggingMiddleware."""
    print("=== LoggingMiddleware (minimal level) ===\n")

    logs: list[str] = []
    logging_mw = LoggingMiddleware(logger=logs.append, log_level="minimal")

    agent = Agent(
        provider=MockProvider(),
        system_prompt="You are helpful.",
        middleware=[logging_mw],
    )

    async for _ in agent.prompt("Hi"):
        pass

    for log in logs:
        print(f"  {log}")


async def demo_token_tracking() -> None:
    """Demonstrate TokenTrackingMiddleware."""
    print("\n=== TokenTrackingMiddleware ===\n")

    tracker = TokenTrackingMiddleware()

    agent = Agent(
        provider=MockProvider(),
        system_prompt="You are helpful.",
        middleware=[tracker],
    )

    async for _ in agent.prompt("Hi"):
        pass

    print(f"  Total usage: {tracker.total_usage}")
    print(f"  Turn count: {tracker.turn_count}")
    print(f"  Per-turn usage: {tracker.per_turn_usage}")


async def demo_event_filter() -> None:
    """Demonstrate EventFilterMiddleware."""
    print("\n=== EventFilterMiddleware ===\n")

    # Filter out message_update events (only keep lifecycle events)
    filter_mw = EventFilterMiddleware(exclude={"message_update"})

    agent = Agent(
        provider=MockProvider(),
        system_prompt="You are helpful.",
        middleware=[filter_mw],
    )

    event_types: list[str] = []
    async for event in agent.prompt("Hi"):
        event_types.append(event.type)

    print(f"  Events received: {event_types}")
    print(f"  message_update filtered: {'message_update' not in event_types}")


async def demo_custom_middleware() -> None:
    """Demonstrate custom middleware."""
    print("\n=== Custom Middleware (Timing + Counter) ===\n")

    timing = TimingMiddleware()
    counter = MessageCounterMiddleware()

    agent = Agent(
        provider=MockProvider(),
        system_prompt="You are helpful.",
        middleware=[timing, counter],
    )

    async for _ in agent.prompt("Hi"):
        pass

    print("  Event timing:")
    for event_type, elapsed in timing.event_times:
        print(f"    {event_type}: {elapsed:.4f}s")

    print(f"\n  Message counts: {counter.counts}")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    await demo_logging_middleware()
    await demo_token_tracking()
    await demo_event_filter()
    await demo_custom_middleware()


if __name__ == "__main__":
    asyncio.run(main())
