"""Multi-provider example — router with fallback.

This example demonstrates:
- RouterProvider with primary + fallback providers
- Circuit breaker behavior
- Aggregated usage tracking across providers

Uses mock providers so no API key is needed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

from isotopo_core import Agent, RouterProvider
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
# Mock providers
# =============================================================================


class ReliableProvider:
    """A provider that always succeeds."""

    def __init__(self, name: str, model: str) -> None:
        self._name = name
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

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
        ts = int(time.time() * 1000)
        msg = AssistantMessage(
            content=[TextContent(text=f"Response from {self._name} ({self._model})")],
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=10, output_tokens=8),
            timestamp=ts,
        )
        yield StreamStartEvent(partial=msg)
        yield StreamTextDeltaEvent(
            content_index=0,
            delta=f"Response from {self._name} ({self._model})",
            partial=msg,
        )
        yield StreamDoneEvent(message=msg)


class UnreliableProvider:
    """A provider that fails with retryable errors."""

    def __init__(self, name: str, model: str) -> None:
        self._name = name
        self._model = model
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model

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
        # Always fail with a retryable error
        raise Exception("Service unavailable — rate limit exceeded")
        yield  # pragma: no cover


# =============================================================================
# Demos
# =============================================================================


async def demo_basic_routing() -> None:
    """Demonstrate basic routing with a reliable primary provider."""
    print("=== Basic Router (reliable primary) ===\n")

    primary = ReliableProvider("primary-openai", "gpt-4o")
    fallback = ReliableProvider("fallback-anthropic", "claude-sonnet-4-20250514")

    router = RouterProvider(
        primary=primary,
        fallbacks=[fallback],
    )

    agent = Agent(provider=router, system_prompt="You are helpful.")

    async for event in agent.prompt("Hello"):
        if event.type == "message_update" and event.delta:  # type: ignore[union-attr]
            print(f"  {event.delta}")  # type: ignore[union-attr]

    usage = router.get_usage()
    print(f"\n  Total input tokens: {usage.total_input_tokens}")
    print(f"  Total output tokens: {usage.total_output_tokens}")
    print(f"  Provider usage: {dict(usage.provider_usage)}")


async def demo_fallback() -> None:
    """Demonstrate fallback when primary fails."""
    print("\n=== Router Fallback (unreliable primary → reliable fallback) ===\n")

    primary = UnreliableProvider("broken-primary", "gpt-4o")
    fallback = ReliableProvider("fallback-anthropic", "claude-sonnet-4-20250514")

    router = RouterProvider(
        primary=primary,
        fallbacks=[fallback],
        circuit_breaker_threshold=2,
    )

    agent = Agent(provider=router, system_prompt="You are helpful.")

    async for event in agent.prompt("Hello"):
        if event.type == "message_update" and event.delta:  # type: ignore[union-attr]
            print(f"  {event.delta}")  # type: ignore[union-attr]
        elif event.type == "agent_end":
            print(f"  [agent_end] reason={event.reason}")  # type: ignore[union-attr]

    usage = router.get_usage()
    print(f"\n  Provider usage: {dict(usage.provider_usage)}")
    print(f"  Primary called {primary.call_count} times (all failed)")


async def demo_switch_primary() -> None:
    """Demonstrate dynamically switching the primary provider."""
    print("\n=== Dynamic Provider Switching ===\n")

    provider_a = ReliableProvider("provider-a", "model-a")
    provider_b = ReliableProvider("provider-b", "model-b")

    router = RouterProvider(primary=provider_a)

    # First request
    agent = Agent(provider=router, system_prompt="You are helpful.")

    async for event in agent.prompt("First request"):
        if event.type == "message_update" and event.delta:  # type: ignore[union-attr]
            print(f"  Request 1: {event.delta}")  # type: ignore[union-attr]

    # Switch primary provider
    print("  [switching primary to provider-b]")
    router.set_primary(provider_b)
    agent.reset()

    async for event in agent.prompt("Second request"):
        if event.type == "message_update" and event.delta:  # type: ignore[union-attr]
            print(f"  Request 2: {event.delta}")  # type: ignore[union-attr]

    usage = router.get_usage()
    print(f"\n  Model usage: {dict(usage.model_usage)}")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    await demo_basic_routing()
    await demo_fallback()
    await demo_switch_primary()


if __name__ == "__main__":
    asyncio.run(main())
