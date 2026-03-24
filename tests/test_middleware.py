"""Tests for isotope_core.middleware — chain ordering, passthrough, suppression, modification."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import pytest

from isotope_core.loop import AgentLoopConfig, agent_loop
from isotope_core.middleware import (
    MiddlewareContext,
    run_middleware_chain,
)
from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
)
from isotope_core.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    UserMessage,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_context(config: Any | None = None) -> MiddlewareContext:
    return MiddlewareContext(
        messages=[],
        turn_number=1,
        cumulative_tokens=0,
        agent_config=config,
    )


class MockProvider:
    """Minimal mock provider for integration tests."""

    def __init__(self, responses: list[AssistantMessage] | None = None) -> None:
        self.responses = responses or []
        self.call_count = 0

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self.call_count >= len(self.responses):
            msg = AssistantMessage(
                content=[TextContent(text="Done")],
                stop_reason=StopReason.END_TURN,
                timestamp=int(time.time() * 1000),
            )
        else:
            msg = self.responses[self.call_count]
        self.call_count += 1

        yield StreamStartEvent(partial=msg)
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Tests: Middleware Chain (unit)
# =============================================================================


class TestMiddlewareChain:
    """Unit tests for run_middleware_chain."""

    @pytest.mark.asyncio
    async def test_single_middleware_passthrough(self) -> None:
        """A single middleware that just calls next() passes the event through."""

        class PassthroughMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                return await next(event)

        event = AgentStartEvent()
        result = await run_middleware_chain(event, _make_context(), [PassthroughMW()])
        assert result is not None
        assert result.type == "agent_start"

    @pytest.mark.asyncio
    async def test_chain_ordering(self) -> None:
        """First added middleware is outermost — sees event first."""
        order: list[str] = []

        class MW:
            def __init__(self, name: str) -> None:
                self._name = name

            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                order.append(f"{self._name}_before")
                result = await next(event)
                order.append(f"{self._name}_after")
                return result

        event = AgentStartEvent()
        await run_middleware_chain(event, _make_context(), [MW("A"), MW("B"), MW("C")])

        assert order == [
            "A_before",
            "B_before",
            "C_before",
            "C_after",
            "B_after",
            "A_after",
        ]

    @pytest.mark.asyncio
    async def test_event_modification(self) -> None:
        """Middleware can modify the event."""

        class ModifyMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                if isinstance(event, AgentEndEvent):
                    event = AgentEndEvent(messages=event.messages, reason="modified")
                return await next(event)

        event = AgentEndEvent(messages=[], reason="completed")
        result = await run_middleware_chain(event, _make_context(), [ModifyMW()])
        assert result is not None
        assert isinstance(result, AgentEndEvent)
        assert result.reason == "modified"

    @pytest.mark.asyncio
    async def test_event_suppression(self) -> None:
        """Middleware can suppress events by returning None."""

        class SuppressMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                if event.type == "agent_start":
                    return None
                return await next(event)

        event = AgentStartEvent()
        result = await run_middleware_chain(event, _make_context(), [SuppressMW()])
        assert result is None

    @pytest.mark.asyncio
    async def test_context_has_correct_data(self) -> None:
        """MiddlewareContext is passed with correct data."""
        captured_ctx: list[MiddlewareContext] = []

        class CaptureMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                captured_ctx.append(context)
                return await next(event)

        msg = UserMessage(
            content=[TextContent(text="Hello")],
            timestamp=int(time.time() * 1000),
        )
        ctx = MiddlewareContext(
            messages=[msg],
            turn_number=3,
            cumulative_tokens=500,
            agent_config=None,
        )

        await run_middleware_chain(AgentStartEvent(), ctx, [CaptureMW()])

        assert len(captured_ctx) == 1
        assert captured_ctx[0].turn_number == 3
        assert captured_ctx[0].cumulative_tokens == 500
        assert len(captured_ctx[0].messages) == 1

    @pytest.mark.asyncio
    async def test_middleware_error_passes_event_through(self) -> None:
        """If middleware raises, the event passes through unchanged."""

        class ErrorMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                raise RuntimeError("Middleware exploded")

        event = AgentStartEvent()
        result = await run_middleware_chain(event, _make_context(), [ErrorMW()])
        assert result is not None
        assert result.type == "agent_start"

    @pytest.mark.asyncio
    async def test_empty_middleware_list_passes_through(self) -> None:
        """Empty middleware list returns event unchanged."""
        event = AgentStartEvent()
        result = await run_middleware_chain(event, _make_context(), [])
        assert result is event


# =============================================================================
# Tests: Middleware Integration with Agent Loop
# =============================================================================


class TestMiddlewareIntegration:
    """Integration tests for middleware with the full agent loop."""

    @pytest.mark.asyncio
    async def test_middleware_sees_all_events(self) -> None:
        """Middleware receives every event yielded by the loop."""
        seen_types: list[str] = []

        class RecordMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                seen_types.append(event.type)
                return await next(event)

        response = AssistantMessage(
            content=[TextContent(text="Hello!")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        provider = MockProvider([response])
        config = AgentLoopConfig(provider=provider, middleware=[RecordMW()])

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        context = Context(system_prompt="Be helpful.")

        async for _ in agent_loop([prompt], context, config):
            pass

        assert "agent_start" in seen_types
        assert "agent_end" in seen_types
        assert "turn_start" in seen_types
        assert "turn_end" in seen_types
        assert "message_start" in seen_types
        assert "message_end" in seen_types

    @pytest.mark.asyncio
    async def test_middleware_suppresses_events_in_loop(self) -> None:
        """Events suppressed by middleware are not yielded to the caller."""

        class SuppressUpdatesMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                if event.type == "message_start":
                    return None
                return await next(event)

        response = AssistantMessage(
            content=[TextContent(text="Hello!")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        provider = MockProvider([response])
        config = AgentLoopConfig(provider=provider, middleware=[SuppressUpdatesMW()])

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "message_start" not in event_types
        # Other events should still be present
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_middleware_error_doesnt_break_loop(self) -> None:
        """Middleware errors don't crash the loop — events pass through."""

        class ExplodingMW:
            async def on_event(
                self,
                event: AgentEvent,
                context: MiddlewareContext,
                next: Callable[[AgentEvent], Awaitable[AgentEvent | None]],
            ) -> AgentEvent | None:
                raise RuntimeError("BOOM")

        response = AssistantMessage(
            content=[TextContent(text="Hello!")],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        provider = MockProvider([response])
        config = AgentLoopConfig(provider=provider, middleware=[ExplodingMW()])

        prompt = UserMessage(
            content=[TextContent(text="Hi")],
            timestamp=int(time.time() * 1000),
        )
        context = Context()

        events = []
        async for event in agent_loop([prompt], context, config):
            events.append(event)

        event_types = [e.type for e in events]
        # Loop should still complete successfully despite middleware errors
        assert "agent_start" in event_types
        assert "agent_end" in event_types
