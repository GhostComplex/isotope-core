"""Tests for EventFilterMiddleware — exclude event types, passthrough."""

from __future__ import annotations

import time

import pytest

from isotope_core.middleware import EventFilterMiddleware, MiddlewareContext, run_middleware_chain
from isotope_core.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StopReason,
    TextContent,
    ToolEndEvent,
    ToolStartEvent,
    ToolUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)


def _ctx() -> MiddlewareContext:
    return MiddlewareContext(messages=[], turn_number=1, cumulative_tokens=0, agent_config=None)


def _msg() -> UserMessage:
    return UserMessage(content=[TextContent(text="Hello")], timestamp=int(time.time() * 1000))


def _assistant_msg() -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="Hi")],
        stop_reason=StopReason.END_TURN,
        timestamp=int(time.time() * 1000),
    )


# =============================================================================
# Tests
# =============================================================================


class TestEventFilterExclusion:
    """Test that excluded event types are filtered."""

    @pytest.mark.asyncio
    async def test_excludes_specified_types(self) -> None:
        mw = EventFilterMiddleware(exclude={"message_update", "tool_update"})
        ctx = _ctx()

        result1 = await run_middleware_chain(
            MessageUpdateEvent(message=_msg()), ctx, [mw]
        )
        result2 = await run_middleware_chain(
            ToolUpdateEvent(tool_call_id="1", tool_name="test", args={}), ctx, [mw]
        )

        assert result1 is None
        assert result2 is None

    @pytest.mark.asyncio
    async def test_allows_non_excluded_types(self) -> None:
        mw = EventFilterMiddleware(exclude={"message_update"})
        ctx = _ctx()

        result = await run_middleware_chain(AgentStartEvent(), ctx, [mw])
        assert result is not None
        assert result.type == "agent_start"

    @pytest.mark.asyncio
    async def test_exclude_single_type(self) -> None:
        mw = EventFilterMiddleware(exclude={"agent_start"})
        ctx = _ctx()

        result = await run_middleware_chain(AgentStartEvent(), ctx, [mw])
        assert result is None

        result2 = await run_middleware_chain(AgentEndEvent(), ctx, [mw])
        assert result2 is not None


class TestEventFilterPassthrough:
    """Test that all events pass when no exclusions."""

    @pytest.mark.asyncio
    async def test_no_exclusion_passes_all(self) -> None:
        mw = EventFilterMiddleware(exclude=set())
        ctx = _ctx()

        events: list[AgentEvent] = [
            AgentStartEvent(),
            TurnStartEvent(),
            MessageStartEvent(message=_msg()),
            MessageUpdateEvent(message=_msg()),
            MessageEndEvent(message=_msg()),
            ToolStartEvent(tool_call_id="1", tool_name="test", args={}),
            ToolUpdateEvent(tool_call_id="1", tool_name="test", args={}),
            ToolEndEvent(tool_call_id="1", tool_name="test", result={}),
            TurnEndEvent(message=_assistant_msg(), tool_results=[]),
            AgentEndEvent(),
        ]

        for event in events:
            result = await run_middleware_chain(event, ctx, [mw])
            assert result is not None, f"Event {event.type} should pass through"

    @pytest.mark.asyncio
    async def test_none_exclude_passes_all(self) -> None:
        """EventFilterMiddleware with None exclusion passes all events."""
        mw = EventFilterMiddleware(exclude=None)
        ctx = _ctx()

        result = await run_middleware_chain(AgentStartEvent(), ctx, [mw])
        assert result is not None

    @pytest.mark.asyncio
    async def test_default_constructor_passes_all(self) -> None:
        """Default EventFilterMiddleware passes all events."""
        mw = EventFilterMiddleware()
        ctx = _ctx()

        result = await run_middleware_chain(AgentStartEvent(), ctx, [mw])
        assert result is not None
