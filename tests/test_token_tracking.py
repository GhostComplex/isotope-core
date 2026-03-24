"""Tests for TokenTrackingMiddleware — usage accumulation, per-turn tracking, turn count."""

from __future__ import annotations

import time

import pytest

from isotope_core.middleware import MiddlewareContext, TokenTrackingMiddleware, run_middleware_chain
from isotope_core.types import (
    AgentStartEvent,
    AssistantMessage,
    MessageEndEvent,
    StopReason,
    TextContent,
    Usage,
    UserMessage,
)


def _ctx() -> MiddlewareContext:
    return MiddlewareContext(messages=[], turn_number=1, cumulative_tokens=0, agent_config=None)


def _assistant_msg(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="Hello")],
        stop_reason=StopReason.END_TURN,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
        timestamp=int(time.time() * 1000),
    )


def _user_msg() -> UserMessage:
    return UserMessage(
        content=[TextContent(text="Hi")],
        timestamp=int(time.time() * 1000),
    )


# =============================================================================
# Tests
# =============================================================================


class TestTokenTrackingUsage:
    """Test total usage accumulation."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        mw = TokenTrackingMiddleware()
        assert mw.total_usage.input_tokens == 0
        assert mw.total_usage.output_tokens == 0
        assert mw.turn_count == 0
        assert mw.per_turn_usage == []

    @pytest.mark.asyncio
    async def test_single_message_usage(self) -> None:
        mw = TokenTrackingMiddleware()
        msg = _assistant_msg(input_tokens=100, output_tokens=50)
        event = MessageEndEvent(message=msg)

        await run_middleware_chain(event, _ctx(), [mw])

        assert mw.total_usage.input_tokens == 100
        assert mw.total_usage.output_tokens == 50
        assert mw.total_usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_accumulates_across_messages(self) -> None:
        mw = TokenTrackingMiddleware()
        ctx = _ctx()

        msg1 = _assistant_msg(input_tokens=100, output_tokens=50)
        msg2 = _assistant_msg(input_tokens=200, output_tokens=75)

        await run_middleware_chain(MessageEndEvent(message=msg1), ctx, [mw])
        await run_middleware_chain(MessageEndEvent(message=msg2), ctx, [mw])

        assert mw.total_usage.input_tokens == 300
        assert mw.total_usage.output_tokens == 125
        assert mw.total_usage.total_tokens == 425

    @pytest.mark.asyncio
    async def test_cache_tokens_tracked(self) -> None:
        mw = TokenTrackingMiddleware()
        msg = _assistant_msg(
            input_tokens=100, output_tokens=50, cache_read=30, cache_write=20
        )
        await run_middleware_chain(MessageEndEvent(message=msg), _ctx(), [mw])

        assert mw.total_usage.cache_read_tokens == 30
        assert mw.total_usage.cache_write_tokens == 20


class TestTokenTrackingPerTurn:
    """Test per-turn usage tracking."""

    @pytest.mark.asyncio
    async def test_per_turn_list(self) -> None:
        mw = TokenTrackingMiddleware()
        ctx = _ctx()

        msg1 = _assistant_msg(input_tokens=100, output_tokens=50)
        msg2 = _assistant_msg(input_tokens=200, output_tokens=75)

        await run_middleware_chain(MessageEndEvent(message=msg1), ctx, [mw])
        await run_middleware_chain(MessageEndEvent(message=msg2), ctx, [mw])

        per_turn = mw.per_turn_usage
        assert len(per_turn) == 2
        assert per_turn[0].input_tokens == 100
        assert per_turn[0].output_tokens == 50
        assert per_turn[1].input_tokens == 200
        assert per_turn[1].output_tokens == 75

    @pytest.mark.asyncio
    async def test_per_turn_returns_copy(self) -> None:
        """per_turn_usage returns a copy, not the internal list."""
        mw = TokenTrackingMiddleware()
        list1 = mw.per_turn_usage
        list1.append(Usage(input_tokens=999))
        assert len(mw.per_turn_usage) == 0


class TestTokenTrackingTurnCount:
    """Test turn count accuracy."""

    @pytest.mark.asyncio
    async def test_turn_count_increments(self) -> None:
        mw = TokenTrackingMiddleware()
        ctx = _ctx()

        for i in range(5):
            msg = _assistant_msg(input_tokens=i * 10, output_tokens=i * 5)
            await run_middleware_chain(MessageEndEvent(message=msg), ctx, [mw])

        assert mw.turn_count == 5

    @pytest.mark.asyncio
    async def test_ignores_non_assistant_messages(self) -> None:
        """Token tracking only counts assistant messages, not user messages."""
        mw = TokenTrackingMiddleware()
        ctx = _ctx()

        user = _user_msg()
        await run_middleware_chain(MessageEndEvent(message=user), ctx, [mw])

        assert mw.turn_count == 0
        assert mw.total_usage.input_tokens == 0

    @pytest.mark.asyncio
    async def test_ignores_non_message_end_events(self) -> None:
        """Token tracking only acts on MessageEndEvent."""
        mw = TokenTrackingMiddleware()
        ctx = _ctx()

        await run_middleware_chain(AgentStartEvent(), ctx, [mw])

        assert mw.turn_count == 0

    @pytest.mark.asyncio
    async def test_passthrough_behavior(self) -> None:
        """TokenTrackingMiddleware always passes events through."""
        mw = TokenTrackingMiddleware()
        event = MessageEndEvent(message=_assistant_msg(input_tokens=10, output_tokens=5))
        result = await run_middleware_chain(event, _ctx(), [mw])
        assert result is not None
        assert result.type == "message_end"
