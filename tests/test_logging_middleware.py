"""Tests for LoggingMiddleware — log levels, custom logger, content inclusion."""

from __future__ import annotations

import time

import pytest

from isotope_core.middleware import LoggingMiddleware, MiddlewareContext, run_middleware_chain
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


class TestLoggingMiddlewareMinimal:
    """Test minimal log level — only agent_start, agent_end, turn_start, turn_end."""

    @pytest.mark.asyncio
    async def test_minimal_logs_lifecycle_events(self) -> None:
        logged: list[str] = []
        mw = LoggingMiddleware(logger=logged.append, log_level="minimal")

        for event in [
            AgentStartEvent(),
            AgentEndEvent(),
            TurnStartEvent(),
            TurnEndEvent(message=_assistant_msg(), tool_results=[]),
        ]:
            await run_middleware_chain(event, _ctx(), [mw])

        assert len(logged) == 4
        assert "[agent_start]" in logged[0]
        assert "[agent_end]" in logged[1]
        assert "[turn_start]" in logged[2]
        assert "[turn_end]" in logged[3]

    @pytest.mark.asyncio
    async def test_minimal_does_not_log_message_events(self) -> None:
        logged: list[str] = []
        mw = LoggingMiddleware(logger=logged.append, log_level="minimal")

        for event in [
            MessageStartEvent(message=_msg()),
            MessageEndEvent(message=_msg()),
            MessageUpdateEvent(message=_msg()),
        ]:
            await run_middleware_chain(event, _ctx(), [mw])

        assert len(logged) == 0


class TestLoggingMiddlewareNormal:
    """Test normal log level — lifecycle + message/tool start/end."""

    @pytest.mark.asyncio
    async def test_normal_logs_message_and_tool_events(self) -> None:
        logged: list[str] = []
        mw = LoggingMiddleware(logger=logged.append, log_level="normal")

        events: list[AgentEvent] = [
            AgentStartEvent(),
            TurnStartEvent(),
            MessageStartEvent(message=_msg()),
            MessageEndEvent(message=_msg()),
            ToolStartEvent(tool_call_id="1", tool_name="test", args={}),
            ToolEndEvent(tool_call_id="1", tool_name="test", result={}),
            TurnEndEvent(message=_assistant_msg(), tool_results=[]),
            AgentEndEvent(),
        ]

        for event in events:
            await run_middleware_chain(event, _ctx(), [mw])

        assert len(logged) == 8

    @pytest.mark.asyncio
    async def test_normal_does_not_log_update_events(self) -> None:
        logged: list[str] = []
        mw = LoggingMiddleware(logger=logged.append, log_level="normal")

        for event in [
            MessageUpdateEvent(message=_msg()),
            ToolUpdateEvent(tool_call_id="1", tool_name="test", args={}),
        ]:
            await run_middleware_chain(event, _ctx(), [mw])

        assert len(logged) == 0


class TestLoggingMiddlewareVerbose:
    """Test verbose log level — everything."""

    @pytest.mark.asyncio
    async def test_verbose_logs_update_events(self) -> None:
        logged: list[str] = []
        mw = LoggingMiddleware(logger=logged.append, log_level="verbose")

        for event in [
            MessageUpdateEvent(message=_msg()),
            ToolUpdateEvent(tool_call_id="1", tool_name="test", args={}),
        ]:
            await run_middleware_chain(event, _ctx(), [mw])

        assert len(logged) == 2
        assert "[message_update]" in logged[0]
        assert "[tool_update]" in logged[1]


class TestLoggingMiddlewareContent:
    """Test include_content flag."""

    @pytest.mark.asyncio
    async def test_include_content_false(self) -> None:
        logged: list[str] = []
        mw = LoggingMiddleware(logger=logged.append, log_level="minimal", include_content=False)

        await run_middleware_chain(AgentStartEvent(), _ctx(), [mw])

        assert len(logged) == 1
        assert logged[0] == "[agent_start]"

    @pytest.mark.asyncio
    async def test_include_content_true(self) -> None:
        logged: list[str] = []
        mw = LoggingMiddleware(logger=logged.append, log_level="minimal", include_content=True)

        await run_middleware_chain(AgentStartEvent(), _ctx(), [mw])

        assert len(logged) == 1
        assert "[agent_start]" in logged[0]
        # Should include JSON content
        assert "agent_start" in logged[0]
        assert len(logged[0]) > len("[agent_start]")


class TestLoggingMiddlewareCustomLogger:
    """Test custom logger callback."""

    @pytest.mark.asyncio
    async def test_custom_logger_is_called(self) -> None:
        call_count = 0

        def custom_logger(msg: str) -> None:
            nonlocal call_count
            call_count += 1

        mw = LoggingMiddleware(logger=custom_logger, log_level="minimal")
        await run_middleware_chain(AgentStartEvent(), _ctx(), [mw])
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_passthrough_behavior(self) -> None:
        """LoggingMiddleware should always pass events through."""
        mw = LoggingMiddleware(logger=lambda _: None, log_level="minimal")
        event = AgentStartEvent()
        result = await run_middleware_chain(event, _ctx(), [mw])
        assert result is not None
        assert result.type == "agent_start"
