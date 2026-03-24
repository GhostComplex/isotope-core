"""Integration tests using the real proxy at localhost:4141/v1.

These tests exercise isotope-core against a live OpenAI-compatible proxy.
All tests are marked with ``@pytest.mark.integration`` and are skipped
automatically when the proxy is unreachable.

Run:
    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from isotope_core import Agent
from isotope_core.providers.proxy import ProxyProvider
from isotope_core.tools import Tool
from isotope_core.types import (
    AgentEvent,
    AssistantMessage,
    Context,
    TextContent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROXY_BASE_URL = "http://localhost:4141/v1"
DEFAULT_MODEL = "gpt-4o-mini"


async def _collect_events(
    gen: Any,
) -> list[AgentEvent]:
    """Drain an async generator of events into a list."""
    events: list[AgentEvent] = []
    async for event in gen:
        events.append(event)
    return events


# ===========================================================================
# 1. Basic streaming
# ===========================================================================


@pytest.mark.integration
async def test_basic_streaming(proxy_provider: ProxyProvider) -> None:
    """Send a prompt and verify core streaming events appear."""
    context = Context(
        system_prompt="Reply with exactly one short sentence.",
        messages=[
            UserMessage(
                content=[TextContent(text="Say hello.")],
                timestamp=0,
            )
        ],
    )

    event_types: list[str] = []
    async for event in proxy_provider.stream(context):
        event_types.append(event.type)

    # Must see: start → (text_delta)+ → done
    assert "start" in event_types, f"Missing start event. Got: {event_types}"
    assert "text_delta" in event_types, f"Missing text_delta event. Got: {event_types}"
    assert event_types[-1] == "done", f"Last event should be done, got: {event_types[-1]}"


# ===========================================================================
# 2. Multi-turn conversation
# ===========================================================================


@pytest.mark.integration
async def test_multi_turn_conversation(proxy_provider: ProxyProvider) -> None:
    """Two-turn conversation where the model must recall the first turn."""
    agent = Agent(
        provider=proxy_provider,
        system_prompt="You are a helpful assistant. Remember what the user tells you.",
    )

    # Turn 1: tell the model a secret word
    events_1 = await _collect_events(
        agent.prompt("Remember this secret word: banana42. Acknowledge you received it.")
    )
    assert any(e.type == "agent_end" for e in events_1)

    # Turn 2: ask the model to recall
    events_2 = await _collect_events(
        agent.prompt("What was the secret word I told you? Reply with just the word.")
    )
    assert any(e.type == "agent_end" for e in events_2)

    # The assistant's last message should contain the secret word
    assistant_messages = [
        m for m in agent.messages if isinstance(m, AssistantMessage)
    ]
    assert len(assistant_messages) >= 2
    last_text = "".join(
        block.text for block in assistant_messages[-1].content if isinstance(block, TextContent)
    )
    assert "banana42" in last_text.lower(), f"Model didn't recall the secret. Got: {last_text}"


# ===========================================================================
# 3. Tool calling
# ===========================================================================


@pytest.mark.integration
async def test_tool_calling(
    proxy_provider: ProxyProvider,
    get_current_time_tool: Tool,
) -> None:
    """Register a tool, ask the model to use it, verify it gets called."""
    agent = Agent(
        provider=proxy_provider,
        system_prompt="You have access to tools. Use them when asked.",
        tools=[get_current_time_tool],
    )

    events = await _collect_events(agent.prompt("What is the current time?"))
    event_types = [e.type for e in events]

    # The model should have called the tool
    assert "tool_start" in event_types, f"Expected tool_start. Got: {event_types}"
    assert "tool_end" in event_types, f"Expected tool_end. Got: {event_types}"

    # Should complete successfully
    assert any(e.type == "agent_end" for e in events)


# ===========================================================================
# 4. Agent class end-to-end
# ===========================================================================


@pytest.mark.integration
async def test_agent_e2e(proxy_provider: ProxyProvider) -> None:
    """Full Agent with ProxyProvider — basic prompt → response cycle."""
    agent = Agent(
        provider=proxy_provider,
        system_prompt="You are a concise assistant. Reply in one sentence.",
    )

    events = await _collect_events(agent.prompt("What is 2 + 2?"))
    event_types = [e.type for e in events]

    # Expected event flow
    assert event_types[0] == "agent_start"
    assert "turn_start" in event_types
    assert "message_start" in event_types
    assert "message_update" in event_types
    assert "message_end" in event_types
    assert "turn_end" in event_types
    assert event_types[-1] == "agent_end"

    # Agent should have messages
    assert len(agent.messages) >= 2  # user + assistant

    # Assistant message should have usage info
    assistant_msgs = [m for m in agent.messages if isinstance(m, AssistantMessage)]
    assert len(assistant_msgs) >= 1
    assert assistant_msgs[0].usage.total_tokens > 0


# ===========================================================================
# 5. Abort mid-stream
# ===========================================================================


@pytest.mark.integration
async def test_abort_mid_stream(proxy_provider: ProxyProvider) -> None:
    """Start a long generation, abort partway, verify clean stop."""
    context = Context(
        system_prompt="Write a very long story about a dragon.",
        messages=[
            UserMessage(
                content=[TextContent(text="Tell me a very long detailed story.")],
                timestamp=0,
            )
        ],
    )

    signal = asyncio.Event()
    event_types: list[str] = []
    text_count = 0

    async for event in proxy_provider.stream(context, signal=signal):
        event_types.append(event.type)
        if event.type == "text_delta":
            text_count += 1
            # Abort after receiving a few text chunks
            if text_count >= 3:
                signal.set()

    # After abort, the stream should have terminated cleanly
    # Last event should be either done or error (abort → error with ABORTED)
    assert event_types[-1] in ("done", "error"), (
        f"Expected done or error after abort, got: {event_types[-1]}"
    )


# ===========================================================================
# 6. Event ordering
# ===========================================================================


@pytest.mark.integration
async def test_event_ordering(proxy_provider: ProxyProvider) -> None:
    """Verify the correct ordering of event types from the Agent."""
    agent = Agent(
        provider=proxy_provider,
        system_prompt="Reply briefly.",
    )

    events = await _collect_events(agent.prompt("Hi"))
    event_types = [e.type for e in events]

    # Verify ordering constraints
    # agent_start must be first
    assert event_types[0] == "agent_start"
    # agent_end must be last
    assert event_types[-1] == "agent_end"

    # turn_start must come before turn_end
    ts_idx = event_types.index("turn_start")
    te_idx = event_types.index("turn_end")
    assert ts_idx < te_idx

    # message_start must come before message_end
    ms_indices = [i for i, t in enumerate(event_types) if t == "message_start"]
    me_indices = [i for i, t in enumerate(event_types) if t == "message_end"]
    # Each message_start should have a corresponding message_end after it
    assert len(ms_indices) >= 1
    assert len(me_indices) >= 1
    # First message_start should come before first message_end
    assert ms_indices[0] < me_indices[0]


# ===========================================================================
# 7. Token usage tracking
# ===========================================================================


@pytest.mark.integration
async def test_token_usage(proxy_provider: ProxyProvider) -> None:
    """Verify token usage is reported in assistant messages."""
    agent = Agent(
        provider=proxy_provider,
        system_prompt="You are brief.",
    )

    await _collect_events(agent.prompt("Say one word."))

    assistant_msgs = [m for m in agent.messages if isinstance(m, AssistantMessage)]
    assert len(assistant_msgs) >= 1

    usage = assistant_msgs[0].usage
    assert usage.input_tokens > 0, "Expected non-zero input_tokens"
    assert usage.output_tokens > 0, "Expected non-zero output_tokens"
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


# ===========================================================================
# 8. Agent with tool calling end-to-end
# ===========================================================================


@pytest.mark.integration
async def test_agent_tool_calling_e2e(
    proxy_provider: ProxyProvider,
    get_current_time_tool: Tool,
    calculator_tool: Tool,
) -> None:
    """Full Agent flow with multiple tools registered."""
    agent = Agent(
        provider=proxy_provider,
        system_prompt="Use the calculator tool to compute math expressions.",
        tools=[get_current_time_tool, calculator_tool],
    )

    events = await _collect_events(
        agent.prompt("Use the calculator tool to compute 123 * 456.")
    )
    event_types = [e.type for e in events]

    # Should have used a tool
    assert "tool_start" in event_types
    assert "tool_end" in event_types

    # Should have completed
    assert event_types[-1] == "agent_end"

    # Final assistant message should reference the result (56088 or 56,088)
    assistant_msgs = [m for m in agent.messages if isinstance(m, AssistantMessage)]
    last_text = "".join(
        block.text
        for block in assistant_msgs[-1].content
        if isinstance(block, TextContent)
    )
    # Models may format with commas (56,088) or without (56088)
    normalized = last_text.replace(",", "")
    assert "56088" in normalized, f"Expected 56088 in response. Got: {last_text}"
