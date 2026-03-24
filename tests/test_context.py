"""Tests for isotope_core.context module — Milestone 3."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

from isotope_core.context import (
    _CHARS_PER_TOKEN,
    _HAS_TIKTOKEN,
    MODEL_CONTEXT_WINDOWS,
    ContextUsage,
    PruneResult,
    PruningStrategy,
    SelectivePruningStrategy,
    SlidingWindowStrategy,
    SummarizationStrategy,
    count_message_tokens,
    count_tokens,
    create_sliding_window_transform,
    create_summarization_transform,
    estimate_context_usage,
    get_context_window,
    pin_message,
    unpin_message,
)
from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
)
from isotope_core.types import (
    AssistantMessage,
    Context,
    ContextPrunedEvent,
    Message,
    StopReason,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)

# =============================================================================
# Helpers
# =============================================================================

_TS = int(time.time() * 1000)


def _user(text: str, *, pinned: bool = False) -> UserMessage:
    return UserMessage(
        content=[TextContent(text=text)],
        timestamp=_TS,
        pinned=pinned,
    )


def _assistant(text: str, *, pinned: bool = False) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        stop_reason=StopReason.END_TURN,
        timestamp=_TS,
        pinned=pinned,
    )


def _tool_result(text: str, *, pinned: bool = False) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="call_1",
        tool_name="test_tool",
        content=[TextContent(text=text)],
        timestamp=_TS,
        pinned=pinned,
    )


# =============================================================================
# Mock Provider for summarization tests
# =============================================================================


class MockSummarizationProvider:
    """Mock provider that returns a canned summary."""

    def __init__(self, summary: str = "This is a summary.") -> None:
        self.summary = summary
        self.call_count = 0

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.call_count += 1
        msg = AssistantMessage(
            content=[TextContent(text=self.summary)],
            stop_reason=StopReason.END_TURN,
            timestamp=int(time.time() * 1000),
        )
        yield StreamStartEvent(partial=msg)
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Token Counting Tests
# =============================================================================


class TestCountMessageTokens:
    """Tests for count_message_tokens."""

    def test_simple_text_message(self) -> None:
        """Counting tokens for a simple text message produces a positive number."""
        msg = _user("Hello, world!")
        tokens = count_message_tokens(msg)
        assert tokens > 0

    def test_empty_text_message(self) -> None:
        """An empty text message still has framing overhead."""
        msg = _user("")
        tokens = count_message_tokens(msg)
        # Should have at least the overhead
        assert tokens >= 4

    def test_assistant_message(self) -> None:
        """Token counting works for assistant messages."""
        msg = _assistant("This is a response from the assistant.")
        tokens = count_message_tokens(msg)
        assert tokens > 0

    def test_tool_result_message(self) -> None:
        """Token counting works for tool result messages."""
        msg = _tool_result("Tool output here")
        tokens = count_message_tokens(msg)
        assert tokens > 0

    def test_message_with_tool_call(self) -> None:
        """Token counting includes tool call name and arguments."""
        msg = AssistantMessage(
            content=[
                TextContent(text="Calling tool"),
                ToolCallContent(id="c1", name="get_weather", arguments={"loc": "NYC"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            timestamp=_TS,
        )
        tokens = count_message_tokens(msg)
        assert tokens > count_message_tokens(_assistant("Calling tool"))

    def test_longer_text_more_tokens(self) -> None:
        """Longer text produces more tokens."""
        short = count_message_tokens(_user("hi"))
        long = count_message_tokens(_user("hello world, this is a much longer text message"))
        assert long > short


class TestCountTokens:
    """Tests for count_tokens on message lists."""

    def test_empty_list(self) -> None:
        """Empty message list returns 0."""
        assert count_tokens([]) == 0

    def test_multiple_messages(self) -> None:
        """Total is sum of individual counts."""
        msgs: list[Message] = [_user("hello"), _assistant("hi back")]
        total = count_tokens(msgs)
        individual = sum(count_message_tokens(m) for m in msgs)
        assert total == individual

    def test_with_model_parameter(self) -> None:
        """Passing a model parameter doesn't crash."""
        msgs: list[Message] = [_user("hello")]
        tokens = count_tokens(msgs, model="gpt-4o")
        assert tokens > 0


class TestTokenCountingFallback:
    """Tests for fallback token counting when tiktoken is not available."""

    def test_fallback_estimation(self) -> None:
        """Fallback estimation uses ~4 chars per token."""
        text = "a" * 100  # 100 chars -> ~25 tokens
        msg = _user(text)
        with patch("isotope_core.context._get_encoding", return_value=None):
            tokens = count_message_tokens(msg, model=None)
            # Tokens = text tokens + overhead (4)
            expected_text_tokens = 100 // _CHARS_PER_TOKEN
            assert tokens == expected_text_tokens + 4


@pytest.mark.skipif(not _HAS_TIKTOKEN, reason="tiktoken not installed")
class TestTokenCountingWithTiktoken:
    """Tests that tiktoken-based counting works when available."""

    def test_tiktoken_counting(self) -> None:
        """With tiktoken, counting should produce reasonable values."""
        msg = _user("Hello world, how are you today?")
        tokens = count_message_tokens(msg, model="gpt-4o")
        # tiktoken should give us a reasonable count
        assert tokens > 0
        assert tokens < 100  # Sanity check


# =============================================================================
# Context Usage Estimation Tests
# =============================================================================


class TestEstimateContextUsage:
    """Tests for estimate_context_usage."""

    def test_empty_context(self) -> None:
        """An empty context has zero utilization."""
        ctx = Context()
        usage = estimate_context_usage(ctx)
        assert isinstance(usage, ContextUsage)
        assert usage.total_tokens == 0
        assert usage.system_tokens == 0
        assert usage.message_tokens == 0
        assert usage.tool_tokens == 0
        assert usage.utilization == 0.0
        assert usage.remaining_tokens == usage.context_window

    def test_context_with_system_prompt(self) -> None:
        """System prompt tokens are counted separately."""
        ctx = Context(system_prompt="You are a helpful assistant.")
        usage = estimate_context_usage(ctx)
        assert usage.system_tokens > 0
        assert usage.message_tokens == 0
        assert usage.total_tokens == usage.system_tokens

    def test_context_with_messages(self) -> None:
        """Messages contribute to message_tokens."""
        ctx = Context(
            messages=[_user("Hello"), _assistant("Hi there")],
        )
        usage = estimate_context_usage(ctx)
        assert usage.message_tokens > 0
        assert usage.total_tokens == usage.system_tokens + usage.message_tokens

    def test_context_with_tools(self) -> None:
        """Tools contribute to tool_tokens."""
        ctx = Context(
            tools=[
                ToolSchema(
                    name="get_weather",
                    description="Get the current weather",
                    parameters={"type": "object", "properties": {"loc": {"type": "string"}}},
                )
            ],
        )
        usage = estimate_context_usage(ctx)
        assert usage.tool_tokens > 0

    def test_utilization_calculation(self) -> None:
        """Utilization is total/window, clamped to [0, 1]."""
        ctx = Context(
            system_prompt="System prompt " * 100,
            messages=[_user("Long message " * 50)],
        )
        # With a tiny window, utilization should be 1.0
        usage_small = estimate_context_usage(ctx, model="test", custom_windows={"test": 100})
        assert usage_small.utilization == 1.0
        assert usage_small.remaining_tokens == 0

    def test_remaining_tokens(self) -> None:
        """Remaining tokens = context_window - total_tokens (min 0)."""
        ctx = Context(system_prompt="Hi")
        usage = estimate_context_usage(ctx)
        assert usage.remaining_tokens == usage.context_window - usage.total_tokens

    def test_model_specific_window(self) -> None:
        """Model-specific context windows are used."""
        ctx = Context()
        usage_gpt4o = estimate_context_usage(ctx, model="gpt-4o")
        assert usage_gpt4o.context_window == 128_000

        usage_claude = estimate_context_usage(ctx, model="claude-sonnet-4-20250514")
        assert usage_claude.context_window == 200_000

    def test_custom_window_override(self) -> None:
        """Custom windows take precedence over built-in ones."""
        ctx = Context()
        usage = estimate_context_usage(
            ctx, model="gpt-4o", custom_windows={"gpt-4o": 50_000}
        )
        assert usage.context_window == 50_000


# =============================================================================
# Model Context Window Tests
# =============================================================================


class TestGetContextWindow:
    """Tests for get_context_window."""

    def test_known_model(self) -> None:
        """Known models return their expected window sizes."""
        assert get_context_window("gpt-4o") == 128_000
        assert get_context_window("o3") == 200_000
        assert get_context_window("claude-sonnet-4-20250514") == 200_000

    def test_unknown_model(self) -> None:
        """Unknown models return the default window."""
        assert get_context_window("unknown-model-xyz") == 128_000

    def test_none_model(self) -> None:
        """None model returns the default window."""
        assert get_context_window(None) == 128_000

    def test_custom_override(self) -> None:
        """Custom overrides take precedence."""
        assert get_context_window("gpt-4o", {"gpt-4o": 50_000}) == 50_000

    def test_custom_override_for_unknown_model(self) -> None:
        """Custom override for unknown model works."""
        assert get_context_window("my-model", {"my-model": 32_000}) == 32_000

    def test_model_context_windows_dict(self) -> None:
        """MODEL_CONTEXT_WINDOWS contains expected models."""
        assert "gpt-4o" in MODEL_CONTEXT_WINDOWS
        assert "claude-sonnet-4-20250514" in MODEL_CONTEXT_WINDOWS


# =============================================================================
# Message Pinning Tests
# =============================================================================


class TestMessagePinning:
    """Tests for pin_message and unpin_message helpers."""

    def test_pin_user_message(self) -> None:
        """Pinning a user message sets pinned=True."""
        msgs: list[Message] = [_user("hello")]
        result = pin_message(msgs, 0)
        assert result[0].pinned is True  # type: ignore[union-attr]
        # Original should be unchanged
        assert msgs[0].pinned is False

    def test_pin_assistant_message(self) -> None:
        """Pinning an assistant message sets pinned=True."""
        msgs: list[Message] = [_assistant("hi")]
        result = pin_message(msgs, 0)
        assert result[0].pinned is True  # type: ignore[union-attr]

    def test_pin_tool_result_message(self) -> None:
        """Pinning a tool result message sets pinned=True."""
        msgs: list[Message] = [_tool_result("result")]
        result = pin_message(msgs, 0)
        assert result[0].pinned is True  # type: ignore[union-attr]

    def test_unpin_message(self) -> None:
        """Unpinning sets pinned=False."""
        msgs: list[Message] = [_user("hello", pinned=True)]
        result = unpin_message(msgs, 0)
        assert result[0].pinned is False  # type: ignore[union-attr]
        # Original remains pinned
        assert msgs[0].pinned is True

    def test_pin_out_of_range(self) -> None:
        """Pinning with out-of-range index raises IndexError."""
        msgs: list[Message] = [_user("hello")]
        with pytest.raises(IndexError):
            pin_message(msgs, 5)
        with pytest.raises(IndexError):
            pin_message(msgs, -1)

    def test_unpin_out_of_range(self) -> None:
        """Unpinning with out-of-range index raises IndexError."""
        msgs: list[Message] = [_user("hello")]
        with pytest.raises(IndexError):
            unpin_message(msgs, 5)


# =============================================================================
# Sliding Window Strategy Tests
# =============================================================================


class TestSlidingWindowStrategy:
    """Tests for SlidingWindowStrategy."""

    @pytest.mark.asyncio
    async def test_no_pruning_needed(self) -> None:
        """When under budget, no messages are pruned."""
        strategy = SlidingWindowStrategy(keep_recent=5)
        msgs: list[Message] = [_user("hi"), _assistant("hello")]
        result = await strategy.prune(msgs, target_tokens=100_000)
        assert result.pruned_count == 0
        assert result.pruned_tokens == 0
        assert len(result.messages) == 2

    @pytest.mark.asyncio
    async def test_prunes_oldest(self) -> None:
        """When over budget, oldest non-protected messages are pruned first."""
        strategy = SlidingWindowStrategy(keep_recent=2)
        msgs: list[Message] = [
            _user("old message 1"),
            _assistant("old response 1"),
            _user("old message 2"),
            _assistant("old response 2"),
            _user("recent message"),
            _assistant("recent response"),
        ]
        # Set a very low budget to force pruning
        result = await strategy.prune(msgs, target_tokens=30)
        assert result.pruned_count > 0
        # Recent messages should be kept
        assert len(result.messages) >= 2
        # Last two messages should be the recent ones
        assert result.messages[-1] == msgs[-1]
        assert result.messages[-2] == msgs[-2]

    @pytest.mark.asyncio
    async def test_respects_pinned_messages(self) -> None:
        """Pinned messages are never pruned."""
        strategy = SlidingWindowStrategy(keep_recent=1)
        msgs: list[Message] = [
            _user("important", pinned=True),
            _assistant("old response"),
            _user("recent"),
        ]
        result = await strategy.prune(msgs, target_tokens=20)
        # Pinned message should still be there
        kept_texts = []
        for m in result.messages:
            for c in m.content:
                if isinstance(c, TextContent):
                    kept_texts.append(c.text)
        assert "important" in kept_texts
        assert "recent" in kept_texts

    @pytest.mark.asyncio
    async def test_keep_first_n(self) -> None:
        """keep_first_n preserves the first N messages."""
        strategy = SlidingWindowStrategy(keep_recent=1, keep_first_n=1)
        msgs: list[Message] = [
            _user("context setup"),
            _assistant("old response 1"),
            _user("old message 2"),
            _assistant("old response 2"),
            _user("recent"),
        ]
        result = await strategy.prune(msgs, target_tokens=30)
        # First message and last message should be kept
        kept_texts = [
            c.text for m in result.messages for c in m.content if isinstance(c, TextContent)
        ]
        assert "context setup" in kept_texts
        assert "recent" in kept_texts

    @pytest.mark.asyncio
    async def test_all_pinned(self) -> None:
        """When all messages are pinned, none are pruned even if over budget."""
        strategy = SlidingWindowStrategy(keep_recent=0)
        msgs: list[Message] = [
            _user("pinned 1", pinned=True),
            _assistant("pinned 2", pinned=True),
        ]
        result = await strategy.prune(msgs, target_tokens=1)
        assert result.pruned_count == 0
        assert len(result.messages) == 2

    @pytest.mark.asyncio
    async def test_empty_messages(self) -> None:
        """Empty message list returns empty result."""
        strategy = SlidingWindowStrategy()
        result = await strategy.prune([], target_tokens=100)
        assert result.messages == []
        assert result.pruned_count == 0

    @pytest.mark.asyncio
    async def test_single_message(self) -> None:
        """Single message list, under budget, is kept."""
        strategy = SlidingWindowStrategy(keep_recent=1)
        msgs: list[Message] = [_user("hi")]
        result = await strategy.prune(msgs, target_tokens=100_000)
        assert len(result.messages) == 1

    @pytest.mark.asyncio
    async def test_prune_result_type(self) -> None:
        """Result is a PruneResult dataclass."""
        strategy = SlidingWindowStrategy()
        result = await strategy.prune([], target_tokens=100)
        assert isinstance(result, PruneResult)


# =============================================================================
# Summarization Strategy Tests
# =============================================================================


class TestSummarizationStrategy:
    """Tests for SummarizationStrategy."""

    @pytest.mark.asyncio
    async def test_no_pruning_needed(self) -> None:
        """When under budget, no summarization occurs."""
        provider = MockSummarizationProvider()
        strategy = SummarizationStrategy(provider=provider, keep_recent=5)
        msgs: list[Message] = [_user("hi"), _assistant("hello")]
        result = await strategy.prune(msgs, target_tokens=100_000)
        assert result.pruned_count == 0
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_summarization_creates_summary_message(self) -> None:
        """When over budget, old messages are replaced by a summary."""
        provider = MockSummarizationProvider(summary="Conversation summary here.")
        strategy = SummarizationStrategy(provider=provider, keep_recent=1)
        msgs: list[Message] = [
            _user("old message 1"),
            _assistant("old response 1"),
            _user("old message 2"),
            _assistant("old response 2"),
            _user("recent message"),
        ]
        result = await strategy.prune(msgs, target_tokens=30)
        assert result.pruned_count == 4  # 4 old messages summarized
        assert provider.call_count == 1

        # First message should be the summary (pinned)
        first = result.messages[0]
        assert isinstance(first, UserMessage)
        assert first.pinned is True
        assert "summary" in first.content[0].text.lower()  # type: ignore[union-attr]

        # Last message should be the recent one
        assert result.messages[-1] == msgs[-1]

    @pytest.mark.asyncio
    async def test_summarization_respects_pinned(self) -> None:
        """Pinned old messages are kept alongside the summary."""
        provider = MockSummarizationProvider(summary="Summary.")
        strategy = SummarizationStrategy(provider=provider, keep_recent=1)
        msgs: list[Message] = [
            _user("pinned important", pinned=True),
            _assistant("old response"),
            _user("recent"),
        ]
        result = await strategy.prune(msgs, target_tokens=20)
        # The pinned message should be preserved
        kept_texts = [
            c.text for m in result.messages for c in m.content if isinstance(c, TextContent)
        ]
        assert "pinned important" in kept_texts

    @pytest.mark.asyncio
    async def test_summarization_custom_prompt(self) -> None:
        """Custom summary prompt is used."""
        provider = MockSummarizationProvider()
        custom_prompt = "Custom: summarize this:\n"
        strategy = SummarizationStrategy(
            provider=provider,
            keep_recent=1,
            summary_prompt=custom_prompt,
        )
        msgs: list[Message] = [
            _user("old"),
            _assistant("old resp"),
            _user("recent"),
        ]
        await strategy.prune(msgs, target_tokens=10)
        # If provider was called, the custom prompt should have been used
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_summarization_nothing_to_summarize(self) -> None:
        """If all old messages are pinned, nothing to summarize."""
        provider = MockSummarizationProvider()
        strategy = SummarizationStrategy(provider=provider, keep_recent=1)
        msgs: list[Message] = [
            _user("pinned 1", pinned=True),
            _assistant("pinned 2", pinned=True),
            _user("recent"),
        ]
        result = await strategy.prune(msgs, target_tokens=10)
        # All old are pinned → nothing to summarize → return original
        assert provider.call_count == 0
        assert len(result.messages) == 3


# =============================================================================
# Selective Pruning Strategy Tests
# =============================================================================


class TestSelectivePruningStrategy:
    """Tests for SelectivePruningStrategy."""

    @pytest.mark.asyncio
    async def test_no_pruning_needed(self) -> None:
        """Under budget, no pruning."""
        strategy = SelectivePruningStrategy(keep_recent=5)
        msgs: list[Message] = [_user("hi"), _assistant("hello")]
        result = await strategy.prune(msgs, target_tokens=100_000)
        assert result.pruned_count == 0
        assert len(result.messages) == 2

    @pytest.mark.asyncio
    async def test_keeps_recent_and_drops_old(self) -> None:
        """Keeps recent N messages, drops the rest."""
        strategy = SelectivePruningStrategy(keep_recent=2)
        msgs: list[Message] = [
            _user("old 1"),
            _assistant("old 2"),
            _user("old 3"),
            _assistant("old 4"),
            _user("recent 1"),
            _assistant("recent 2"),
        ]
        result = await strategy.prune(msgs, target_tokens=10)
        assert result.pruned_count == 4
        assert len(result.messages) == 2
        # Recent messages preserved
        kept_texts = [
            c.text for m in result.messages for c in m.content if isinstance(c, TextContent)
        ]
        assert "recent 1" in kept_texts
        assert "recent 2" in kept_texts

    @pytest.mark.asyncio
    async def test_respects_pinned_messages(self) -> None:
        """Pinned messages in old section are kept."""
        strategy = SelectivePruningStrategy(keep_recent=1)
        msgs: list[Message] = [
            _user("pinned old", pinned=True),
            _assistant("old response"),
            _user("old message"),
            _user("recent"),
        ]
        result = await strategy.prune(msgs, target_tokens=10)
        kept_texts = [
            c.text for m in result.messages for c in m.content if isinstance(c, TextContent)
        ]
        assert "pinned old" in kept_texts
        assert "recent" in kept_texts
        # Non-pinned old messages should be dropped
        assert "old response" not in kept_texts
        assert "old message" not in kept_texts

    @pytest.mark.asyncio
    async def test_all_pinned(self) -> None:
        """All pinned → nothing is dropped."""
        strategy = SelectivePruningStrategy(keep_recent=0)
        msgs: list[Message] = [
            _user("a", pinned=True),
            _assistant("b", pinned=True),
        ]
        result = await strategy.prune(msgs, target_tokens=1)
        assert result.pruned_count == 0
        assert len(result.messages) == 2

    @pytest.mark.asyncio
    async def test_empty_messages(self) -> None:
        """Empty list returns empty result."""
        strategy = SelectivePruningStrategy()
        result = await strategy.prune([], target_tokens=100)
        assert result.messages == []
        assert result.pruned_count == 0


# =============================================================================
# PruningStrategy Protocol Tests
# =============================================================================


class TestPruningStrategyProtocol:
    """Tests that built-in strategies satisfy the PruningStrategy protocol."""

    def test_sliding_window_is_pruning_strategy(self) -> None:
        """SlidingWindowStrategy satisfies PruningStrategy."""
        s = SlidingWindowStrategy()
        assert isinstance(s, PruningStrategy)

    def test_summarization_is_pruning_strategy(self) -> None:
        """SummarizationStrategy satisfies PruningStrategy."""
        provider = MockSummarizationProvider()
        s = SummarizationStrategy(provider=provider)
        assert isinstance(s, PruningStrategy)

    def test_selective_is_pruning_strategy(self) -> None:
        """SelectivePruningStrategy satisfies PruningStrategy."""
        s = SelectivePruningStrategy()
        assert isinstance(s, PruningStrategy)


# =============================================================================
# Transform Context Hook Factory Tests
# =============================================================================


class TestCreateSlidingWindowTransform:
    """Tests for create_sliding_window_transform."""

    @pytest.mark.asyncio
    async def test_returns_callable(self) -> None:
        """Factory returns a callable hook."""
        hook = create_sliding_window_transform(max_tokens=1000)
        assert callable(hook)

    @pytest.mark.asyncio
    async def test_transform_prunes_when_over_budget(self) -> None:
        """The transform hook prunes messages when over token budget."""
        hook = create_sliding_window_transform(max_tokens=30, keep_recent=1)
        msgs: list[Message] = [
            _user("old message 1"),
            _assistant("old response 1"),
            _user("old message 2"),
            _assistant("old response 2"),
            _user("recent"),
        ]
        result = await hook(msgs, None)
        assert len(result) < len(msgs)
        # Recent message preserved
        assert result[-1] == msgs[-1]

    @pytest.mark.asyncio
    async def test_transform_no_pruning_under_budget(self) -> None:
        """Under budget, all messages are kept."""
        hook = create_sliding_window_transform(max_tokens=100_000, keep_recent=5)
        msgs: list[Message] = [_user("hi"), _assistant("hello")]
        result = await hook(msgs, None)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_transform_with_signal(self) -> None:
        """Transform works with an asyncio.Event signal."""
        hook = create_sliding_window_transform(max_tokens=100_000)
        signal = asyncio.Event()
        msgs: list[Message] = [_user("hi")]
        result = await hook(msgs, signal)
        assert len(result) == 1


class TestCreateSummarizationTransform:
    """Tests for create_summarization_transform."""

    @pytest.mark.asyncio
    async def test_returns_callable(self) -> None:
        """Factory returns a callable hook."""
        provider = MockSummarizationProvider()
        hook = create_summarization_transform(provider=provider, max_tokens=1000)
        assert callable(hook)

    @pytest.mark.asyncio
    async def test_transform_summarizes_when_over_budget(self) -> None:
        """The transform hook summarizes messages when over budget."""
        provider = MockSummarizationProvider(summary="Summary of old messages.")
        hook = create_summarization_transform(
            provider=provider, max_tokens=10, keep_recent=1
        )
        msgs: list[Message] = [
            _user("old message number one with some extra text"),
            _assistant("old response number one with some extra text"),
            _user("old message number two with some extra text"),
            _assistant("old response number two with some extra text"),
            _user("recent message here"),
        ]
        result = await hook(msgs, None)
        assert provider.call_count == 1
        # Should have summary + recent
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_transform_no_pruning_under_budget(self) -> None:
        """Under budget, no summarization is done."""
        provider = MockSummarizationProvider()
        hook = create_summarization_transform(
            provider=provider, max_tokens=100_000, keep_recent=5
        )
        msgs: list[Message] = [_user("hi"), _assistant("hello")]
        result = await hook(msgs, None)
        assert len(result) == 2
        assert provider.call_count == 0


# =============================================================================
# ContextPrunedEvent Tests
# =============================================================================


class TestContextPrunedEvent:
    """Tests for the ContextPrunedEvent type."""

    def test_create_event(self) -> None:
        """Creating a ContextPrunedEvent with all fields."""
        event = ContextPrunedEvent(
            strategy="sliding_window",
            pruned_count=5,
            pruned_tokens=500,
            remaining_tokens=99_500,
        )
        assert event.type == "context_pruned"
        assert event.strategy == "sliding_window"
        assert event.pruned_count == 5
        assert event.pruned_tokens == 500
        assert event.remaining_tokens == 99_500

    def test_serialization(self) -> None:
        """ContextPrunedEvent serializes correctly."""
        event = ContextPrunedEvent(
            strategy="summarization",
            pruned_count=3,
            pruned_tokens=300,
            remaining_tokens=50_000,
        )
        data = event.model_dump()
        assert data["type"] == "context_pruned"
        assert data["strategy"] == "summarization"

    def test_default_type(self) -> None:
        """Type field defaults to 'context_pruned'."""
        event = ContextPrunedEvent(
            strategy="selective",
            pruned_count=0,
            pruned_tokens=0,
            remaining_tokens=100_000,
        )
        assert event.type == "context_pruned"


# =============================================================================
# Pinned Field on Message Types Tests
# =============================================================================


class TestPinnedField:
    """Tests that the pinned field works correctly on message types."""

    def test_user_message_default_not_pinned(self) -> None:
        """UserMessage defaults to pinned=False."""
        msg = UserMessage(
            content=[TextContent(text="hi")],
            timestamp=_TS,
        )
        assert msg.pinned is False

    def test_user_message_pinned(self) -> None:
        """UserMessage can be created with pinned=True."""
        msg = UserMessage(
            content=[TextContent(text="hi")],
            timestamp=_TS,
            pinned=True,
        )
        assert msg.pinned is True

    def test_assistant_message_default_not_pinned(self) -> None:
        """AssistantMessage defaults to pinned=False."""
        msg = AssistantMessage(
            content=[TextContent(text="hi")],
            timestamp=_TS,
        )
        assert msg.pinned is False

    def test_assistant_message_pinned(self) -> None:
        """AssistantMessage can be created with pinned=True."""
        msg = AssistantMessage(
            content=[TextContent(text="hi")],
            timestamp=_TS,
            pinned=True,
        )
        assert msg.pinned is True

    def test_tool_result_message_default_not_pinned(self) -> None:
        """ToolResultMessage defaults to pinned=False."""
        msg = _tool_result("data")
        assert msg.pinned is False

    def test_tool_result_message_pinned(self) -> None:
        """ToolResultMessage can be created with pinned=True."""
        msg = _tool_result("data", pinned=True)
        assert msg.pinned is True

    def test_pinned_serialization(self) -> None:
        """Pinned field round-trips through serialization."""
        msg = _user("hi", pinned=True)
        data = msg.model_dump()
        assert data["pinned"] is True
        restored = UserMessage.model_validate(data)
        assert restored.pinned is True


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestTransformHookWithAgentLoop:
    """Tests that transform hooks integrate with the existing loop signature."""

    @pytest.mark.asyncio
    async def test_sliding_window_hook_signature(self) -> None:
        """Sliding window transform matches TransformContextHook signature."""
        from isotope_core.loop import TransformContextHook

        hook: TransformContextHook = create_sliding_window_transform(max_tokens=1000)
        msgs: list[Message] = [_user("hi")]
        result = await hook(msgs, None)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_summarization_hook_signature(self) -> None:
        """Summarization transform matches TransformContextHook signature."""
        from isotope_core.loop import TransformContextHook

        provider = MockSummarizationProvider()
        hook: TransformContextHook = create_summarization_transform(
            provider=provider, max_tokens=1000
        )
        msgs: list[Message] = [_user("hi")]
        result = await hook(msgs, None)
        assert isinstance(result, list)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for context management."""

    @pytest.mark.asyncio
    async def test_sliding_window_all_recent(self) -> None:
        """When keep_recent >= total messages, nothing is pruned."""
        strategy = SlidingWindowStrategy(keep_recent=100)
        msgs: list[Message] = [_user("a"), _assistant("b")]
        result = await strategy.prune(msgs, target_tokens=1)
        # All protected as recent → nothing pruned
        assert result.pruned_count == 0

    @pytest.mark.asyncio
    async def test_selective_keep_recent_zero(self) -> None:
        """keep_recent=0 means only pinned are kept."""
        strategy = SelectivePruningStrategy(keep_recent=0)
        msgs: list[Message] = [
            _user("a"),
            _assistant("b", pinned=True),
            _user("c"),
        ]
        result = await strategy.prune(msgs, target_tokens=1)
        assert result.pruned_count == 2
        assert len(result.messages) == 1
        assert result.messages[0].pinned is True  # type: ignore[union-attr]

    def test_count_tokens_with_none_model(self) -> None:
        """count_tokens works with model=None."""
        msgs: list[Message] = [_user("hello")]
        tokens = count_tokens(msgs, model=None)
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_prune_result_pruned_tokens_accuracy(self) -> None:
        """pruned_tokens reflects the tokens of removed messages."""
        strategy = SelectivePruningStrategy(keep_recent=1)
        msgs: list[Message] = [
            _user("old message that will be pruned"),
            _user("recent"),
        ]
        result = await strategy.prune(msgs, target_tokens=1)
        expected_pruned = count_message_tokens(msgs[0])
        assert result.pruned_tokens == expected_pruned

    def test_estimate_context_usage_full_context(self) -> None:
        """Full context with system, messages, and tools."""
        ctx = Context(
            system_prompt="You are a helpful assistant with tools.",
            messages=[
                _user("What's the weather?"),
                _assistant("Let me check."),
            ],
            tools=[
                ToolSchema(
                    name="weather",
                    description="Get weather",
                    parameters={"type": "object"},
                ),
            ],
        )
        usage = estimate_context_usage(ctx, model="gpt-4o")
        assert usage.total_tokens > 0
        assert usage.system_tokens > 0
        assert usage.message_tokens > 0
        assert usage.tool_tokens > 0
        assert usage.total_tokens == (
            usage.system_tokens + usage.message_tokens + usage.tool_tokens
        )
