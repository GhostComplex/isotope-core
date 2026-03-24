"""Context management for isotope-core.

This module provides token counting, context usage estimation, pruning
strategies, message pinning helpers, and transform context hook factories.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from isotope_core.providers.base import Provider
from isotope_core.types import (
    AssistantMessage,
    Context,
    Message,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)

# =============================================================================
# Optional tiktoken import
# =============================================================================

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False

# =============================================================================
# Model Context Windows
# =============================================================================

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o3": 200_000,
    "o4-mini": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
}

_DEFAULT_CONTEXT_WINDOW = 128_000

# Characters per token for fallback estimation
_CHARS_PER_TOKEN = 4


def get_context_window(
    model: str | None = None,
    custom_windows: dict[str, int] | None = None,
) -> int:
    """Get the context window size for a model.

    Args:
        model: The model name. If None, returns the default.
        custom_windows: Optional custom overrides for model context windows.

    Returns:
        The context window size in tokens.
    """
    if model is not None:
        if custom_windows and model in custom_windows:
            return custom_windows[model]
        if model in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model]
    return _DEFAULT_CONTEXT_WINDOW


# =============================================================================
# Token Counting
# =============================================================================


def _get_encoding(model: str | None = None) -> Any:
    """Get a tiktoken encoding for the model, or None if unavailable."""
    if not _HAS_TIKTOKEN or tiktoken is None:
        return None
    try:
        if model is not None:
            return tiktoken.encoding_for_model(model)
    except KeyError:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _estimate_tokens_fallback(text: str) -> int:
    """Estimate token count using character-based heuristic (~4 chars/token)."""
    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0


def _count_text_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    encoding = _get_encoding(model)
    if encoding is not None:
        return len(encoding.encode(text))
    return _estimate_tokens_fallback(text)


def _extract_message_text(message: Message) -> str:
    """Extract all text content from a message for token counting."""
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolCallContent):
            parts.append(block.name)
            try:
                parts.append(json.dumps(block.arguments))
            except (TypeError, ValueError):
                parts.append(str(block.arguments))
    # Add tool-related metadata for ToolResultMessage
    if isinstance(message, ToolResultMessage):
        parts.append(message.tool_name)
    return " ".join(parts)


def count_message_tokens(message: Message, model: str | None = None) -> int:
    """Count tokens in a single message.

    Args:
        message: The message to count tokens for.
        model: Optional model name for accurate tiktoken counting.

    Returns:
        The estimated number of tokens.
    """
    text = _extract_message_text(message)
    # Add overhead per message (role, formatting, etc.)
    overhead = 4  # Approximate message framing overhead
    return _count_text_tokens(text, model) + overhead


def count_tokens(
    messages: list[Message],
    model: str | None = None,
) -> int:
    """Count total tokens in a message list.

    Args:
        messages: List of messages to count.
        model: Optional model name for accurate tiktoken counting.

    Returns:
        The estimated total token count.
    """
    return sum(count_message_tokens(msg, model) for msg in messages)


def _count_tool_tokens(tools: list[ToolSchema], model: str | None = None) -> int:
    """Count tokens used by tool definitions."""
    if not tools:
        return 0
    total = 0
    for t in tools:
        text = f"{t.name} {t.description} {json.dumps(t.parameters)}"
        total += _count_text_tokens(text, model)
    return total


# =============================================================================
# Context Usage
# =============================================================================


@dataclass
class ContextUsage:
    """Usage statistics for a context."""

    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    context_window: int
    remaining_tokens: int
    utilization: float  # 0.0 to 1.0


def estimate_context_usage(
    context: Context,
    model: str | None = None,
    custom_windows: dict[str, int] | None = None,
) -> ContextUsage:
    """Estimate the token usage of a context.

    Args:
        context: The context to estimate.
        model: Optional model name for accurate token counting and window lookup.
        custom_windows: Optional custom context window overrides.

    Returns:
        A ContextUsage with estimated token counts.
    """
    system_tokens = _count_text_tokens(context.system_prompt, model)
    message_tokens = count_tokens(context.messages, model)
    tool_tokens = _count_tool_tokens(context.tools, model)

    total_tokens = system_tokens + message_tokens + tool_tokens
    context_window = get_context_window(model, custom_windows)
    remaining_tokens = max(0, context_window - total_tokens)
    utilization = min(1.0, total_tokens / context_window) if context_window > 0 else 0.0

    return ContextUsage(
        total_tokens=total_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        context_window=context_window,
        remaining_tokens=remaining_tokens,
        utilization=utilization,
    )


# =============================================================================
# Message Pinning Helpers
# =============================================================================


def pin_message(messages: list[Message], index: int) -> list[Message]:
    """Pin a message at the given index.

    Returns a new list with the message at `index` having pinned=True.

    Args:
        messages: The message list.
        index: Index of the message to pin.

    Returns:
        A new list with the pinned message.

    Raises:
        IndexError: If the index is out of range.
    """
    if index < 0 or index >= len(messages):
        raise IndexError(f"Message index {index} out of range (0..{len(messages) - 1})")
    result = list(messages)
    msg = result[index]
    if isinstance(msg, (UserMessage, AssistantMessage, ToolResultMessage)):
        result[index] = msg.model_copy(update={"pinned": True})
    return result


def unpin_message(messages: list[Message], index: int) -> list[Message]:
    """Unpin a message at the given index.

    Returns a new list with the message at `index` having pinned=False.

    Args:
        messages: The message list.
        index: Index of the message to unpin.

    Returns:
        A new list with the unpinned message.

    Raises:
        IndexError: If the index is out of range.
    """
    if index < 0 or index >= len(messages):
        raise IndexError(f"Message index {index} out of range (0..{len(messages) - 1})")
    result = list(messages)
    msg = result[index]
    if isinstance(msg, (UserMessage, AssistantMessage, ToolResultMessage)):
        result[index] = msg.model_copy(update={"pinned": False})
    return result


# =============================================================================
# Pruning Strategies
# =============================================================================


@dataclass
class PruneResult:
    """Result of a pruning operation."""

    messages: list[Message]
    pruned_count: int
    pruned_tokens: int


@runtime_checkable
class PruningStrategy(Protocol):
    """Protocol for pluggable pruning strategies."""

    async def prune(
        self,
        messages: list[Message],
        target_tokens: int,
        *,
        model: str | None = None,
    ) -> PruneResult:
        """Prune messages to fit within the target token budget.

        Args:
            messages: The messages to prune.
            target_tokens: The target maximum token count.
            model: Optional model name for token counting.

        Returns:
            A PruneResult with pruned messages and statistics.
        """
        ...


class SlidingWindowStrategy:
    """Drop oldest turns until the messages fit within budget.

    Always keeps the system prompt (not in messages), respects pinned
    messages, and preserves the most recent messages.
    """

    def __init__(
        self,
        keep_recent: int = 10,
        keep_first_n: int = 0,
    ) -> None:
        """Initialize the sliding window strategy.

        Args:
            keep_recent: Number of most recent messages to always keep.
            keep_first_n: Number of messages from the start to always keep
                (useful for context-setting messages).
        """
        self.keep_recent = keep_recent
        self.keep_first_n = keep_first_n

    async def prune(
        self,
        messages: list[Message],
        target_tokens: int,
        *,
        model: str | None = None,
    ) -> PruneResult:
        """Prune by dropping oldest non-protected messages."""
        current_tokens = count_tokens(messages, model)
        if current_tokens <= target_tokens:
            return PruneResult(messages=list(messages), pruned_count=0, pruned_tokens=0)

        total = len(messages)
        # Build set of protected indices
        protected: set[int] = set()

        # Protect first N
        for i in range(min(self.keep_first_n, total)):
            protected.add(i)

        # Protect last keep_recent
        recent_start = max(0, total - self.keep_recent)
        for i in range(recent_start, total):
            protected.add(i)

        # Protect pinned
        for i, msg in enumerate(messages):
            if getattr(msg, "pinned", False):
                protected.add(i)

        # Find prunable messages (oldest first)
        prunable = [i for i in range(total) if i not in protected]

        pruned_count = 0
        pruned_tokens = 0
        removed: set[int] = set()

        for idx in prunable:
            if current_tokens <= target_tokens:
                break
            msg_tokens = count_message_tokens(messages[idx], model)
            removed.add(idx)
            current_tokens -= msg_tokens
            pruned_count += 1
            pruned_tokens += msg_tokens

        kept = [msg for i, msg in enumerate(messages) if i not in removed]
        return PruneResult(messages=kept, pruned_count=pruned_count, pruned_tokens=pruned_tokens)


class SummarizationStrategy:
    """Compress older messages into a summary via an LLM call.

    Keeps the most recent N messages verbatim and summarizes the rest.
    The summary is inserted as a pinned user message at the beginning.
    """

    def __init__(
        self,
        provider: Provider,
        keep_recent: int = 5,
        summary_prompt: str | None = None,
    ) -> None:
        """Initialize the summarization strategy.

        Args:
            provider: A Provider instance used for the summarization LLM call.
            keep_recent: Number of most recent messages to keep verbatim.
            summary_prompt: Custom prompt template for summarization. If None,
                a default prompt is used.
        """
        self.provider = provider
        self.keep_recent = keep_recent
        self.summary_prompt = summary_prompt or (
            "Summarize the following conversation concisely, preserving "
            "key facts, decisions, and context needed for continuation:\n\n"
        )

    async def prune(
        self,
        messages: list[Message],
        target_tokens: int,
        *,
        model: str | None = None,
    ) -> PruneResult:
        """Prune by summarizing old messages."""
        current_tokens = count_tokens(messages, model)
        if current_tokens <= target_tokens:
            return PruneResult(messages=list(messages), pruned_count=0, pruned_tokens=0)

        total = len(messages)
        recent_start = max(0, total - self.keep_recent)

        # Separate old and recent
        old_messages = list(messages[:recent_start])
        recent_messages = list(messages[recent_start:])

        # Extract pinned from old messages — they must be kept
        pinned_old: list[Message] = []
        to_summarize: list[Message] = []
        for msg in old_messages:
            if getattr(msg, "pinned", False):
                pinned_old.append(msg)
            else:
                to_summarize.append(msg)

        if not to_summarize:
            # Nothing to summarize
            return PruneResult(messages=list(messages), pruned_count=0, pruned_tokens=0)

        # Build summarization text
        summary_parts: list[str] = []
        for msg in to_summarize:
            role = msg.role
            text = _extract_message_text(msg)
            summary_parts.append(f"{role}: {text}")

        conversation_text = "\n".join(summary_parts)
        summarize_input = f"{self.summary_prompt}{conversation_text}"

        # Call the provider for summarization
        import time

        summarize_context = Context(
            system_prompt="You are a helpful summarizer.",
            messages=[
                UserMessage(
                    content=[TextContent(text=summarize_input)],
                    timestamp=int(time.time() * 1000),
                )
            ],
        )

        summary_text = ""
        async for event in self.provider.stream(summarize_context):
            if event.type == "done":
                # Extract text from the completed message
                for block in event.message.content:
                    if isinstance(block, TextContent):
                        summary_text += block.text

        # Create a pinned summary message
        summary_message = UserMessage(
            content=[TextContent(text=f"[Summary of earlier conversation]\n{summary_text}")],
            timestamp=int(time.time() * 1000),
            pinned=True,
        )

        pruned_count = len(to_summarize)
        pruned_tokens = count_tokens(to_summarize, model)

        result_messages: list[Message] = [summary_message, *pinned_old, *recent_messages]
        return PruneResult(
            messages=result_messages,
            pruned_count=pruned_count,
            pruned_tokens=pruned_tokens,
        )


class SelectivePruningStrategy:
    """Keep system + recent N + pinned messages, drop everything else."""

    def __init__(self, keep_recent: int = 10) -> None:
        """Initialize the selective pruning strategy.

        Args:
            keep_recent: Number of most recent messages to always keep.
        """
        self.keep_recent = keep_recent

    async def prune(
        self,
        messages: list[Message],
        target_tokens: int,
        *,
        model: str | None = None,
    ) -> PruneResult:
        """Prune by keeping only recent + pinned messages."""
        current_tokens = count_tokens(messages, model)
        if current_tokens <= target_tokens:
            return PruneResult(messages=list(messages), pruned_count=0, pruned_tokens=0)

        total = len(messages)
        recent_start = max(0, total - self.keep_recent)

        # Build protected indices: recent + pinned
        protected: set[int] = set()
        for i in range(recent_start, total):
            protected.add(i)
        for i, msg in enumerate(messages):
            if getattr(msg, "pinned", False):
                protected.add(i)

        pruned_count = 0
        pruned_tokens = 0
        kept: list[Message] = []

        for i, msg in enumerate(messages):
            if i in protected:
                kept.append(msg)
            else:
                pruned_count += 1
                pruned_tokens += count_message_tokens(msg, model)

        return PruneResult(messages=kept, pruned_count=pruned_count, pruned_tokens=pruned_tokens)


# =============================================================================
# Transform Context Hook Factories
# =============================================================================

# Import the hook type alias
from isotope_core.loop import TransformContextHook  # noqa: E402


def create_sliding_window_transform(
    max_tokens: int | None = None,
    keep_recent: int = 10,
    model: str | None = None,
    keep_first_n: int = 0,
) -> TransformContextHook:
    """Create a transform_context hook that applies sliding window pruning.

    Args:
        max_tokens: Maximum token budget for messages. If None, uses model's
            context window (or default).
        keep_recent: Number of recent messages to always keep.
        model: Optional model name for token counting and window lookup.
        keep_first_n: Number of first messages to preserve.

    Returns:
        A TransformContextHook function.
    """
    strategy = SlidingWindowStrategy(keep_recent=keep_recent, keep_first_n=keep_first_n)

    async def transform(
        messages: list[Message], signal: asyncio.Event | None
    ) -> list[Message]:
        target = max_tokens if max_tokens is not None else get_context_window(model)
        result = await strategy.prune(messages, target, model=model)
        return result.messages

    return transform


def create_summarization_transform(
    provider: Provider,
    max_tokens: int | None = None,
    keep_recent: int = 5,
    model: str | None = None,
    summary_prompt: str | None = None,
) -> TransformContextHook:
    """Create a transform_context hook that applies summarization pruning.

    Args:
        provider: A Provider instance for summarization LLM calls.
        max_tokens: Maximum token budget for messages. If None, uses model's
            context window (or default).
        keep_recent: Number of recent messages to keep verbatim.
        model: Optional model name for token counting and window lookup.
        summary_prompt: Custom summarization prompt.

    Returns:
        A TransformContextHook function.
    """
    strategy = SummarizationStrategy(
        provider=provider,
        keep_recent=keep_recent,
        summary_prompt=summary_prompt,
    )

    async def transform(
        messages: list[Message], signal: asyncio.Event | None
    ) -> list[Message]:
        target = max_tokens if max_tokens is not None else get_context_window(model)
        result = await strategy.prune(messages, target, model=model)
        return result.messages

    return transform
