"""Context management example — pruning strategies.

This example demonstrates the context management features:
- Sliding window pruning
- Selective pruning
- Message pinning
- Context usage estimation

Uses a mock provider so no API key is needed.
"""

from __future__ import annotations

import asyncio
import time

from isotopo_core import (
    SelectivePruningStrategy,
    SlidingWindowStrategy,
    count_tokens,
    create_sliding_window_transform,
    estimate_context_usage,
    get_context_window,
    pin_message,
)
from isotopo_core.types import (
    AssistantMessage,
    Context,
    Message,
    StopReason,
    TextContent,
    Usage,
    UserMessage,
)


def _ts() -> int:
    return int(time.time() * 1000)


def _user(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=_ts())


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=5, output_tokens=5),
        timestamp=_ts(),
    )


# =============================================================================
# Demonstrate pruning strategies
# =============================================================================


async def demo_sliding_window() -> None:
    """Demonstrate sliding window pruning."""
    print("=== Sliding Window Pruning ===\n")

    messages: list[Message] = [
        _user("Hello"),
        _assistant("Hi there!"),
        _user("Tell me about Python"),
        _assistant("Python is a programming language."),
        _user("What about JavaScript?"),
        _assistant("JavaScript runs in browsers."),
        _user("And Rust?"),
        _assistant("Rust is a systems language."),
        _user("Thanks!"),
        _assistant("You're welcome!"),
    ]

    print(f"Original messages: {len(messages)}")
    print(f"Original tokens: {count_tokens(messages)}")

    strategy = SlidingWindowStrategy(keep_recent=4, keep_first_n=1)
    result = await strategy.prune(messages, target_tokens=30)

    print(f"After pruning: {len(result.messages)} messages")
    print(f"Pruned: {result.pruned_count} messages, {result.pruned_tokens} tokens\n")

    for msg in result.messages:
        role = msg.role
        text = msg.content[0].text if msg.content else ""  # type: ignore[union-attr]
        print(f"  [{role}] {text}")


async def demo_selective_pruning() -> None:
    """Demonstrate selective pruning with pinning."""
    print("\n=== Selective Pruning with Pinning ===\n")

    messages: list[Message] = [
        _user("Important context: project uses Python 3.11"),
        _assistant("Noted!"),
        _user("What is 2+2?"),
        _assistant("4"),
        _user("What is 3+3?"),
        _assistant("6"),
        _user("What is 4+4?"),
        _assistant("8"),
    ]

    # Pin the first message (important context)
    messages = pin_message(messages, 0)
    print(f"Pinned message 0: {messages[0].content[0].text}")  # type: ignore[union-attr]

    strategy = SelectivePruningStrategy(keep_recent=4)
    result = await strategy.prune(messages, target_tokens=30)

    print(f"After pruning: {len(result.messages)} messages")
    print(f"Pruned: {result.pruned_count} messages\n")

    for msg in result.messages:
        role = msg.role
        text = msg.content[0].text if msg.content else ""  # type: ignore[union-attr]
        pinned = " (pinned)" if getattr(msg, "pinned", False) else ""
        print(f"  [{role}]{pinned} {text}")


async def demo_context_usage() -> None:
    """Demonstrate context usage estimation."""
    print("\n=== Context Usage Estimation ===\n")

    context = Context(
        system_prompt="You are a helpful coding assistant specializing in Python.",
        messages=[
            _user("Write a function to sort a list"),
            _assistant("def sort_list(lst):\n    return sorted(lst)"),
            _user("Add type hints"),
            _assistant("def sort_list(lst: list[int]) -> list[int]:\n    return sorted(lst)"),
        ],
    )

    # Default model window
    usage = estimate_context_usage(context)
    print(f"System tokens: {usage.system_tokens}")
    print(f"Message tokens: {usage.message_tokens}")
    print(f"Tool tokens: {usage.tool_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Context window: {usage.context_window}")
    print(f"Remaining: {usage.remaining_tokens}")
    print(f"Utilization: {usage.utilization:.1%}")

    # With custom model
    usage2 = estimate_context_usage(context, model="gpt-4")
    print("\nWith GPT-4 (8k window):")
    print(f"  Context window: {usage2.context_window}")
    print(f"  Utilization: {usage2.utilization:.1%}")


async def demo_transform_context() -> None:
    """Demonstrate using a transform_context hook factory."""
    print("\n=== Transform Context Hook ===\n")

    # Create a sliding window transform
    transform = create_sliding_window_transform(
        max_tokens=50,
        keep_recent=4,
    )

    messages: list[Message] = [
        _user("Message 1"),
        _assistant("Reply 1"),
        _user("Message 2"),
        _assistant("Reply 2"),
        _user("Message 3"),
        _assistant("Reply 3"),
        _user("Message 4"),
        _assistant("Reply 4"),
    ]

    result = await transform(messages, None)
    print(f"Before transform: {len(messages)} messages")
    print(f"After transform: {len(result)} messages")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    await demo_sliding_window()
    await demo_selective_pruning()
    await demo_context_usage()
    await demo_transform_context()

    print(f"\n\nDefault context window: {get_context_window()}")
    print(f"GPT-4o context window: {get_context_window('gpt-4o')}")


if __name__ == "__main__":
    asyncio.run(main())
