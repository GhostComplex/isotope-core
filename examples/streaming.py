"""Streaming example — consuming the event stream.

This example demonstrates how to consume the full event stream from an agent,
showing every event type. Uses a mock provider so no API key is needed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

from isotope_core import Agent
from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDeltaEvent,
)
from isotope_core.types import (
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    Usage,
)

# =============================================================================
# Mock provider with word-by-word streaming
# =============================================================================


class StreamingMockProvider:
    """Mock provider that yields text word by word."""

    @property
    def model_name(self) -> str:
        return "mock-streaming"

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
        words = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog."]
        ts = int(time.time() * 1000)

        msg = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=10, output_tokens=len(words)),
            timestamp=ts,
        )

        yield StreamStartEvent(partial=msg)

        full_text = ""
        for word in words:
            if signal and signal.is_set():
                break
            full_text += word
            msg.content[0].text = full_text  # type: ignore[union-attr]
            yield StreamTextDeltaEvent(content_index=0, delta=word, partial=msg)

        msg.content[0].text = full_text  # type: ignore[union-attr]
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    agent = Agent(
        provider=StreamingMockProvider(),
        system_prompt="You are a helpful assistant.",
    )

    print("=== Streaming Example ===\n")
    print("Events:")
    async for event in agent.prompt("Tell me a sentence"):
        match event.type:
            case "agent_start":
                print("  [agent_start]")
            case "turn_start":
                print("  [turn_start]")
            case "message_start":
                print(f"  [message_start] role={event.message.role}")  # type: ignore[union-attr]
            case "message_update":
                delta = event.delta or ""  # type: ignore[union-attr]
                print(f"  [message_update] delta={delta!r}")
            case "message_end":
                print(f"  [message_end] role={event.message.role}")  # type: ignore[union-attr]
            case "turn_end":
                print("  [turn_end]")
            case "agent_end":
                print(f"  [agent_end] reason={event.reason}")  # type: ignore[union-attr]
            case _:
                print(f"  [{event.type}]")

    print("\n--- Reconstructed output ---")
    for msg in agent.messages:
        if msg.role == "assistant":
            for block in msg.content:
                if isinstance(block, TextContent):
                    print(block.text)


if __name__ == "__main__":
    asyncio.run(main())
