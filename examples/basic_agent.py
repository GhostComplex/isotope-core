"""Basic agent example — minimal agent with a tool.

This example demonstrates how to create an agent with a single tool
and process a user prompt. Uses a mock provider so no API key is needed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from isotope_core import Agent
from isotope_core.providers.base import (
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDeltaEvent,
)
from isotope_core.tools import Tool, ToolResult
from isotope_core.types import (
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    ToolCallContent,
    Usage,
)

# =============================================================================
# Mock provider — no real API calls
# =============================================================================


class MockProvider:
    """A mock provider that returns canned responses."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-model"

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
        self.call_count += 1
        ts = int(time.time() * 1000)

        if self.call_count == 1:
            # First call: the model wants to use the tool
            msg = AssistantMessage(
                content=[
                    TextContent(text="Let me read that file for you."),
                    ToolCallContent(
                        id="call_1", name="read_file",
                        arguments={"path": "/tmp/hello.txt"},
                    ),
                ],
                stop_reason=StopReason.TOOL_USE,
                usage=Usage(input_tokens=20, output_tokens=15),
                timestamp=ts,
            )
        else:
            # Second call: final answer after tool result
            msg = AssistantMessage(
                content=[TextContent(text="The file contains: Hello, World!")],
                stop_reason=StopReason.END_TURN,
                usage=Usage(input_tokens=30, output_tokens=10),
                timestamp=ts,
            )

        yield StreamStartEvent(partial=msg)
        for block in msg.content:
            if isinstance(block, TextContent):
                yield StreamTextDeltaEvent(content_index=0, delta=block.text, partial=msg)
        yield StreamDoneEvent(message=msg)


# =============================================================================
# Tool definition
# =============================================================================


async def read_file_impl(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> ToolResult:
    """Mock implementation of read_file."""
    path = params.get("path", "")
    return ToolResult.text(f"Contents of {path}: Hello, World!")


read_file = Tool(
    name="read_file",
    description="Read a file from disk",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path to read"}},
        "required": ["path"],
    },
    execute=read_file_impl,
)


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    agent = Agent(
        provider=MockProvider(),
        system_prompt="You are a helpful coding assistant.",
        tools=[read_file],
    )

    print("=== Basic Agent Example ===\n")
    async for event in agent.prompt("Read the file /tmp/hello.txt"):
        if event.type == "message_start":
            role = getattr(event.message, "role", "?")  # type: ignore[union-attr]
            print(f"[{event.type}] role={role}")
        elif event.type == "message_update" and event.delta:  # type: ignore[union-attr]
            print(event.delta, end="")  # type: ignore[union-attr]
        elif event.type == "tool_start":
            print(f"\n[tool_start] {event.tool_name}({event.args})")  # type: ignore[union-attr]
        elif event.type == "tool_end":
            print(f"[tool_end] error={event.is_error}")  # type: ignore[union-attr]
        elif event.type == "agent_end":
            print(f"\n[agent_end] reason={event.reason}")  # type: ignore[union-attr]

    print("\n\nFinal messages:", len(agent.messages))


if __name__ == "__main__":
    asyncio.run(main())
