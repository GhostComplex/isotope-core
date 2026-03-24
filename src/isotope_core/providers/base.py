"""Provider protocol and stream event types.

This module defines the provider interface that LLM providers must implement,
along with the streaming event types they produce.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from isotope_core.types import Context

# Import AssistantMessage directly (not under TYPE_CHECKING) so Pydantic can resolve it
from isotope_core.types import AssistantMessage

# =============================================================================
# Stream Events (yielded by provider.stream())
# =============================================================================


class StreamStartEvent(BaseModel):
    """Emitted at the start of streaming."""

    type: Literal["start"] = "start"
    partial: AssistantMessage


class StreamTextDeltaEvent(BaseModel):
    """Emitted for each text chunk during streaming."""

    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamTextEndEvent(BaseModel):
    """Emitted when text content block ends."""

    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class StreamThinkingDeltaEvent(BaseModel):
    """Emitted for each thinking chunk during streaming."""

    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamThinkingEndEvent(BaseModel):
    """Emitted when thinking content block ends."""

    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class StreamToolCallStartEvent(BaseModel):
    """Emitted when a tool call starts."""

    type: Literal["tool_call_start"] = "tool_call_start"
    content_index: int
    partial: AssistantMessage


class StreamToolCallDeltaEvent(BaseModel):
    """Emitted for tool call argument chunks."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamToolCallEndEvent(BaseModel):
    """Emitted when a tool call ends."""

    type: Literal["tool_call_end"] = "tool_call_end"
    content_index: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    partial: AssistantMessage


class StreamDoneEvent(BaseModel):
    """Emitted when streaming completes successfully."""

    type: Literal["done"] = "done"
    message: AssistantMessage


class StreamErrorEvent(BaseModel):
    """Emitted when streaming fails."""

    type: Literal["error"] = "error"
    error: AssistantMessage


# Union of all stream event types
StreamEvent = (
    StreamStartEvent
    | StreamTextDeltaEvent
    | StreamTextEndEvent
    | StreamThinkingDeltaEvent
    | StreamThinkingEndEvent
    | StreamToolCallStartEvent
    | StreamToolCallDeltaEvent
    | StreamToolCallEndEvent
    | StreamDoneEvent
    | StreamErrorEvent
)


# =============================================================================
# Provider Protocol
# =============================================================================


@runtime_checkable
class Provider(Protocol):
    """Protocol for LLM providers.

    Providers must implement the stream method which yields StreamEvents
    as an async generator.

    Contract:
    - Must not throw for request/model/runtime failures
    - Failures must be encoded in the stream via StreamErrorEvent
    - The final event must be either StreamDoneEvent or StreamErrorEvent
    - The final AssistantMessage must have appropriate stop_reason
    """

    @property
    def model_name(self) -> str:
        """Return the model identifier used by this provider."""
        ...

    @property
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'openai', 'anthropic', 'proxy', 'router')."""
        ...

    def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from the provider.

        Args:
            context: The conversation context including system prompt, messages, and tools.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            signal: An asyncio.Event that, when set, signals abortion.

        Yields:
            StreamEvent: Events describing the streaming response.
        """
        ...
