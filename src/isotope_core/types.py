"""Core types for isotope-core agent loop.

This module defines all foundational types for the agent loop system using
Pydantic models and Python Protocols.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Content Types
# =============================================================================


class TextContent(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content block with base64 data."""

    type: Literal["image"] = "image"
    data: str  # base64 encoded image data
    mime_type: str  # e.g., "image/jpeg", "image/png"


class ThinkingContent(BaseModel):
    """Reasoning/thinking content block."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False


class ToolCallContent(BaseModel):
    """Tool call content block."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, object]


# Union of all content types
Content = TextContent | ImageContent | ThinkingContent | ToolCallContent


# =============================================================================
# Stop Reasons
# =============================================================================


class StopReason(StrEnum):
    """Reasons why the agent might stop."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"
    ABORTED = "aborted"


# =============================================================================
# Usage Tracking
# =============================================================================


class Usage(BaseModel):
    """Token usage tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class AggregatedUsage(BaseModel):
    """Aggregated usage across multiple providers in a router.

    Tracks total token usage and per-provider / per-model breakdowns.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    provider_usage: dict[str, Usage] = Field(default_factory=dict)
    model_usage: dict[str, Usage] = Field(default_factory=dict)


# =============================================================================
# Message Types
# =============================================================================


class UserMessage(BaseModel):
    """User message."""

    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent]
    timestamp: int  # Unix timestamp in milliseconds
    pinned: bool = False


class AssistantMessage(BaseModel):
    """Assistant message with optional tool calls and metadata."""

    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCallContent]
    stop_reason: StopReason | None = None
    usage: Usage = Field(default_factory=Usage)
    error_message: str | None = None
    timestamp: int  # Unix timestamp in milliseconds
    pinned: bool = False


class ToolResultMessage(BaseModel):
    """Tool result message."""

    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    is_error: bool = False
    timestamp: int  # Unix timestamp in milliseconds
    pinned: bool = False


# Union of all message types
Message = UserMessage | AssistantMessage | ToolResultMessage


# =============================================================================
# Context
# =============================================================================


class ToolSchema(BaseModel):
    """JSON Schema for tool parameters."""

    name: str
    description: str
    parameters: dict[str, object]  # JSON Schema object


class Context(BaseModel):
    """Context for LLM calls."""

    system_prompt: str = ""
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSchema] = Field(default_factory=list)


# =============================================================================
# Agent Events (discriminated union on `type` field)
# =============================================================================


class AgentStartEvent(BaseModel):
    """Emitted when agent starts."""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """Emitted when agent ends."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[Message] = Field(default_factory=list)
    reason: str = "completed"


class TurnStartEvent(BaseModel):
    """Emitted when a turn starts."""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    """Emitted when a turn ends."""

    type: Literal["turn_end"] = "turn_end"
    message: Message
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStartEvent(BaseModel):
    """Emitted when a message starts."""

    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdateEvent(BaseModel):
    """Emitted during message streaming."""

    type: Literal["message_update"] = "message_update"
    message: Message
    delta: str | None = None


class MessageEndEvent(BaseModel):
    """Emitted when a message ends."""

    type: Literal["message_end"] = "message_end"
    message: Message


class ToolStartEvent(BaseModel):
    """Emitted when tool execution starts."""

    type: Literal["tool_start"] = "tool_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, object]


class ToolUpdateEvent(BaseModel):
    """Emitted during tool execution."""

    type: Literal["tool_update"] = "tool_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, object]
    partial_result: object | None = None


class ToolEndEvent(BaseModel):
    """Emitted when tool execution ends."""

    type: Literal["tool_end"] = "tool_end"
    tool_call_id: str
    tool_name: str
    result: object
    is_error: bool = False


class ContextPrunedEvent(BaseModel):
    """Emitted when context is pruned."""

    type: Literal["context_pruned"] = "context_pruned"
    strategy: str
    pruned_count: int
    pruned_tokens: int
    remaining_tokens: int


class SteerEvent(BaseModel):
    """Emitted when a steering message is processed."""

    type: Literal["steer"] = "steer"
    message: Message
    turn_number: int


class FollowUpEvent(BaseModel):
    """Emitted when a follow-up message triggers a new turn."""

    type: Literal["follow_up"] = "follow_up"
    message: Message
    turn_number: int


# Union of all event types (discriminated by type field)
AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolStartEvent
    | ToolUpdateEvent
    | ToolEndEvent
    | ContextPrunedEvent
    | SteerEvent
    | FollowUpEvent
)
