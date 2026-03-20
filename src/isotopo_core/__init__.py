"""isotopo-core - Core primitives for building AI agent loops.

This module exports the public API for isotopo-core.
"""

from isotopo_core.agent import Agent, AgentState
from isotopo_core.events import AgentEventStream, EventStream
from isotopo_core.loop import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
    agent_loop,
)
from isotopo_core.providers.base import (
    Provider,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
    StreamThinkingDeltaEvent,
    StreamThinkingEndEvent,
    StreamToolCallDeltaEvent,
    StreamToolCallEndEvent,
    StreamToolCallStartEvent,
)
from isotopo_core.providers.utils import (
    RetryConfig,
    retry_with_backoff,
)
from isotopo_core.tools import (
    Tool,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolResult,
    ToolValidationError,
    tool,
)
from isotopo_core.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    Content,
    Context,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolEndEvent,
    ToolResultMessage,
    ToolSchema,
    ToolStartEvent,
    ToolUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
    UserMessage,
)

__all__ = [
    # Agent
    "Agent",
    "AgentState",
    # Events
    "EventStream",
    "AgentEventStream",
    # Loop
    "agent_loop",
    "AgentLoopConfig",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallContext",
    "AfterToolCallResult",
    # Provider
    "Provider",
    "StreamEvent",
    "StreamStartEvent",
    "StreamTextDeltaEvent",
    "StreamTextEndEvent",
    "StreamThinkingDeltaEvent",
    "StreamThinkingEndEvent",
    "StreamToolCallStartEvent",
    "StreamToolCallDeltaEvent",
    "StreamToolCallEndEvent",
    "StreamDoneEvent",
    "StreamErrorEvent",
    # Provider Utilities
    "RetryConfig",
    "retry_with_backoff",
    # Tools
    "Tool",
    "ToolResult",
    "tool",
    "ToolError",
    "ToolValidationError",
    "ToolNotFoundError",
    "ToolExecutionError",
    # Types - Content
    "Content",
    "TextContent",
    "ImageContent",
    "ThinkingContent",
    "ToolCallContent",
    # Types - Messages
    "Message",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    # Types - Context
    "Context",
    "ToolSchema",
    "Usage",
    "StopReason",
    # Types - Events
    "AgentEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolStartEvent",
    "ToolUpdateEvent",
    "ToolEndEvent",
]

__version__ = "0.1.0"

# Graceful imports for optional provider dependencies
try:
    from isotopo_core.providers.openai import OpenAIProvider  # noqa: F401

    __all__.append("OpenAIProvider")
except ImportError:
    pass

try:
    from isotopo_core.providers.anthropic import AnthropicProvider, ThinkingConfig  # noqa: F401

    __all__.extend(["AnthropicProvider", "ThinkingConfig"])
except ImportError:
    pass
