"""isotope-core - Core primitives for building AI agent loops.

This module exports the public API for isotope-core.
"""

from isotope_core.agent import Agent, AgentState
from isotope_core.context import (
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
from isotope_core.events import AgentEventStream, EventStream
from isotope_core.loop import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
    TransformContextHook,
    agent_loop,
)
from isotope_core.middleware import (
    EventFilterMiddleware,
    LifecycleHooks,
    LoggingMiddleware,
    Middleware,
    MiddlewareContext,
    OnAgentEndHook,
    OnAgentStartHook,
    OnErrorHook,
    OnTurnEndHook,
    OnTurnStartHook,
    TokenTrackingMiddleware,
    run_middleware_chain,
)
from isotope_core.providers.base import (
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
from isotope_core.providers.router import CircuitState, RouterProvider
from isotope_core.providers.utils import (
    RetryConfig,
    retry_with_backoff,
)
from isotope_core.tools import (
    Tool,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolResult,
    ToolValidationError,
    tool,
)
from isotope_core.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AggregatedUsage,
    AssistantMessage,
    Content,
    Context,
    ContextPrunedEvent,
    FollowUpEvent,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    SteerEvent,
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
    # Context Management
    "MODEL_CONTEXT_WINDOWS",
    "ContextUsage",
    "PruneResult",
    "PruningStrategy",
    "SlidingWindowStrategy",
    "SummarizationStrategy",
    "SelectivePruningStrategy",
    "count_tokens",
    "count_message_tokens",
    "estimate_context_usage",
    "get_context_window",
    "pin_message",
    "unpin_message",
    "create_sliding_window_transform",
    "create_summarization_transform",
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
    "TransformContextHook",
    # Middleware
    "Middleware",
    "MiddlewareContext",
    "run_middleware_chain",
    "LoggingMiddleware",
    "TokenTrackingMiddleware",
    "EventFilterMiddleware",
    "LifecycleHooks",
    "OnAgentStartHook",
    "OnAgentEndHook",
    "OnTurnStartHook",
    "OnTurnEndHook",
    "OnErrorHook",
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
    # Router
    "RouterProvider",
    "CircuitState",
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
    "AggregatedUsage",
    "StopReason",
    # Types - Events
    "AgentEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "ContextPrunedEvent",
    "SteerEvent",
    "FollowUpEvent",
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
    from isotope_core.providers.openai import OpenAIProvider  # noqa: F401

    __all__.append("OpenAIProvider")
except ImportError:
    pass

try:
    from isotope_core.providers.anthropic import AnthropicProvider, ThinkingConfig  # noqa: F401

    __all__.extend(["AnthropicProvider", "ThinkingConfig"])
except ImportError:
    pass

try:
    from isotope_core.providers.proxy import ProxyProvider  # noqa: F401

    __all__.append("ProxyProvider")
except ImportError:
    pass
