"""Stateful Agent class wrapping the agent loop.

This module provides the Agent class that manages state and provides
a high-level interface for interacting with the agent loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field

from isotopo_core.loop import (
    AfterToolCallHook,
    AgentLoopConfig,
    BeforeToolCallHook,
    TransformContextHook,
    agent_loop,
)
from isotopo_core.providers.base import Provider
from isotopo_core.tools import Tool
from isotopo_core.types import (
    AgentEvent,
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    UserMessage,
)

# =============================================================================
# Agent State
# =============================================================================


@dataclass
class AgentState:
    """State of the agent."""

    system_prompt: str = ""
    provider: Provider | None = None
    tools: list[Tool] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: Message | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


# =============================================================================
# Agent Class
# =============================================================================


class Agent:
    """Stateful agent wrapping the agent loop.

    The Agent class provides:
    - State management for messages, tools, and configuration
    - High-level prompt() and continue_() methods
    - Abort support via asyncio.Event
    - Subscription system for event notifications

    Example:
        agent = Agent(provider=my_provider)
        agent.set_system_prompt("You are a helpful assistant.")
        agent.set_tools([my_tool])

        async for event in agent.prompt("Hello!"):
            print(event)
    """

    def __init__(
        self,
        provider: Provider | None = None,
        system_prompt: str = "",
        tools: list[Tool] | None = None,
        tool_execution: str = "parallel",
        temperature: float | None = None,
        max_tokens: int | None = None,
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
        transform_context: TransformContextHook | None = None,
    ):
        """Initialize the Agent.

        Args:
            provider: The LLM provider to use.
            system_prompt: Initial system prompt.
            tools: Initial list of tools.
            tool_execution: "parallel" or "sequential" tool execution.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            before_tool_call: Hook called before each tool execution.
            after_tool_call: Hook called after each tool execution.
            transform_context: Hook to transform context before LLM calls.
        """
        self._state = AgentState(
            system_prompt=system_prompt,
            provider=provider,
            tools=tools or [],
        )
        self._tool_execution = tool_execution
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._transform_context = transform_context

        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._abort_signal: asyncio.Event | None = None
        self._running_task: asyncio.Task[None] | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> AgentState:
        """Get the current agent state."""
        return self._state

    @property
    def messages(self) -> list[Message]:
        """Get the current message history."""
        return self._state.messages

    @property
    def is_streaming(self) -> bool:
        """Check if the agent is currently streaming."""
        return self._state.is_streaming

    # =========================================================================
    # State Mutators
    # =========================================================================

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._state.system_prompt = prompt

    def set_provider(self, provider: Provider) -> None:
        """Set the LLM provider."""
        self._state.provider = provider

    def set_tools(self, tools: list[Tool]) -> None:
        """Set the available tools."""
        self._state.tools = tools

    def replace_messages(self, messages: list[Message]) -> None:
        """Replace the entire message history."""
        self._state.messages = list(messages)

    def append_message(self, message: Message) -> None:
        """Append a message to the history."""
        self._state.messages.append(message)

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._state.messages = []

    def reset(self) -> None:
        """Reset the agent state."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None

    # =========================================================================
    # Subscription
    # =========================================================================

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to agent events.

        Args:
            callback: Function called for each event.

        Returns:
            An unsubscribe function.
        """
        self._listeners.append(callback)

        def unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return unsubscribe

    def _emit(self, event: AgentEvent) -> None:
        """Emit an event to all listeners."""
        for listener in self._listeners:
            with contextlib.suppress(Exception):
                listener(event)

    # =========================================================================
    # Abort
    # =========================================================================

    def abort(self) -> None:
        """Abort the current operation."""
        if self._abort_signal:
            self._abort_signal.set()

    # =========================================================================
    # Prompt Methods
    # =========================================================================

    async def prompt(
        self,
        text: str | None = None,
        images: list[ImageContent] | None = None,
        messages: list[Message] | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Send a prompt to the agent.

        Args:
            text: Text content for a user message.
            images: Optional images to include.
            messages: Alternatively, provide raw messages.

        Yields:
            AgentEvent: Events from the agent loop.

        Raises:
            ValueError: If neither text nor messages is provided.
            RuntimeError: If agent is already streaming or no provider is set.
        """
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing. Use abort() first.")

        if self._state.provider is None:
            raise RuntimeError("No provider configured. Call set_provider() first.")

        # Build the prompt messages
        if messages is not None:
            prompt_messages = messages
        elif text is not None:
            content: list[TextContent | ImageContent] = [TextContent(text=text)]
            if images:
                content.extend(images)
            prompt_messages = [UserMessage(content=content, timestamp=int(time.time() * 1000))]
        else:
            raise ValueError("Either text or messages must be provided")

        async for event in self._run_loop(prompt_messages):
            yield event

    async def continue_(self) -> AsyncGenerator[AgentEvent, None]:
        """Continue from the current context.

        Used for retries or resuming after tool results.

        Yields:
            AgentEvent: Events from the agent loop.

        Raises:
            RuntimeError: If agent is already streaming, no messages exist,
                          or the last message is from the assistant.
        """
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing.")

        if self._state.provider is None:
            raise RuntimeError("No provider configured.")

        if not self._state.messages:
            raise RuntimeError("No messages to continue from.")

        last_message = self._state.messages[-1]
        if isinstance(last_message, AssistantMessage):
            raise RuntimeError(
                "Cannot continue from an assistant message. "
                "Add a user message or tool result first."
            )

        async for event in self._run_loop([]):
            yield event

    # =========================================================================
    # Internal Loop
    # =========================================================================

    async def _run_loop(self, prompts: list[Message]) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent loop with the given prompts."""
        if self._state.provider is None:
            raise RuntimeError("No provider configured")

        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None
        self._abort_signal = asyncio.Event()

        try:
            context = Context(
                system_prompt=self._state.system_prompt,
                messages=list(self._state.messages),
                tools=[t.to_schema() for t in self._state.tools],  # type: ignore[misc]
            )

            config = AgentLoopConfig(
                provider=self._state.provider,
                tools=self._state.tools,
                tool_execution=self._tool_execution,  # type: ignore[arg-type]
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                before_tool_call=self._before_tool_call,
                after_tool_call=self._after_tool_call,
                transform_context=self._transform_context,
            )

            async for event in agent_loop(prompts, context, config, self._abort_signal):
                # Update state based on event type
                self._process_event(event)

                # Emit to listeners
                self._emit(event)

                # Yield to caller
                yield event

        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._abort_signal = None

    def _process_event(self, event: AgentEvent) -> None:
        """Process an event and update internal state."""
        event_type = event.type

        if event_type == "message_start" or event_type == "message_update":
            self._state.stream_message = event.message  # type: ignore[union-attr]

        elif event_type == "message_end":
            self._state.stream_message = None
            self._state.messages.append(event.message)  # type: ignore[union-attr]

        elif event_type == "tool_start":
            self._state.pending_tool_calls.add(event.tool_call_id)  # type: ignore[union-attr]

        elif event_type == "tool_end":
            self._state.pending_tool_calls.discard(event.tool_call_id)  # type: ignore[union-attr]

        elif event_type == "turn_end":
            msg = event.message  # type: ignore[union-attr]
            if isinstance(msg, AssistantMessage) and msg.error_message:
                self._state.error = msg.error_message

        elif event_type == "agent_end":
            self._state.is_streaming = False
            self._state.stream_message = None
