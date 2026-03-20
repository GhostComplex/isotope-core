"""Core agent loop implementation.

This module provides the main agent loop that orchestrates LLM interactions
with tool execution.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from isotopo_core.providers.base import Provider
from isotopo_core.tools import Tool, ToolResult
from isotopo_core.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StopReason,
    TextContent,
    ToolCallContent,
    ToolEndEvent,
    ToolResultMessage,
    ToolStartEvent,
    ToolUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

# =============================================================================
# Configuration Types
# =============================================================================


@dataclass
class BeforeToolCallContext:
    """Context passed to before_tool_call hook."""

    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: dict[str, Any]
    context: Context


@dataclass
class BeforeToolCallResult:
    """Result from before_tool_call hook."""

    block: bool = False
    reason: str | None = None


@dataclass
class AfterToolCallContext:
    """Context passed to after_tool_call hook."""

    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: dict[str, Any]
    result: ToolResult
    is_error: bool
    context: Context


@dataclass
class AfterToolCallResult:
    """Result from after_tool_call hook."""

    content: list[TextContent | ImageContent] | None = None
    is_error: bool | None = None


BeforeToolCallHook = Callable[
    [BeforeToolCallContext, asyncio.Event | None],
    Awaitable[BeforeToolCallResult | None],
]

AfterToolCallHook = Callable[
    [AfterToolCallContext, asyncio.Event | None],
    Awaitable[AfterToolCallResult | None],
]

TransformContextHook = Callable[
    [list[Message], asyncio.Event | None],
    Awaitable[list[Message]],
]


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    provider: Provider
    tools: list[Tool] = field(default_factory=list)
    tool_execution: Literal["parallel", "sequential"] = "parallel"
    temperature: float | None = None
    max_tokens: int | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    transform_context: TransformContextHook | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _create_error_tool_result(message: str) -> ToolResult:
    """Create an error ToolResult with a text message."""
    return ToolResult(content=[TextContent(text=message)], is_error=True)


def _find_tool(tools: list[Tool], name: str) -> Tool | None:
    """Find a tool by name."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None


# =============================================================================
# Agent Loop
# =============================================================================


async def agent_loop(
    prompts: list[Message],
    context: Context,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Run the agent loop.

    This is the core execution engine that:
    1. Emits agent_start
    2. For each turn:
       - Emits turn_start
       - Streams user messages (if any)
       - Streams assistant response
       - Executes tool calls (if any)
       - Emits turn_end
    3. Repeats if there are tool calls
    4. Emits agent_end

    Args:
        prompts: Initial messages to add to the context.
        context: The conversation context.
        config: Loop configuration including provider and tools.
        signal: Optional abort signal (asyncio.Event).

    Yields:
        AgentEvent: Events describing the agent's progress.
    """
    # Build the current context
    current_messages = list(context.messages) + list(prompts)
    new_messages: list[Message] = list(prompts)

    # Emit agent_start
    yield AgentStartEvent()

    # Emit turn_start
    yield TurnStartEvent()

    # Emit message events for the initial prompts
    for prompt in prompts:
        yield MessageStartEvent(message=prompt)
        yield MessageEndEvent(message=prompt)

    # Main loop
    while True:
        # Check for abort
        if signal and signal.is_set():
            error_message = AssistantMessage(
                content=[TextContent(text="")],
                stop_reason=StopReason.ABORTED,
                error_message="Aborted by user",
                timestamp=int(time.time() * 1000),
            )
            yield MessageStartEvent(message=error_message)
            yield MessageEndEvent(message=error_message)
            yield TurnEndEvent(message=error_message, tool_results=[])
            yield AgentEndEvent(messages=new_messages + [error_message])
            return

        # Apply context transform if configured
        messages_for_llm = current_messages
        if config.transform_context:
            messages_for_llm = await config.transform_context(current_messages, signal)

        # Build the context for the LLM call
        llm_context = Context(
            system_prompt=context.system_prompt,
            messages=messages_for_llm,
            tools=[
                tool.to_schema()  # type: ignore[misc]
                for tool in config.tools
            ],
        )

        # Stream assistant response
        assistant_message: AssistantMessage | None = None
        message_started = False

        async for event in config.provider.stream(
            llm_context,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            signal=signal,
        ):
            if event.type == "start":
                assistant_message = event.partial
                yield MessageStartEvent(message=assistant_message)
                message_started = True

            elif event.type in (
                "text_delta",
                "text_end",
                "thinking_delta",
                "thinking_end",
                "tool_call_start",
                "tool_call_delta",
                "tool_call_end",
            ):
                if hasattr(event, "partial"):
                    assistant_message = event.partial
                    delta = getattr(event, "delta", None)
                    yield MessageUpdateEvent(message=assistant_message, delta=delta)

            elif event.type == "done":
                assistant_message = event.message
                if not message_started:
                    yield MessageStartEvent(message=assistant_message)
                yield MessageEndEvent(message=assistant_message)

            elif event.type == "error":
                assistant_message = event.error
                if not message_started:
                    yield MessageStartEvent(message=assistant_message)
                yield MessageEndEvent(message=assistant_message)

        if assistant_message is None:
            # Provider didn't yield any events - create an error message
            assistant_message = AssistantMessage(
                content=[TextContent(text="")],
                stop_reason=StopReason.ERROR,
                error_message="No response from provider",
                timestamp=int(time.time() * 1000),
            )
            yield MessageStartEvent(message=assistant_message)
            yield MessageEndEvent(message=assistant_message)

        # Add assistant message to context
        current_messages.append(assistant_message)
        new_messages.append(assistant_message)

        # Check for error or abort
        if assistant_message.stop_reason in (StopReason.ERROR, StopReason.ABORTED):
            yield TurnEndEvent(message=assistant_message, tool_results=[])
            yield AgentEndEvent(messages=new_messages)
            return

        # Extract tool calls
        tool_calls = [
            content
            for content in assistant_message.content
            if isinstance(content, ToolCallContent)
        ]

        tool_results: list[ToolResultMessage] = []

        if tool_calls:
            # Execute tool calls
            if config.tool_execution == "sequential":
                for tool_call in tool_calls:
                    result = await _execute_tool_call(
                        tool_call,
                        assistant_message,
                        Context(
                            system_prompt=context.system_prompt,
                            messages=current_messages,
                            tools=llm_context.tools,
                        ),
                        config,
                        signal,
                    )
                    async for tool_event in result[0]:
                        yield tool_event
                    tool_results.append(result[1])
            else:
                # Parallel execution
                # First emit all tool_start events
                for tool_call in tool_calls:
                    yield ToolStartEvent(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        args=tool_call.arguments,
                    )

                # Execute in parallel
                tasks = [
                    _execute_tool_call_inner(
                        tool_call,
                        assistant_message,
                        Context(
                            system_prompt=context.system_prompt,
                            messages=current_messages,
                            tools=llm_context.tools,
                        ),
                        config,
                        signal,
                    )
                    for tool_call in tool_calls
                ]
                results = await asyncio.gather(*tasks)

                # Emit results in order
                for _tool_call, (events, result_msg) in zip(
                    tool_calls, results, strict=True
                ):
                    for tool_event in events:
                        yield tool_event
                    tool_results.append(result_msg)

            # Add tool results to context
            for result_msg in tool_results:
                current_messages.append(result_msg)
                new_messages.append(result_msg)
                yield MessageStartEvent(message=result_msg)
                yield MessageEndEvent(message=result_msg)

        # Emit turn_end
        yield TurnEndEvent(message=assistant_message, tool_results=tool_results)

        # Check if we should continue
        if not tool_calls:
            # No tool calls - we're done
            yield AgentEndEvent(messages=new_messages)
            return

        # Start a new turn for tool results
        yield TurnStartEvent()


async def _execute_tool_call(
    tool_call: ToolCallContent,
    assistant_message: AssistantMessage,
    context: Context,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> tuple[AsyncGenerator[AgentEvent, None], ToolResultMessage]:
    """Execute a single tool call with events."""

    async def event_gen() -> AsyncGenerator[AgentEvent, None]:
        yield ToolStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        )

    events, result = await _execute_tool_call_inner(
        tool_call, assistant_message, context, config, signal
    )

    async def combined_gen() -> AsyncGenerator[AgentEvent, None]:
        async for e in event_gen():
            yield e
        for e in events:
            yield e

    return combined_gen(), result


async def _execute_tool_call_inner(
    tool_call: ToolCallContent,
    assistant_message: AssistantMessage,
    context: Context,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> tuple[list[AgentEvent], ToolResultMessage]:
    """Execute a tool call and return events and result."""
    events: list[AgentEvent] = []
    timestamp = int(time.time() * 1000)

    # Find the tool
    tool = _find_tool(config.tools, tool_call.name)
    if tool is None:
        result = _create_error_tool_result(f"Tool '{tool_call.name}' not found")
        events.append(
            ToolEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result={"content": [c.model_dump() for c in result.content]},
                is_error=True,
            )
        )
        return events, ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            is_error=True,
            timestamp=timestamp,
        )

    # Validate arguments
    valid, error = tool.validate_arguments(tool_call.arguments)
    if not valid:
        result = _create_error_tool_result(f"Invalid arguments: {error}")
        events.append(
            ToolEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result={"content": [c.model_dump() for c in result.content]},
                is_error=True,
            )
        )
        return events, ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            is_error=True,
            timestamp=timestamp,
        )

    # Call before_tool_call hook if configured
    if config.before_tool_call:
        try:
            before_result = await config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=tool_call.arguments,
                    context=context,
                ),
                signal,
            )
            if before_result and before_result.block:
                reason = before_result.reason or "Tool execution was blocked"
                result = _create_error_tool_result(reason)
                events.append(
                    ToolEndEvent(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result={"content": [c.model_dump() for c in result.content]},
                        is_error=True,
                    )
                )
                return events, ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=result.content,
                    is_error=True,
                    timestamp=timestamp,
                )
        except Exception as e:
            result = _create_error_tool_result(f"before_tool_call hook error: {e}")
            events.append(
                ToolEndEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result={"content": [c.model_dump() for c in result.content]},
                    is_error=True,
                )
            )
            return events, ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=result.content,
                is_error=True,
                timestamp=timestamp,
            )

    # Execute the tool
    is_error = False
    try:

        def on_update(partial_result: ToolResult) -> None:
            events.append(
                ToolUpdateEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                    partial_result={
                        "content": [c.model_dump() for c in partial_result.content]
                    },
                )
            )

        result = await tool._execute(
            tool_call.id, tool_call.arguments, signal, on_update
        )
        is_error = result.is_error
    except Exception as e:
        result = _create_error_tool_result(f"Tool execution error: {e}")
        is_error = True

    # Call after_tool_call hook if configured
    if config.after_tool_call:
        try:
            after_result = await config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=tool_call.arguments,
                    result=result,
                    is_error=is_error,
                    context=context,
                ),
                signal,
            )
            if after_result:
                if after_result.content is not None:
                    result = ToolResult(
                        content=after_result.content, is_error=result.is_error
                    )
                if after_result.is_error is not None:
                    is_error = after_result.is_error
        except Exception:
            # Don't let after_tool_call errors break the loop
            pass

    # Emit tool_end event
    events.append(
        ToolEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result={"content": [c.model_dump() for c in result.content]},
            is_error=is_error,
        )
    )

    return events, ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        is_error=is_error,
        timestamp=timestamp,
    )
