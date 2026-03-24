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

from isotope_core.middleware import (
    LifecycleHooks,
    MiddlewareContext,
    run_middleware_chain,
)
from isotope_core.providers.base import Provider
from isotope_core.tools import Tool, ToolResult
from isotope_core.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    Context,
    FollowUpEvent,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    SteerEvent,
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
    steering_queue: asyncio.Queue[Message] | None = None
    follow_up_queue: asyncio.Queue[Message] | None = None
    max_turns: int | None = None
    max_total_tokens: int | None = None
    middleware: list[Any] | None = None  # list[Middleware]
    lifecycle_hooks: LifecycleHooks | None = None


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
       - Checks budget limits (turn count, total tokens)
       - Emits turn_start
       - Streams user messages (if any)
       - Streams assistant response
       - Executes tool calls (if any)
       - Checks steering queue after tool execution
       - Checks follow-up queue when no tool calls remain
       - Emits turn_end
    3. Repeats if there are tool calls, steering, or follow-up messages
    4. Emits agent_end with reason

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

    # Budget tracking
    turn_number = 0
    cumulative_tokens = 0

    # Middleware chain setup (built once per loop run)
    mw_list: list[Any] = config.middleware or []
    hooks = config.lifecycle_hooks

    # Middleware context (mutated in-place as state changes)
    mw_ctx = MiddlewareContext(
        messages=current_messages,
        turn_number=turn_number,
        cumulative_tokens=cumulative_tokens,
        agent_config=config,
    )

    async def _emit(event: AgentEvent) -> AgentEvent | None:
        """Run an event through the middleware chain."""
        if mw_list:
            return await run_middleware_chain(event, mw_ctx, mw_list)
        return event

    async def _call_hook(hook: Callable[..., Awaitable[None]], *args: Any) -> None:
        """Call a lifecycle hook, catching and suppressing errors."""
        try:
            await hook(*args)
        except Exception:
            import logging as _logging

            _logging.getLogger(__name__).exception("Lifecycle hook error")

    # Emit agent_start
    start_evt = await _emit(AgentStartEvent())
    if start_evt is not None:
        yield start_evt
    if hooks and hooks.on_agent_start:
        await _call_hook(hooks.on_agent_start)

    # Emit turn_start
    ts_evt = await _emit(TurnStartEvent())
    if ts_evt is not None:
        yield ts_evt

    # Emit message events for the initial prompts
    for prompt in prompts:
        ms_evt = await _emit(MessageStartEvent(message=prompt))
        if ms_evt is not None:
            yield ms_evt
        me_evt = await _emit(MessageEndEvent(message=prompt))
        if me_evt is not None:
            yield me_evt

    # Main loop
    while True:
        turn_number += 1
        mw_ctx.turn_number = turn_number

        # Invoke on_turn_start hook
        if hooks and hooks.on_turn_start:
            await _call_hook(hooks.on_turn_start, turn_number)

        # Check budget: max turns
        if config.max_turns is not None and turn_number > config.max_turns:
            te_evt = await _emit(
                TurnEndEvent(
                    message=AssistantMessage(
                        content=[TextContent(text="")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    ),
                    tool_results=[],
                )
            )
            if te_evt is not None:
                yield te_evt
            ae_evt = await _emit(AgentEndEvent(messages=new_messages, reason="max_turns"))
            if ae_evt is not None:
                yield ae_evt
            if hooks and hooks.on_agent_end:
                await _call_hook(hooks.on_agent_end, "max_turns")
            return

        # Check budget: max total tokens
        if config.max_total_tokens is not None and cumulative_tokens >= config.max_total_tokens:
            te_evt = await _emit(
                TurnEndEvent(
                    message=AssistantMessage(
                        content=[TextContent(text="")],
                        stop_reason=StopReason.END_TURN,
                        timestamp=int(time.time() * 1000),
                    ),
                    tool_results=[],
                )
            )
            if te_evt is not None:
                yield te_evt
            ae_evt = await _emit(AgentEndEvent(messages=new_messages, reason="max_budget"))
            if ae_evt is not None:
                yield ae_evt
            if hooks and hooks.on_agent_end:
                await _call_hook(hooks.on_agent_end, "max_budget")
            return

        # Check for abort
        if signal and signal.is_set():
            error_message = AssistantMessage(
                content=[TextContent(text="")],
                stop_reason=StopReason.ABORTED,
                error_message="Aborted by user",
                timestamp=int(time.time() * 1000),
            )
            ms_evt = await _emit(MessageStartEvent(message=error_message))
            if ms_evt is not None:
                yield ms_evt
            me_evt = await _emit(MessageEndEvent(message=error_message))
            if me_evt is not None:
                yield me_evt
            te_evt = await _emit(TurnEndEvent(message=error_message, tool_results=[]))
            if te_evt is not None:
                yield te_evt
            ae_evt = await _emit(
                AgentEndEvent(messages=new_messages + [error_message], reason="aborted")
            )
            if ae_evt is not None:
                yield ae_evt
            if hooks and hooks.on_agent_end:
                await _call_hook(hooks.on_agent_end, "aborted")
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

        try:
            async for event in config.provider.stream(
                llm_context,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                signal=signal,
            ):
                if event.type == "start":
                    assistant_message = event.partial
                    ms_evt = await _emit(MessageStartEvent(message=assistant_message))
                    if ms_evt is not None:
                        yield ms_evt
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
                        mu_evt = await _emit(
                            MessageUpdateEvent(message=assistant_message, delta=delta)
                        )
                        if mu_evt is not None:
                            yield mu_evt

                elif event.type == "done":
                    assistant_message = event.message
                    if not message_started:
                        ms_evt = await _emit(MessageStartEvent(message=assistant_message))
                        if ms_evt is not None:
                            yield ms_evt
                    me_evt = await _emit(MessageEndEvent(message=assistant_message))
                    if me_evt is not None:
                        yield me_evt

                elif event.type == "error":
                    assistant_message = event.error
                    if not message_started:
                        ms_evt = await _emit(MessageStartEvent(message=assistant_message))
                        if ms_evt is not None:
                            yield ms_evt
                    me_evt = await _emit(MessageEndEvent(message=assistant_message))
                    if me_evt is not None:
                        yield me_evt
        except Exception as exc:
            if hooks and hooks.on_error:
                await _call_hook(hooks.on_error, exc)
            raise

        if assistant_message is None:
            # Provider didn't yield any events - create an error message
            assistant_message = AssistantMessage(
                content=[TextContent(text="")],
                stop_reason=StopReason.ERROR,
                error_message="No response from provider",
                timestamp=int(time.time() * 1000),
            )
            ms_evt = await _emit(MessageStartEvent(message=assistant_message))
            if ms_evt is not None:
                yield ms_evt
            me_evt = await _emit(MessageEndEvent(message=assistant_message))
            if me_evt is not None:
                yield me_evt

        # Track cumulative tokens
        cumulative_tokens += assistant_message.usage.total_tokens
        mw_ctx.cumulative_tokens = cumulative_tokens

        # Add assistant message to context
        current_messages.append(assistant_message)
        new_messages.append(assistant_message)

        # Check for error or abort
        if assistant_message.stop_reason == StopReason.ERROR:
            te_evt = await _emit(TurnEndEvent(message=assistant_message, tool_results=[]))
            if te_evt is not None:
                yield te_evt
            if hooks and hooks.on_turn_end:
                await _call_hook(hooks.on_turn_end, turn_number, assistant_message)
            if hooks and hooks.on_error:
                await _call_hook(
                    hooks.on_error,
                    Exception(assistant_message.error_message or "Unknown error"),
                )
            ae_evt = await _emit(AgentEndEvent(messages=new_messages, reason="error"))
            if ae_evt is not None:
                yield ae_evt
            if hooks and hooks.on_agent_end:
                await _call_hook(hooks.on_agent_end, "error")
            return

        if assistant_message.stop_reason == StopReason.ABORTED:
            te_evt = await _emit(TurnEndEvent(message=assistant_message, tool_results=[]))
            if te_evt is not None:
                yield te_evt
            if hooks and hooks.on_turn_end:
                await _call_hook(hooks.on_turn_end, turn_number, assistant_message)
            ae_evt = await _emit(AgentEndEvent(messages=new_messages, reason="aborted"))
            if ae_evt is not None:
                yield ae_evt
            if hooks and hooks.on_agent_end:
                await _call_hook(hooks.on_agent_end, "aborted")
            return

        # Extract tool calls
        tool_calls = [
            content for content in assistant_message.content if isinstance(content, ToolCallContent)
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
                        te_out = await _emit(tool_event)
                        if te_out is not None:
                            yield te_out
                    tool_results.append(result[1])
            else:
                # Parallel execution
                # First emit all tool_start events
                for tool_call in tool_calls:
                    ts_out = await _emit(
                        ToolStartEvent(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            args=tool_call.arguments,
                        )
                    )
                    if ts_out is not None:
                        yield ts_out

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
                for _tool_call, (events, result_msg) in zip(tool_calls, results, strict=True):
                    for tool_event in events:
                        te_out = await _emit(tool_event)
                        if te_out is not None:
                            yield te_out
                    tool_results.append(result_msg)

            # Add tool results to context
            for result_msg in tool_results:
                current_messages.append(result_msg)
                new_messages.append(result_msg)
                ms_evt = await _emit(MessageStartEvent(message=result_msg))
                if ms_evt is not None:
                    yield ms_evt
                me_evt = await _emit(MessageEndEvent(message=result_msg))
                if me_evt is not None:
                    yield me_evt

        # Emit turn_end
        te_evt = await _emit(TurnEndEvent(message=assistant_message, tool_results=tool_results))
        if te_evt is not None:
            yield te_evt
        if hooks and hooks.on_turn_end:
            await _call_hook(hooks.on_turn_end, turn_number, assistant_message)

        # After tool execution: check steering queue
        if config.steering_queue is not None:
            steered = False
            while not config.steering_queue.empty():
                try:
                    steer_msg = config.steering_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                # Append steering message to context
                current_messages.append(steer_msg)
                new_messages.append(steer_msg)
                se_evt = await _emit(SteerEvent(message=steer_msg, turn_number=turn_number))
                if se_evt is not None:
                    yield se_evt
                ms_evt = await _emit(MessageStartEvent(message=steer_msg))
                if ms_evt is not None:
                    yield ms_evt
                me_evt = await _emit(MessageEndEvent(message=steer_msg))
                if me_evt is not None:
                    yield me_evt
                steered = True

            if steered:
                # Start a new turn for the steered context
                ts_evt = await _emit(TurnStartEvent())
                if ts_evt is not None:
                    yield ts_evt
                continue

        # Check if we should continue
        if tool_calls:
            # Had tool calls — start a new turn for tool results
            ts_evt = await _emit(TurnStartEvent())
            if ts_evt is not None:
                yield ts_evt
            continue

        # No tool calls — check follow-up queue before ending
        if config.follow_up_queue is not None and not config.follow_up_queue.empty():
            followed_up = False
            while not config.follow_up_queue.empty():
                try:
                    followup_msg = config.follow_up_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                # Append follow-up message to context
                current_messages.append(followup_msg)
                new_messages.append(followup_msg)
                fu_evt = await _emit(
                    FollowUpEvent(message=followup_msg, turn_number=turn_number)
                )
                if fu_evt is not None:
                    yield fu_evt
                ms_evt = await _emit(MessageStartEvent(message=followup_msg))
                if ms_evt is not None:
                    yield ms_evt
                me_evt = await _emit(MessageEndEvent(message=followup_msg))
                if me_evt is not None:
                    yield me_evt
                followed_up = True

            if followed_up:
                # Start a new turn for the follow-up
                ts_evt = await _emit(TurnStartEvent())
                if ts_evt is not None:
                    yield ts_evt
                continue

        # No tool calls, no steering, no follow-up — we're done
        ae_evt = await _emit(AgentEndEvent(messages=new_messages, reason="completed"))
        if ae_evt is not None:
            yield ae_evt
        if hooks and hooks.on_agent_end:
            await _call_hook(hooks.on_agent_end, "completed")
        return


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
                    partial_result={"content": [c.model_dump() for c in partial_result.content]},
                )
            )

        result = await tool._execute(tool_call.id, tool_call.arguments, signal, on_update)
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
                    result = ToolResult(content=after_result.content, is_error=result.is_error)
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
