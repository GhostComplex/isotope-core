"""OpenAI provider implementation.

This module provides the OpenAI provider that implements the Provider protocol
for OpenAI's Chat Completions API with streaming.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from isotope_core.providers.base import (
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
from isotope_core.providers.utils import (
    create_error_message,
    current_timestamp_ms,
)
from isotope_core.types import (
    AssistantMessage,
    Context,
    ImageContent,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultMessage,
    ToolSchema,
    Usage,
    UserMessage,
)

if TYPE_CHECKING:
    import openai


def _parse_streaming_json(partial_json: str) -> dict[str, Any]:
    """Parse potentially incomplete JSON from streaming.

    Args:
        partial_json: Potentially incomplete JSON string.

    Returns:
        Parsed dictionary, or empty dict if parsing fails.
    """
    if not partial_json:
        return {}

    # Try parsing as-is first
    try:
        return dict(json.loads(partial_json))
    except json.JSONDecodeError:
        pass

    # Try to repair common incomplete JSON patterns
    repaired = partial_json.rstrip()

    # Count open braces/brackets
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    # Add missing closing characters
    repaired += "]" * open_brackets
    repaired += "}" * open_braces

    try:
        return dict(json.loads(repaired))
    except json.JSONDecodeError:
        return {}


def _convert_context_to_openai(
    context: Context,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Convert our Context to OpenAI message format.

    Args:
        context: Our Context object.

    Returns:
        Tuple of (messages, tools) in OpenAI format.
    """
    messages: list[dict[str, Any]] = []

    # Add system prompt as first message
    if context.system_prompt:
        messages.append({"role": "system", "content": context.system_prompt})

    # Convert messages
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            messages.append(_convert_user_message(msg))
        elif isinstance(msg, AssistantMessage):
            converted = _convert_assistant_message(msg)
            if converted:  # Skip empty assistant messages
                messages.append(converted)
        elif isinstance(msg, ToolResultMessage):
            messages.append(_convert_tool_result_message(msg))

    # Convert tools
    tools = _convert_tools(context.tools) if context.tools else None

    return messages, tools


def _convert_user_message(msg: UserMessage) -> dict[str, Any]:
    """Convert UserMessage to OpenAI format."""
    content: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextContent):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )

    # OpenAI accepts string content for text-only messages
    if len(content) == 1 and content[0]["type"] == "text":
        return {"role": "user", "content": content[0]["text"]}

    return {"role": "user", "content": content}


def _convert_assistant_message(msg: AssistantMessage) -> dict[str, Any] | None:
    """Convert AssistantMessage to OpenAI format."""
    result: dict[str, Any] = {"role": "assistant"}

    # Extract text content
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextContent):
            if block.text.strip():
                text_parts.append(block.text)
        elif isinstance(block, ThinkingContent):
            # OpenAI doesn't have native thinking support, but some
            # compatible APIs support reasoning_content field
            pass
        elif isinstance(block, ToolCallContent):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    },
                }
            )

    # Set content (None if only tool calls)
    if text_parts:
        result["content"] = "".join(text_parts)
    else:
        result["content"] = None

    # Add tool calls if present
    if tool_calls:
        result["tool_calls"] = tool_calls

    # Skip empty assistant messages (no content and no tool calls)
    if result["content"] is None and not tool_calls:
        return None

    return result


def _convert_tool_result_message(msg: ToolResultMessage) -> dict[str, Any]:
    """Convert ToolResultMessage to OpenAI format."""
    # Extract text content
    text_parts: list[str] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
        elif isinstance(block, ImageContent):
            # OpenAI doesn't support images in tool results directly
            text_parts.append("(image attachment)")

    return {
        "role": "tool",
        "tool_call_id": msg.tool_call_id,
        "content": "\n".join(text_parts) if text_parts else "",
    }


def _convert_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert our ToolSchema list to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _map_finish_reason(reason: str | None) -> StopReason:
    """Map OpenAI finish_reason to our StopReason."""
    if reason is None:
        return StopReason.END_TURN

    match reason:
        case "stop" | "end":
            return StopReason.END_TURN
        case "length":
            return StopReason.MAX_TOKENS
        case "tool_calls" | "function_call":
            return StopReason.TOOL_USE
        case "content_filter":
            return StopReason.ERROR
        case _:
            return StopReason.ERROR


class OpenAIProvider:
    """OpenAI provider implementing the Provider protocol.

    Uses the OpenAI Python SDK to stream responses from OpenAI's Chat Completions API.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        default_headers: dict[str, str] | None = None,
        api_key_resolver: Callable[[], Awaitable[str]] | None = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            model: The model to use (e.g., "gpt-4o", "gpt-4o-mini", "o3").
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Base URL for OpenAI-compatible proxies.
            organization: OpenAI organization ID.
            default_headers: Additional headers to include in requests.
            api_key_resolver: Optional async callable that returns a fresh API key.
                If provided, called before each stream() to get the current key.
        """
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        self._default_headers = default_headers
        self._api_key_resolver = api_key_resolver
        self._client: openai.AsyncOpenAI | None = None

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "openai"

    def _get_client(self) -> openai.AsyncOpenAI:
        """Lazily create and return the OpenAI client."""
        if self._client is None:
            # Lazy import to handle optional dependency
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for OpenAIProvider. "
                    "Install it with: pip install isotope-core[openai]"
                ) from e

            self._client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                organization=self._organization,
                default_headers=self._default_headers,
            )
        return self._client

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from OpenAI.

        Args:
            context: The conversation context.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            signal: An asyncio.Event that, when set, signals abortion.

        Yields:
            StreamEvent: Events describing the streaming response.
        """
        timestamp = current_timestamp_ms()

        # Initialize output message
        output = AssistantMessage(
            content=[],
            stop_reason=None,
            usage=Usage(),
            timestamp=timestamp,
        )

        try:
            # Resolve API key if resolver is provided
            if self._api_key_resolver is not None:
                resolved_key = await self._api_key_resolver()
                client = self._get_client()
                client.api_key = resolved_key
            else:
                client = self._get_client()
            messages, tools = _convert_context_to_openai(context)

            # Build request parameters
            params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
            if tools:
                params["tools"] = tools

            # Start streaming
            stream = await client.chat.completions.create(**params)

            # Emit start event
            yield StreamStartEvent(partial=output)

            # Track current content block state
            current_block: TextContent | ThinkingContent | ToolCallContent | None = None
            current_block_index = -1
            # Track tool calls by OpenAI's tool_call index for parallel tool calls
            tool_call_blocks: dict[int, ToolCallContent] = {}
            tool_call_content_indices: dict[int, int] = {}
            partial_tool_args: dict[int, str] = {}

            def finish_current_block() -> StreamEvent | None:
                """Finish the current content block and return end event."""
                nonlocal current_block, current_block_index, partial_tool_args

                if current_block is None:
                    return None

                event: StreamEvent | None = None

                if isinstance(current_block, TextContent):
                    event = StreamTextEndEvent(
                        content_index=current_block_index,
                        content=current_block.text,
                        partial=output,
                    )
                elif isinstance(current_block, ThinkingContent):
                    event = StreamThinkingEndEvent(
                        content_index=current_block_index,
                        content=current_block.thinking,
                        partial=output,
                    )
                elif isinstance(current_block, ToolCallContent):
                    # Find which tool call index this block belongs to
                    tc_idx = next(
                        (idx for idx, blk in tool_call_blocks.items() if blk is current_block),
                        None,
                    )
                    args_str = partial_tool_args.get(tc_idx, "") if tc_idx is not None else ""
                    current_block.arguments = _parse_streaming_json(args_str)
                    event = StreamToolCallEndEvent(
                        content_index=current_block_index,
                        tool_call_id=current_block.id,
                        tool_name=current_block.name,
                        arguments=dict(current_block.arguments),
                        partial=output,
                    )

                current_block = None
                return event

            async for chunk in stream:
                # Check for abort signal
                if signal and signal.is_set():
                    output.stop_reason = StopReason.ABORTED
                    break

                # Extract usage from final chunk
                if chunk.usage:
                    output.usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        cache_read_tokens=getattr(
                            chunk.usage.prompt_tokens_details, "cached_tokens", 0
                        )
                        or 0,
                    )

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                # Handle finish reason
                if choice.finish_reason:
                    output.stop_reason = _map_finish_reason(choice.finish_reason)

                delta = choice.delta
                if not delta:
                    continue

                # Handle text content
                if delta.content:
                    if current_block is None or not isinstance(current_block, TextContent):
                        # Finish previous block
                        end_event = finish_current_block()
                        if end_event:
                            yield end_event

                        # Start new text block
                        current_block = TextContent(text="")
                        output.content.append(current_block)
                        current_block_index = len(output.content) - 1

                    # Append text
                    current_block.text += delta.content
                    yield StreamTextDeltaEvent(
                        content_index=current_block_index,
                        delta=delta.content,
                        partial=output,
                    )

                # Handle reasoning content (for o-series models)
                # Some OpenAI-compatible APIs use reasoning_content or reasoning field
                reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning_delta:
                    if current_block is None or not isinstance(current_block, ThinkingContent):
                        end_event = finish_current_block()
                        if end_event:
                            yield end_event

                        current_block = ThinkingContent(thinking="")
                        output.content.append(current_block)
                        current_block_index = len(output.content) - 1

                    current_block.thinking += reasoning_delta
                    yield StreamThinkingDeltaEvent(
                        content_index=current_block_index,
                        delta=reasoning_delta,
                        partial=output,
                    )

                # Handle tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        tc_index = getattr(tool_call, "index", None)
                        if tc_index is None:
                            tc_index = 0

                        if tc_index not in tool_call_blocks:
                            # New tool call — finish previous non-tool block
                            if current_block is not None and not isinstance(
                                current_block, ToolCallContent
                            ):
                                end_event = finish_current_block()
                                if end_event:
                                    yield end_event

                            block = ToolCallContent(
                                id=tool_call.id or "",
                                name=tool_call.function.name if tool_call.function else "",
                                arguments={},
                            )
                            output.content.append(block)
                            content_idx = len(output.content) - 1

                            tool_call_blocks[tc_index] = block
                            tool_call_content_indices[tc_index] = content_idx
                            partial_tool_args[tc_index] = ""

                            current_block = block
                            current_block_index = content_idx

                            yield StreamToolCallStartEvent(
                                content_index=content_idx,
                                partial=output,
                            )

                        # Update the tool call block for this index
                        block = tool_call_blocks[tc_index]
                        content_idx = tool_call_content_indices[tc_index]

                        if tool_call.id:
                            block.id = tool_call.id
                        if tool_call.function:
                            if tool_call.function.name:
                                block.name = tool_call.function.name
                            if tool_call.function.arguments:
                                partial_tool_args[tc_index] += tool_call.function.arguments
                                block.arguments = _parse_streaming_json(
                                    partial_tool_args[tc_index]
                                )

                                yield StreamToolCallDeltaEvent(
                                    content_index=content_idx,
                                    delta=tool_call.function.arguments,
                                    partial=output,
                                )

                        # Keep current_block pointing to the latest active tool call
                        current_block = block
                        current_block_index = content_idx

            # Finish any remaining non-tool-call block (text/thinking)
            if current_block is not None and not isinstance(current_block, ToolCallContent):
                end_event = finish_current_block()
                if end_event:
                    yield end_event

            # Finish all tool call blocks
            for tc_idx in list(tool_call_blocks.keys()):
                block = tool_call_blocks[tc_idx]
                content_idx = tool_call_content_indices[tc_idx]
                args_str = partial_tool_args.get(tc_idx, "")
                block.arguments = _parse_streaming_json(args_str)
                yield StreamToolCallEndEvent(
                    content_index=content_idx,
                    tool_call_id=block.id,
                    tool_name=block.name,
                    arguments=dict(block.arguments),
                    partial=output,
                )

            # Check for abort after stream ends
            if signal and signal.is_set():
                output.stop_reason = StopReason.ABORTED
                output.error_message = "Request was aborted"
                yield StreamErrorEvent(error=output)
                return

            # Set default stop reason if not set
            if output.stop_reason is None:
                output.stop_reason = StopReason.END_TURN

            # Emit done event
            yield StreamDoneEvent(message=output)

        except Exception as e:
            # Check if this was an abort
            if signal and signal.is_set():
                output.stop_reason = StopReason.ABORTED
                output.error_message = "Request was aborted"
            else:
                error_msg = create_error_message(e, timestamp, output.usage)
                output.stop_reason = error_msg.stop_reason
                output.error_message = error_msg.error_message

            yield StreamErrorEvent(error=output)
