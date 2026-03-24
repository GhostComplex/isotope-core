"""Anthropic provider implementation.

This module provides the Anthropic provider that implements the Provider protocol
for Anthropic's Messages API with streaming.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
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
    import anthropic


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


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking.

    Attributes:
        enabled: Whether to enable extended thinking.
        budget_tokens: Token budget for thinking (for budget-based thinking).
    """

    enabled: bool = False
    budget_tokens: int = 1024


def _convert_context_to_anthropic(
    context: Context,
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Convert our Context to Anthropic message format.

    Args:
        context: Our Context object.

    Returns:
        Tuple of (system, messages, tools) in Anthropic format.
    """
    # System prompt
    system: str | list[dict[str, Any]] | None = None
    if context.system_prompt:
        system = context.system_prompt

    # Convert messages
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            messages.append(_convert_user_message(msg))
        elif isinstance(msg, AssistantMessage):
            converted = _convert_assistant_message(msg)
            if converted:
                messages.append(converted)
        elif isinstance(msg, ToolResultMessage):
            # In Anthropic, tool results go in user messages
            _append_tool_result(messages, msg)

    # Convert tools
    tools = _convert_tools(context.tools) if context.tools else None

    return system, messages, tools


def _convert_user_message(msg: UserMessage) -> dict[str, Any]:
    """Convert UserMessage to Anthropic format."""
    content: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextContent):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )

    # Anthropic accepts string for text-only content
    if len(content) == 1 and content[0]["type"] == "text":
        return {"role": "user", "content": content[0]["text"]}

    return {"role": "user", "content": content}


def _convert_assistant_message(msg: AssistantMessage) -> dict[str, Any] | None:
    """Convert AssistantMessage to Anthropic format."""
    content: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextContent):
            if block.text.strip():
                content.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            # Handle redacted thinking
            if block.redacted and block.thinking_signature:
                content.append(
                    {
                        "type": "redacted_thinking",
                        "data": block.thinking_signature,
                    }
                )
            elif block.thinking.strip():
                # Only include if we have a signature
                if block.thinking_signature:
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": block.thinking_signature,
                        }
                    )
                else:
                    # Without signature, convert to text
                    content.append({"type": "text", "text": block.thinking})
        elif isinstance(block, ToolCallContent):
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                }
            )

    if not content:
        return None

    return {"role": "assistant", "content": content}


def _append_tool_result(messages: list[dict[str, Any]], msg: ToolResultMessage) -> None:
    """Append a tool result to messages.

    In Anthropic's format, tool results are part of user messages.
    Multiple consecutive tool results should be combined into one user message.
    """
    # Build content from tool result
    result_content: list[dict[str, Any]] | str = []

    for block in msg.content:
        if isinstance(block, TextContent):
            if isinstance(result_content, list):
                result_content.append({"type": "text", "text": block.text})
            else:
                result_content = block.text
        elif isinstance(block, ImageContent):
            if isinstance(result_content, str):
                result_content = [{"type": "text", "text": result_content}]
            result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )

    # Simplify if only text
    if (
        isinstance(result_content, list)
        and len(result_content) == 1
        and result_content[0]["type"] == "text"
    ):
        result_content = result_content[0]["text"]

    tool_result_block = {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": result_content,
        "is_error": msg.is_error,
    }

    # Check if last message is a user message (we can append to it)
    if messages and messages[-1]["role"] == "user":
        last_content = messages[-1]["content"]
        if isinstance(last_content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in last_content
        ):
            last_content.append(tool_result_block)
            return

    # Create new user message
    messages.append({"role": "user", "content": [tool_result_block]})


def _convert_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert our ToolSchema list to Anthropic format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": tool.parameters.get("properties", {}),
                "required": tool.parameters.get("required", []),
            },
        }
        for tool in tools
    ]


def _map_stop_reason(reason: str) -> StopReason:
    """Map Anthropic stop_reason to our StopReason."""
    match reason:
        case "end_turn":
            return StopReason.END_TURN
        case "max_tokens":
            return StopReason.MAX_TOKENS
        case "tool_use":
            return StopReason.TOOL_USE
        case "stop_sequence":
            return StopReason.END_TURN
        case _:
            return StopReason.ERROR


class AnthropicProvider:
    """Anthropic provider implementing the Provider protocol.

    Uses the Anthropic Python SDK to stream responses from Anthropic's Messages API.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8192,
        thinking: ThinkingConfig | None = None,
        api_key_resolver: Callable[[], Awaitable[str]] | None = None,
    ):
        """Initialize the Anthropic provider.

        Args:
            model: The model to use (e.g., "claude-sonnet-4-20250514", "claude-opus-4-20250514").
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            base_url: Base URL for the API.
            max_tokens: Default maximum tokens to generate.
            thinking: Extended thinking configuration.
            api_key_resolver: Optional async callable that returns a fresh API key.
                If provided, called before each stream() to get the current key.
        """
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._default_max_tokens = max_tokens
        self._thinking = thinking
        self._api_key_resolver = api_key_resolver
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "anthropic"

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Lazily create and return the Anthropic client."""
        if self._client is None:
            # Lazy import to handle optional dependency
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicProvider. "
                    "Install it with: pip install isotope-core[anthropic]"
                ) from e

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def stream(
        self,
        context: Context,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        signal: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from Anthropic.

        Args:
            context: The conversation context.
            temperature: Sampling temperature (0.0 to 1.0).
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
            system, messages, tools = _convert_context_to_anthropic(context)

            # Build request parameters
            params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self._default_max_tokens,
            }

            if system:
                params["system"] = system
            if temperature is not None and not (self._thinking and self._thinking.enabled):
                # Note: temperature is incompatible with extended thinking
                params["temperature"] = temperature
            if tools:
                params["tools"] = tools

            # Configure extended thinking
            if self._thinking and self._thinking.enabled:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking.budget_tokens,
                }

            # Start streaming
            stream = client.messages.stream(**params)

            # Emit start event
            yield StreamStartEvent(partial=output)

            # Track content blocks by their index
            blocks: dict[int, TextContent | ThinkingContent | ToolCallContent] = {}
            partial_tool_args: dict[int, str] = {}
            thinking_signatures: dict[int, str] = {}

            async with stream as response:
                async for event in response:
                    # Check for abort signal
                    if signal and signal.is_set():
                        output.stop_reason = StopReason.ABORTED
                        break

                    if event.type == "message_start":
                        # Capture initial usage from message_start
                        msg = event.message
                        output.usage = Usage(
                            input_tokens=msg.usage.input_tokens or 0,
                            output_tokens=msg.usage.output_tokens or 0,
                            cache_read_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                            cache_write_tokens=getattr(msg.usage, "cache_creation_input_tokens", 0)
                            or 0,
                        )

                    elif event.type == "content_block_start":
                        block_type = event.content_block.type
                        index = event.index
                        block: TextContent | ThinkingContent | ToolCallContent | None = None

                        if block_type == "text":
                            block = TextContent(text="")
                            blocks[index] = block
                            output.content.append(block)

                        elif block_type == "thinking":
                            block = ThinkingContent(thinking="")
                            blocks[index] = block
                            output.content.append(block)

                        elif block_type == "redacted_thinking":
                            # Handle redacted thinking blocks
                            data = getattr(event.content_block, "data", "")
                            block = ThinkingContent(
                                thinking="[Reasoning redacted]",
                                thinking_signature=data,
                                redacted=True,
                            )
                            blocks[index] = block
                            output.content.append(block)

                            # Emit thinking start event
                            yield StreamThinkingDeltaEvent(
                                content_index=len(output.content) - 1,
                                delta="[Reasoning redacted]",
                                partial=output,
                            )

                        elif block_type == "tool_use":
                            cb = event.content_block
                            block = ToolCallContent(
                                id=getattr(cb, "id", ""),
                                name=getattr(cb, "name", ""),
                                arguments={},
                            )
                            blocks[index] = block
                            partial_tool_args[index] = ""
                            output.content.append(block)

                            yield StreamToolCallStartEvent(
                                content_index=len(output.content) - 1,
                                partial=output,
                            )

                    elif event.type == "content_block_delta":
                        index = event.index
                        delta_block = blocks.get(index)
                        if delta_block is None:
                            continue

                        delta_type = event.delta.type

                        if delta_type == "text_delta":
                            if isinstance(delta_block, TextContent):
                                text_delta = getattr(event.delta, "text", "")
                                delta_block.text += text_delta

                                # Find content index
                                content_idx = output.content.index(delta_block)
                                yield StreamTextDeltaEvent(
                                    content_index=content_idx,
                                    delta=text_delta,
                                    partial=output,
                                )

                        elif delta_type == "thinking_delta":
                            if isinstance(delta_block, ThinkingContent):
                                thinking_delta: str = getattr(event.delta, "thinking", "")
                                delta_block.thinking += thinking_delta

                                content_idx = output.content.index(delta_block)
                                yield StreamThinkingDeltaEvent(
                                    content_index=content_idx,
                                    delta=thinking_delta,
                                    partial=output,
                                )

                        elif delta_type == "signature_delta":
                            if isinstance(delta_block, ThinkingContent):
                                sig_delta: str = getattr(event.delta, "signature", "")
                                if index not in thinking_signatures:
                                    thinking_signatures[index] = ""
                                thinking_signatures[index] += sig_delta
                                delta_block.thinking_signature = thinking_signatures[index]

                        elif (
                            delta_type == "input_json_delta"
                            and isinstance(delta_block, ToolCallContent)
                        ):
                            json_delta: str = getattr(event.delta, "partial_json", "")
                            partial_tool_args[index] += json_delta
                            delta_block.arguments = _parse_streaming_json(
                                partial_tool_args[index]
                            )

                            content_idx = output.content.index(delta_block)
                            yield StreamToolCallDeltaEvent(
                                content_index=content_idx,
                                delta=json_delta,
                                partial=output,
                            )

                    elif event.type == "content_block_stop":
                        index = event.index
                        block = blocks.get(index)
                        if block is None:
                            continue

                        content_idx = output.content.index(block)

                        if isinstance(block, TextContent):
                            yield StreamTextEndEvent(
                                content_index=content_idx,
                                content=block.text,
                                partial=output,
                            )
                        elif isinstance(block, ThinkingContent):
                            yield StreamThinkingEndEvent(
                                content_index=content_idx,
                                content=block.thinking,
                                partial=output,
                            )
                        elif isinstance(block, ToolCallContent):
                            # Parse final arguments
                            if index in partial_tool_args:
                                block.arguments = _parse_streaming_json(partial_tool_args[index])
                            yield StreamToolCallEndEvent(
                                content_index=content_idx,
                                tool_call_id=block.id,
                                tool_name=block.name,
                                arguments=dict(block.arguments),
                                partial=output,
                            )

                    elif event.type == "message_delta":
                        if event.delta.stop_reason:
                            output.stop_reason = _map_stop_reason(event.delta.stop_reason)

                        # Update usage
                        if event.usage and event.usage.output_tokens:
                            output.usage.output_tokens = event.usage.output_tokens

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
