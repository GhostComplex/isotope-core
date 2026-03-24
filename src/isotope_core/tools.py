"""Tool framework for agent loop.

This module provides the Tool class and ToolResult type for defining
and executing tools within the agent loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from isotope_core.types import ImageContent, TextContent

# =============================================================================
# Tool Result
# =============================================================================


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        content: List of content blocks (text or images) to return to the model.
        is_error: Whether the tool execution resulted in an error.
    """

    content: list[TextContent | ImageContent] = field(default_factory=list)
    is_error: bool = False

    @classmethod
    def text(cls, text: str, is_error: bool = False) -> ToolResult:
        """Create a ToolResult with text content."""
        return cls(content=[TextContent(text=text)], is_error=is_error)

    @classmethod
    def error(cls, message: str) -> ToolResult:
        """Create an error ToolResult."""
        return cls(content=[TextContent(text=message)], is_error=True)


# =============================================================================
# Tool Update Callback
# =============================================================================

T = TypeVar("T")

ToolUpdateCallback = Callable[[ToolResult], None]


# =============================================================================
# Tool Execution Function Type
# =============================================================================

# Type for the execute function: async (tool_call_id, params, signal?, on_update?) -> ToolResult
ExecuteFn = Callable[
    [str, dict[str, Any], asyncio.Event | None, ToolUpdateCallback | None],
    Awaitable[ToolResult],
]


# =============================================================================
# Tool Schema Validation
# =============================================================================


def validate_json_schema(value: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate a value against a JSON schema.

    This is a simplified validator that handles common cases.
    For production, consider using jsonschema library.

    Args:
        value: The value to validate.
        schema: The JSON schema to validate against.

    Returns:
        A tuple of (is_valid, error_message).
    """
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(value, dict):
            return False, f"Expected object, got {type(value).__name__}"

        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in value:
                return False, f"Missing required property: {prop}"

        # Validate properties
        properties = schema.get("properties", {})
        for prop_name, prop_value in value.items():
            if prop_name in properties:
                valid, error = validate_json_schema(prop_value, properties[prop_name])
                if not valid:
                    return False, f"Property '{prop_name}': {error}"

        return True, None

    elif schema_type == "array":
        if not isinstance(value, list):
            return False, f"Expected array, got {type(value).__name__}"

        items_schema = schema.get("items", {})
        for i, item in enumerate(value):
            valid, error = validate_json_schema(item, items_schema)
            if not valid:
                return False, f"Item {i}: {error}"

        return True, None

    elif schema_type == "string":
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        return True, None

    elif schema_type == "number":
        if not isinstance(value, (int, float)):
            return False, f"Expected number, got {type(value).__name__}"
        return True, None

    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"Expected integer, got {type(value).__name__}"
        return True, None

    elif schema_type == "boolean":
        if not isinstance(value, bool):
            return False, f"Expected boolean, got {type(value).__name__}"
        return True, None

    elif schema_type == "null":
        if value is not None:
            return False, f"Expected null, got {type(value).__name__}"
        return True, None

    # No type specified or unknown type - accept anything
    return True, None


# =============================================================================
# Tool Class
# =============================================================================


class Tool:
    """A tool that can be executed by the agent.

    Tools have a name, description, JSON schema for parameters, and an
    execute function that runs the tool.

    Attributes:
        name: The unique name of the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema object describing the tool's parameters.
        execute: Async function to execute the tool.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute: ExecuteFn,
    ):
        """Initialize a Tool.

        Args:
            name: The unique name of the tool.
            description: Human-readable description of what the tool does.
            parameters: JSON Schema object describing the tool's parameters.
            execute: Async function to execute the tool.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self._execute = execute

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate arguments against the tool's parameter schema.

        Args:
            arguments: The arguments to validate.

        Returns:
            A tuple of (is_valid, error_message).
        """
        return validate_json_schema(arguments, self.parameters)

    async def execute(
        self,
        tool_call_id: str,
        arguments: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            tool_call_id: Unique identifier for this tool call.
            arguments: The arguments to pass to the tool.
            signal: An asyncio.Event that, when set, signals abortion.
            on_update: Optional callback for streaming updates.

        Returns:
            The result of the tool execution.

        Raises:
            ToolValidationError: If arguments don't match the schema.
        """
        # Validate arguments
        valid, error = self.validate_arguments(arguments)
        if not valid:
            raise ToolValidationError(f"Invalid arguments: {error}")

        return await self._execute(tool_call_id, arguments, signal, on_update)

    def to_schema(self) -> dict[str, Any]:
        """Convert the tool to a JSON-serializable schema for LLM APIs.

        Returns:
            A dictionary containing the tool's schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# =============================================================================
# Exceptions
# =============================================================================


class ToolError(Exception):
    """Base exception for tool-related errors."""

    pass


class ToolValidationError(ToolError):
    """Raised when tool arguments fail validation."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""

    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""

    pass


# =============================================================================
# Tool Decorator (convenience)
# =============================================================================


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[ExecuteFn], Tool]:
    """Decorator to create a Tool from an async function.

    Example:
        @tool(
            name="get_weather",
            description="Get the current weather",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"]
            }
        )
        async def get_weather(tool_call_id, params, signal, on_update):
            return ToolResult.text(f"Weather in {params['location']}: Sunny")

    Args:
        name: The unique name of the tool.
        description: Human-readable description.
        parameters: JSON Schema for parameters.

    Returns:
        A decorator that creates a Tool.
    """

    def decorator(fn: ExecuteFn) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            execute=fn,
        )

    return decorator
