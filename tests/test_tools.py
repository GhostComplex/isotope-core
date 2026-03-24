"""Tests for isotope_core.tools module."""

import asyncio
from typing import Any

import pytest

from isotope_core.tools import (
    Tool,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolResult,
    ToolValidationError,
    tool,
    validate_json_schema,
)


class TestToolResult:
    """Tests for ToolResult."""

    def test_default_tool_result(self) -> None:
        """Test default ToolResult."""
        result = ToolResult()
        assert result.content == []
        assert result.is_error is False

    def test_tool_result_text(self) -> None:
        """Test ToolResult.text() factory."""
        result = ToolResult.text("Hello, world!")
        assert len(result.content) == 1
        assert result.content[0].text == "Hello, world!"
        assert result.is_error is False

    def test_tool_result_text_with_error(self) -> None:
        """Test ToolResult.text() with error flag."""
        result = ToolResult.text("Something went wrong", is_error=True)
        assert result.is_error is True

    def test_tool_result_error(self) -> None:
        """Test ToolResult.error() factory."""
        result = ToolResult.error("File not found")
        assert len(result.content) == 1
        assert result.content[0].text == "File not found"
        assert result.is_error is True


class TestValidateJsonSchema:
    """Tests for JSON schema validation."""

    def test_validate_string(self) -> None:
        """Test validating a string."""
        valid, error = validate_json_schema("hello", {"type": "string"})
        assert valid is True
        assert error is None

    def test_validate_string_fails_for_int(self) -> None:
        """Test string validation fails for int."""
        valid, error = validate_json_schema(123, {"type": "string"})
        assert valid is False
        assert "Expected string" in error  # type: ignore[operator]

    def test_validate_number(self) -> None:
        """Test validating a number."""
        valid, error = validate_json_schema(3.14, {"type": "number"})
        assert valid is True

    def test_validate_integer(self) -> None:
        """Test validating an integer."""
        valid, error = validate_json_schema(42, {"type": "integer"})
        assert valid is True

    def test_validate_integer_fails_for_float(self) -> None:
        """Test integer validation fails for float."""
        valid, error = validate_json_schema(3.14, {"type": "integer"})
        assert valid is False

    def test_validate_boolean(self) -> None:
        """Test validating a boolean."""
        valid, error = validate_json_schema(True, {"type": "boolean"})
        assert valid is True

    def test_validate_null(self) -> None:
        """Test validating null."""
        valid, error = validate_json_schema(None, {"type": "null"})
        assert valid is True

    def test_validate_array(self) -> None:
        """Test validating an array."""
        schema = {"type": "array", "items": {"type": "string"}}
        valid, error = validate_json_schema(["a", "b", "c"], schema)
        assert valid is True

    def test_validate_array_fails_for_invalid_items(self) -> None:
        """Test array validation fails for invalid items."""
        schema = {"type": "array", "items": {"type": "string"}}
        valid, error = validate_json_schema(["a", 1, "c"], schema)
        assert valid is False

    def test_validate_object(self) -> None:
        """Test validating an object."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        valid, error = validate_json_schema({"name": "Alice", "age": 30}, schema)
        assert valid is True

    def test_validate_object_missing_required(self) -> None:
        """Test object validation fails for missing required property."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        valid, error = validate_json_schema({}, schema)
        assert valid is False
        assert "Missing required property" in error  # type: ignore[operator]

    def test_validate_nested_object(self) -> None:
        """Test validating a nested object."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        valid, error = validate_json_schema({"user": {"name": "Bob"}}, schema)
        assert valid is True


class TestTool:
    """Tests for Tool class."""

    @pytest.fixture
    def echo_tool(self) -> Tool:
        """Create a simple echo tool."""

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text(f"Echo: {params.get('message', '')}")

        return Tool(
            name="echo",
            description="Echoes the message",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            execute=execute,
        )

    def test_tool_properties(self, echo_tool: Tool) -> None:
        """Test tool properties."""
        assert echo_tool.name == "echo"
        assert echo_tool.description == "Echoes the message"
        assert "message" in echo_tool.parameters["properties"]

    def test_tool_validate_arguments_valid(self, echo_tool: Tool) -> None:
        """Test validating valid arguments."""
        valid, error = echo_tool.validate_arguments({"message": "hello"})
        assert valid is True
        assert error is None

    def test_tool_validate_arguments_missing_required(self, echo_tool: Tool) -> None:
        """Test validating missing required argument."""
        valid, error = echo_tool.validate_arguments({})
        assert valid is False
        assert error is not None

    @pytest.mark.asyncio
    async def test_tool_execute(self, echo_tool: Tool) -> None:
        """Test executing a tool."""
        result = await echo_tool.execute("call_1", {"message": "hello"})
        assert result.is_error is False
        assert result.content[0].text == "Echo: hello"

    @pytest.mark.asyncio
    async def test_tool_execute_invalid_args(self, echo_tool: Tool) -> None:
        """Test executing with invalid arguments raises error."""
        with pytest.raises(ToolValidationError):
            await echo_tool.execute("call_1", {})

    def test_tool_to_schema(self, echo_tool: Tool) -> None:
        """Test converting tool to schema."""
        schema = echo_tool.to_schema()
        assert schema["name"] == "echo"
        assert schema["description"] == "Echoes the message"
        assert "parameters" in schema


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_basic(self) -> None:
        """Test basic tool decorator."""

        @tool(
            name="greet",
            description="Greets someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        async def greet(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            return ToolResult.text(f"Hello, {params['name']}!")

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert greet.description == "Greets someone"

    @pytest.mark.asyncio
    async def test_tool_decorator_execute(self) -> None:
        """Test executing a decorated tool."""

        @tool(
            name="add",
            description="Adds two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )
        async def add(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            result = params["a"] + params["b"]
            return ToolResult.text(str(result))

        result = await add.execute("call_1", {"a": 2, "b": 3})
        assert result.content[0].text == "5"


class TestToolWithSignal:
    """Tests for tool abort signal handling."""

    @pytest.mark.asyncio
    async def test_tool_respects_abort_signal(self) -> None:
        """Test that a tool can check and respond to abort signal."""

        @tool(name="long_task", description="A long running task")
        async def long_task(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            if signal and signal.is_set():
                return ToolResult.error("Aborted")
            return ToolResult.text("Completed")

        # Test without signal
        result = await long_task.execute("call_1", {})
        assert result.content[0].text == "Completed"

        # Test with set signal
        abort_signal = asyncio.Event()
        abort_signal.set()
        result = await long_task.execute("call_1", {}, signal=abort_signal)
        assert result.is_error is True
        assert result.content[0].text == "Aborted"


class TestToolWithUpdates:
    """Tests for tool update callbacks."""

    @pytest.mark.asyncio
    async def test_tool_sends_updates(self) -> None:
        """Test that a tool can send updates via callback."""
        updates: list[ToolResult] = []

        @tool(name="progress_task", description="A task with progress updates")
        async def progress_task(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            if on_update:
                on_update(ToolResult.text("Step 1 complete"))
                on_update(ToolResult.text("Step 2 complete"))
            return ToolResult.text("All done")

        result = await progress_task.execute("call_1", {}, on_update=lambda r: updates.append(r))
        assert len(updates) == 2
        assert updates[0].content[0].text == "Step 1 complete"
        assert updates[1].content[0].text == "Step 2 complete"
        assert result.content[0].text == "All done"


class TestToolErrors:
    """Tests for tool exceptions."""

    def test_tool_error_hierarchy(self) -> None:
        """Test that tool errors inherit correctly."""
        assert issubclass(ToolValidationError, ToolError)
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolExecutionError, ToolError)

    @pytest.mark.asyncio
    async def test_tool_execution_error_is_caught(self) -> None:
        """Test that tool execution errors are properly handled."""

        @tool(name="failing_tool", description="A tool that fails")
        async def failing_tool(
            tool_call_id: str,
            params: dict[str, Any],
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> ToolResult:
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            await failing_tool.execute("call_1", {})
