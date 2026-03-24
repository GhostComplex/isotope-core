"""Fixtures for integration tests.

Provides proxy availability checks and shared fixtures.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest

from isotope_core.providers.proxy import ProxyProvider
from isotope_core.tools import Tool, ToolResult

PROXY_BASE_URL = "http://localhost:4141/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Bypass system HTTP proxies (e.g. Clash on port 7890) for localhost
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")


def _proxy_is_reachable() -> bool:
    """Check if the proxy is reachable.

    Uses trust_env=False to bypass system HTTP proxies (e.g. Clash on
    port 7890) that would otherwise intercept localhost requests.
    """
    try:
        with httpx.Client(trust_env=False, timeout=5) as client:
            resp = client.get(f"{PROXY_BASE_URL}/models")
            if resp.status_code == 200:
                return True
            # Fall back to a minimal chat completion
            resp = client.post(
                f"{PROXY_BASE_URL}/chat/completions",
                json={
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=10,
            )
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


# Skip all integration tests if proxy is down
_proxy_available = _proxy_is_reachable()


@pytest.fixture(autouse=True)
def _skip_if_proxy_down() -> None:
    """Auto-skip integration tests when the proxy is unreachable."""
    if not _proxy_available:
        pytest.skip("Proxy at localhost:4141 is not reachable")


@pytest.fixture
def proxy_provider() -> ProxyProvider:
    """Create a ProxyProvider pointing at the local proxy."""
    return ProxyProvider(
        model=DEFAULT_MODEL,
        base_url=PROXY_BASE_URL,
        api_key="not-needed",
    )


@pytest.fixture
def get_current_time_tool() -> Tool:
    """A simple tool that returns the current time."""

    async def _execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        import datetime

        now = datetime.datetime.now(tz=datetime.UTC).isoformat()
        return ToolResult.text(f"Current time: {now}")

    return Tool(
        name="get_current_time",
        description="Get the current date and time in UTC",
        parameters={"type": "object", "properties": {}},
        execute=_execute,
    )


@pytest.fixture
def calculator_tool() -> Tool:
    """A calculator tool for basic math."""

    async def _execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        expression = params.get("expression", "")
        try:
            # Only allow safe math expressions
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return ToolResult.error(f"Invalid expression: {expression}")
            result = eval(expression)  # noqa: S307
            return ToolResult.text(str(result))
        except Exception as e:
            return ToolResult.error(str(e))

    return Tool(
        name="calculator",
        description="Evaluate a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2')",
                },
            },
            "required": ["expression"],
        },
        execute=_execute,
    )
