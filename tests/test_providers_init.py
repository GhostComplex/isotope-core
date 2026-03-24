"""Tests for providers/__init__.py — optional import guards."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


class TestOptionalProviderImports:
    """Test that optional provider imports are handled gracefully."""

    def test_openai_not_installed(self) -> None:
        """When openai is not installed, OpenAIProvider should not be in providers."""
        # Temporarily remove modules that depend on openai
        modules_to_hide = {
            k: v for k, v in sys.modules.items() if "openai" in k or "isotope_core.providers" in k
        }
        with patch.dict(sys.modules, {**{k: None for k in modules_to_hide}}):
            # Remove cached modules so reimport runs the import guards
            for key in list(sys.modules.keys()):
                if key.startswith("isotope_core.providers"):
                    sys.modules.pop(key, None)

            # Simulate openai not installed
            sys.modules["openai"] = None  # type: ignore[assignment]
            sys.modules["isotope_core.providers.openai"] = None  # type: ignore[assignment]
            # Also block proxy since it depends on openai
            sys.modules["isotope_core.providers.proxy"] = None  # type: ignore[assignment]

            mod = importlib.import_module("isotope_core.providers")
            importlib.reload(mod)
            assert "OpenAIProvider" not in mod.__all__
            assert "ProxyProvider" not in mod.__all__

    def test_anthropic_not_installed(self) -> None:
        """When anthropic SDK is not installed, AnthropicProvider should not be in providers."""
        modules_to_hide = {
            k: v
            for k, v in sys.modules.items()
            if "anthropic" in k or "isotope_core.providers" in k
        }
        with patch.dict(sys.modules, {**{k: None for k in modules_to_hide}}):
            for key in list(sys.modules.keys()):
                if key.startswith("isotope_core.providers"):
                    sys.modules.pop(key, None)

            sys.modules["anthropic"] = None  # type: ignore[assignment]
            sys.modules["isotope_core.providers.anthropic"] = None  # type: ignore[assignment]

            mod = importlib.import_module("isotope_core.providers")
            importlib.reload(mod)
            assert "AnthropicProvider" not in mod.__all__
            assert "ThinkingConfig" not in mod.__all__

    def test_proxy_not_installed(self) -> None:
        """When openai is not installed, ProxyProvider should not be in providers."""
        modules_to_hide = {
            k: v
            for k, v in sys.modules.items()
            if "openai" in k or "isotope_core.providers.proxy" in k
        }
        with patch.dict(sys.modules, {**{k: None for k in modules_to_hide}}):
            for key in list(sys.modules.keys()):
                if key.startswith("isotope_core.providers"):
                    sys.modules.pop(key, None)

            sys.modules["isotope_core.providers.proxy"] = None  # type: ignore[assignment]

            mod = importlib.import_module("isotope_core.providers")
            importlib.reload(mod)
            assert "ProxyProvider" not in mod.__all__

    def test_base_always_available(self) -> None:
        """Base provider types should always be importable."""
        from isotope_core.providers import (
            CircuitState,
            Provider,
            RetryConfig,
            RouterProvider,
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
            create_error_message,
            current_timestamp_ms,
            get_retry_after,
            is_retryable_error,
            map_error_to_stop_reason,
            retry_with_backoff,
        )

        assert Provider is not None
        assert StreamEvent is not None
        assert StreamStartEvent is not None
        assert StreamTextDeltaEvent is not None
        assert StreamTextEndEvent is not None
        assert StreamThinkingDeltaEvent is not None
        assert StreamThinkingEndEvent is not None
        assert StreamToolCallStartEvent is not None
        assert StreamToolCallDeltaEvent is not None
        assert StreamToolCallEndEvent is not None
        assert StreamDoneEvent is not None
        assert StreamErrorEvent is not None
        assert RouterProvider is not None
        assert CircuitState is not None
        assert RetryConfig is not None
        assert retry_with_backoff is not None
        assert is_retryable_error is not None
        assert get_retry_after is not None
        assert map_error_to_stop_reason is not None
        assert create_error_message is not None
        assert current_timestamp_ms is not None
