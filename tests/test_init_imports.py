"""Tests for isotope_core/__init__.py — optional import guards."""

from __future__ import annotations

import importlib
import sys


class TestTopLevelOptionalImports:
    """Test that optional provider imports in __init__.py are handled gracefully."""

    def test_openai_not_installed_top_level(self) -> None:
        """When openai is not installed, OpenAIProvider not in isotope_core.__all__."""
        # Snapshot state
        original_modules = dict(sys.modules)
        try:
            # Remove cached isotope_core and provider modules
            for key in list(sys.modules.keys()):
                if key.startswith("isotope_core"):
                    del sys.modules[key]

            # Block openai-dependent modules
            sys.modules["openai"] = None  # type: ignore[assignment]
            sys.modules["isotope_core.providers.openai"] = None  # type: ignore[assignment]
            sys.modules["isotope_core.providers.proxy"] = None  # type: ignore[assignment]

            mod = importlib.import_module("isotope_core")
            assert "OpenAIProvider" not in mod.__all__
            assert "ProxyProvider" not in mod.__all__
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_anthropic_not_installed_top_level(self) -> None:
        """When anthropic SDK is not installed, AnthropicProvider not in isotope_core.__all__."""
        original_modules = dict(sys.modules)
        try:
            for key in list(sys.modules.keys()):
                if key.startswith("isotope_core"):
                    del sys.modules[key]

            sys.modules["anthropic"] = None  # type: ignore[assignment]
            sys.modules["isotope_core.providers.anthropic"] = None  # type: ignore[assignment]

            mod = importlib.import_module("isotope_core")
            assert "AnthropicProvider" not in mod.__all__
            assert "ThinkingConfig" not in mod.__all__
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_proxy_not_installed_top_level(self) -> None:
        """When openai SDK is not installed, ProxyProvider not in isotope_core.__all__."""
        original_modules = dict(sys.modules)
        try:
            for key in list(sys.modules.keys()):
                if key.startswith("isotope_core"):
                    del sys.modules[key]

            sys.modules["isotope_core.providers.proxy"] = None  # type: ignore[assignment]
            sys.modules["isotope_core.providers.openai"] = None  # type: ignore[assignment]

            mod = importlib.import_module("isotope_core")
            assert "ProxyProvider" not in mod.__all__
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)
