"""Tests for get_default_client_settings factory function."""

import os
from unittest.mock import patch

import pytest

from uipath.llm_client.settings import (
    LLMGatewaySettings,
    PlatformSettings,
    get_default_client_settings,
)


class TestSettingsFactory:
    """Tests for get_default_client_settings factory function."""

    def test_default_returns_agenthub(self, platform_env_vars, mock_platform_auth):
        """Test that default backend is agenthub."""
        env = {**platform_env_vars}
        env.pop("UIPATH_LLM_SERVICE", None)
        with patch.dict(os.environ, env, clear=True):
            settings = get_default_client_settings()
            assert isinstance(settings, PlatformSettings)

    def test_explicit_agenthub(self, platform_env_vars, mock_platform_auth):
        """Test explicit agenthub backend."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = get_default_client_settings(backend="agenthub")
            assert isinstance(settings, PlatformSettings)

    def test_explicit_llmgateway(self, llmgw_env_vars):
        """Test explicit llmgateway backend."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = get_default_client_settings(backend="llmgateway")
            assert isinstance(settings, LLMGatewaySettings)

    def test_env_var_agenthub(self, platform_env_vars, mock_platform_auth):
        """Test UIPATH_LLM_SERVICE=agenthub from environment."""
        env = {**platform_env_vars, "UIPATH_LLM_SERVICE": "agenthub"}
        with patch.dict(os.environ, env, clear=True):
            settings = get_default_client_settings()
            assert isinstance(settings, PlatformSettings)

    def test_env_var_llmgateway(self, llmgw_env_vars):
        """Test UIPATH_LLM_SERVICE=llmgateway from environment."""
        env = {**llmgw_env_vars, "UIPATH_LLM_SERVICE": "llmgateway"}
        with patch.dict(os.environ, env, clear=True):
            settings = get_default_client_settings()
            assert isinstance(settings, LLMGatewaySettings)

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend type"):
            get_default_client_settings(backend="invalid")  # type: ignore
