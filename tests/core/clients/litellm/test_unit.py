"""Unit tests for the UiPathLiteLLM client.

Tests provider resolution, model name construction, api_config building,
and client initialization without making any network calls.
"""

from unittest.mock import MagicMock

import pytest

from uipath.llm_client.clients.litellm.client import (
    _ANTHROPIC_FAMILY,
    _FLAVOR_TO_LITELLM,
    _VENDOR_TO_LITELLM,
    UiPathLiteLLM,
    _drop_nones,
)
from uipath.llm_client.settings.constants import ApiFlavor, RoutingMode, VendorType

MODULE = "uipath.llm_client.clients.litellm.client"

# ---------------------------------------------------------------------------
# Mock discovery data
# ---------------------------------------------------------------------------

_OPENAI_MODEL = {
    "modelName": "gpt-5.2-2025-12-11",
    "vendor": "OpenAi",
    "apiFlavor": None,
    "modelFamily": "OpenAi",
}

_GEMINI_MODEL = {
    "modelName": "gemini-2.5-flash",
    "vendor": "VertexAi",
    "apiFlavor": None,
    "modelFamily": "GoogleGemini",
}

_BEDROCK_CLAUDE_MODEL = {
    "modelName": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "vendor": "AwsBedrock",
    "apiFlavor": None,
    "modelFamily": "AnthropicClaude",
}

_VERTEX_CLAUDE_MODEL = {
    "modelName": "claude-sonnet-4-5@20250929",
    "vendor": "VertexAi",
    "apiFlavor": None,
    "modelFamily": "AnthropicClaude",
}

_EMBEDDING_MODEL = {
    "modelName": "text-embedding-3-large",
    "vendor": "OpenAi",
    "apiFlavor": None,
    "modelFamily": "OpenAi",
}


def _mock_settings(models: list[dict]) -> MagicMock:
    settings = MagicMock()
    settings.get_available_models.return_value = models
    settings.build_base_url.return_value = "https://example.com/api"
    settings.build_auth_headers.return_value = {}
    settings.build_auth_pipeline.return_value = None
    return settings


# ============================================================================
# _drop_nones helper
# ============================================================================


class TestDropNones:
    def test_removes_none_values(self):
        assert _drop_nones(a=1, b=None, c="x") == {"a": 1, "c": "x"}

    def test_empty_when_all_none(self):
        assert _drop_nones(a=None, b=None) == {}

    def test_preserves_falsy_non_none(self):
        assert _drop_nones(a=0, b="", c=False, d=None) == {"a": 0, "b": "", "c": False}


# ============================================================================
# Provider resolution
# ============================================================================


class TestResolveProvider:
    """Test _resolve_llm_provider for various vendor/model combinations."""

    def _make_client(self, model_data: dict, **kwargs) -> UiPathLiteLLM:
        settings = _mock_settings([model_data])
        return UiPathLiteLLM(
            model_name=model_data["modelName"],
            client_settings=settings,
            **kwargs,
        )

    def test_openai_model_resolves_to_openai(self):
        client = self._make_client(_OPENAI_MODEL)
        assert client._custom_llm_provider == "openai"

    def test_gemini_model_resolves_to_gemini(self):
        client = self._make_client(_GEMINI_MODEL)
        assert client._custom_llm_provider == "gemini"

    def test_bedrock_claude_resolves_to_bedrock(self):
        client = self._make_client(_BEDROCK_CLAUDE_MODEL)
        assert client._custom_llm_provider == "bedrock"

    def test_vertex_claude_resolves_to_vertex_ai(self):
        client = self._make_client(_VERTEX_CLAUDE_MODEL)
        assert client._custom_llm_provider == "vertex_ai"

    def test_responses_flavor_resolves_to_openai(self):
        client = self._make_client(_OPENAI_MODEL, api_flavor=ApiFlavor.RESPONSES)
        assert client._custom_llm_provider == "openai"

    def test_converse_flavor_resolves_to_bedrock(self):
        client = self._make_client(_BEDROCK_CLAUDE_MODEL, api_flavor=ApiFlavor.CONVERSE)
        assert client._custom_llm_provider == "bedrock"


# ============================================================================
# Model name resolution
# ============================================================================


class TestResolveLiteLLMModel:
    def _make_client(self, model_data: dict, **kwargs) -> UiPathLiteLLM:
        settings = _mock_settings([model_data])
        return UiPathLiteLLM(
            model_name=model_data["modelName"],
            client_settings=settings,
            **kwargs,
        )

    def test_openai_model_name_unchanged(self):
        client = self._make_client(_OPENAI_MODEL)
        assert client._litellm_model == "gpt-5.2-2025-12-11"

    def test_gemini_model_name_unchanged(self):
        client = self._make_client(_GEMINI_MODEL)
        assert client._litellm_model == "gemini-2.5-flash"

    def test_vertex_claude_model_name_unchanged(self):
        client = self._make_client(_VERTEX_CLAUDE_MODEL)
        assert client._litellm_model == "claude-sonnet-4-5@20250929"

    def test_bedrock_invoke_adds_prefix(self):
        client = self._make_client(_BEDROCK_CLAUDE_MODEL)
        assert client._litellm_model == "invoke/anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_bedrock_converse_adds_prefix(self):
        client = self._make_client(_BEDROCK_CLAUDE_MODEL, api_flavor=ApiFlavor.CONVERSE)
        assert client._litellm_model == "converse/anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_responses_adds_prefix(self):
        client = self._make_client(_OPENAI_MODEL, api_flavor=ApiFlavor.RESPONSES)
        assert client._litellm_model == "responses/gpt-5.2-2025-12-11"


# ============================================================================
# API config discovery
# ============================================================================


class TestDiscoverApiConfig:
    def _make_client(self, model_data: dict, **kwargs) -> UiPathLiteLLM:
        settings = _mock_settings([model_data])
        return UiPathLiteLLM(
            model_name=model_data["modelName"],
            client_settings=settings,
            **kwargs,
        )

    def test_openai_defaults_to_chat_completions(self):
        client = self._make_client(_OPENAI_MODEL)
        assert client._api_config.api_flavor == ApiFlavor.CHAT_COMPLETIONS
        assert client._api_config.vendor_type == "openai"
        assert client._api_config.routing_mode == RoutingMode.PASSTHROUGH

    def test_bedrock_claude_defaults_to_invoke(self):
        client = self._make_client(_BEDROCK_CLAUDE_MODEL)
        assert client._api_config.api_flavor == ApiFlavor.INVOKE
        assert client._api_config.vendor_type == "awsbedrock"

    def test_vertex_claude_defaults_to_anthropic_claude(self):
        client = self._make_client(_VERTEX_CLAUDE_MODEL)
        assert client._api_config.api_flavor == ApiFlavor.ANTHROPIC_CLAUDE
        assert client._api_config.vendor_type == "vertexai"

    def test_gemini_no_default_flavor(self):
        client = self._make_client(_GEMINI_MODEL)
        assert client._api_config.api_flavor is None
        assert client._api_config.vendor_type == "vertexai"

    def test_api_flavor_override(self):
        client = self._make_client(_OPENAI_MODEL, api_flavor=ApiFlavor.RESPONSES)
        assert client._api_config.api_flavor == "responses"

    def test_vendor_type_filter(self):
        """vendor_type param filters discovery results."""
        settings = _mock_settings([_OPENAI_MODEL, _GEMINI_MODEL])
        client = UiPathLiteLLM(
            model_name="gemini-2.5-flash",
            client_settings=settings,
            vendor_type=VendorType.VERTEXAI,
        )
        assert client._api_config.vendor_type == "vertexai"

    def test_model_not_found_raises(self):
        settings = _mock_settings([_OPENAI_MODEL])
        with pytest.raises(ValueError, match="not found"):
            UiPathLiteLLM(model_name="nonexistent-model", client_settings=settings)

    def test_freeze_base_url_always_true(self):
        client = self._make_client(_OPENAI_MODEL)
        assert client._api_config.freeze_base_url is True


# ============================================================================
# Model family detection
# ============================================================================


class TestModelFamily:
    def _make_client(self, model_data: dict) -> UiPathLiteLLM:
        settings = _mock_settings([model_data])
        return UiPathLiteLLM(
            model_name=model_data["modelName"],
            client_settings=settings,
        )

    def test_claude_family_detected(self):
        client = self._make_client(_BEDROCK_CLAUDE_MODEL)
        assert client._model_family == _ANTHROPIC_FAMILY

    def test_openai_family_not_anthropic(self):
        client = self._make_client(_OPENAI_MODEL)
        assert client._model_family != _ANTHROPIC_FAMILY

    def test_gemini_family_not_anthropic(self):
        client = self._make_client(_GEMINI_MODEL)
        assert client._model_family != _ANTHROPIC_FAMILY


# ============================================================================
# Embedding provider override
# ============================================================================


class TestEmbeddingProviderOverride:
    def _make_client(self, model_data: dict, **kwargs) -> UiPathLiteLLM:
        settings = _mock_settings([model_data])
        return UiPathLiteLLM(
            model_name=model_data["modelName"],
            client_settings=settings,
            **kwargs,
        )

    def test_openai_embedding_uses_hosted_vllm(self):
        client = self._make_client(_OPENAI_MODEL)
        assert client._embedding_llm_provider == "hosted_vllm"

    def test_gemini_embedding_unchanged(self):
        client = self._make_client(_GEMINI_MODEL)
        assert client._embedding_llm_provider == "gemini"

    def test_bedrock_embedding_unchanged(self):
        client = self._make_client(_BEDROCK_CLAUDE_MODEL)
        assert client._embedding_llm_provider == "bedrock"


# ============================================================================
# Mapping tables
# ============================================================================


class TestMappingTables:
    def test_vendor_to_litellm_covers_all_vendors(self):
        for vendor in VendorType:
            assert vendor.value in _VENDOR_TO_LITELLM

    def test_flavor_to_litellm_covers_all_flavors(self):
        for flavor in ApiFlavor:
            assert flavor.value in _FLAVOR_TO_LITELLM
