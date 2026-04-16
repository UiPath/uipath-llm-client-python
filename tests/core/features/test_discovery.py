"""Tests for the shared get_model_info utility."""

import pytest

from uipath.llm_client.utils.discovery import get_model_info

_MODELS = [
    {"modelName": "gpt-4o", "vendor": "openai", "modelSubscriptionType": "UiPathOwned"},
    {
        "modelName": "gpt-4o",
        "vendor": "openai",
        "byomDetails": {"integrationServiceConnectionId": "conn-1"},
    },
    {"modelName": "claude-3-opus", "vendor": "anthropic", "modelSubscriptionType": "UiPathOwned"},
    {"modelName": "gemini-2.0-flash", "vendor": "vertexai", "modelSubscriptionType": "UiPathOwned"},
]


class TestGetModelInfo:
    """Tests for get_model_info."""

    def test_finds_model_by_name(self):
        result = get_model_info(_MODELS, "claude-3-opus")
        assert result["modelName"] == "claude-3-opus"

    def test_case_insensitive_match(self):
        result = get_model_info(_MODELS, "Claude-3-Opus")
        assert result["modelName"] == "claude-3-opus"

    def test_filters_by_vendor_type(self):
        result = get_model_info(_MODELS, "gpt-4o", vendor_type="openai")
        assert result["vendor"] == "openai"

    def test_vendor_type_case_insensitive(self):
        result = get_model_info(_MODELS, "gpt-4o", vendor_type="OpenAi")
        assert result["vendor"] == "openai"

    def test_filters_by_byo_connection_id(self):
        result = get_model_info(_MODELS, "gpt-4o", byo_connection_id="conn-1")
        assert result["byomDetails"]["integrationServiceConnectionId"] == "conn-1"

    def test_byo_connection_id_case_insensitive(self):
        result = get_model_info(_MODELS, "gpt-4o", byo_connection_id="CONN-1")
        assert result["byomDetails"]["integrationServiceConnectionId"] == "conn-1"

    def test_prefers_uipath_owned_when_no_byo_connection_id(self):
        result = get_model_info(_MODELS, "gpt-4o")
        assert result.get("modelSubscriptionType") == "UiPathOwned"
        assert result.get("byomDetails") is None

    def test_raises_when_model_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            get_model_info(_MODELS, "nonexistent-model")

    def test_raises_when_vendor_filter_eliminates_all(self):
        with pytest.raises(ValueError, match="not found"):
            get_model_info(_MODELS, "gpt-4o", vendor_type="anthropic")

    def test_raises_when_byo_connection_id_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            get_model_info(_MODELS, "gpt-4o", byo_connection_id="nonexistent-conn")

    def test_returns_first_match_for_single_result(self):
        models = [{"modelName": "my-model", "vendor": "openai"}]
        result = get_model_info(models, "my-model")
        assert result["modelName"] == "my-model"

    def test_error_message_includes_available_model_names(self):
        with pytest.raises(ValueError, match="gpt-4o") as exc_info:
            get_model_info(_MODELS, "missing")
        assert "claude-3-opus" in str(exc_info.value)
