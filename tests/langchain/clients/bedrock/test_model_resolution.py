"""Unit tests for Bedrock backing-model resolution.

Exercised through the module's public surface (``apply_backing_model_detection_hints``
and ``provider_from_model``); the internal ``_resolve_backing_model_id`` /
``_normalize_model_id`` helpers are covered transitively by the hints they produce.
"""

import pytest
from uipath_langchain_client.clients.bedrock.model_resolution import (
    apply_backing_model_detection_hints,
    provider_from_model,
)


class TestApplyBackingModelDetectionHints:
    def test_byo_uses_customer_model(self):
        kwargs: dict = {}
        apply_backing_model_detection_hints(
            kwargs,
            {
                "modelName": "AWS - Bedrock",
                "byomDetails": {
                    "customerModel": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "integrationServiceConnectionId": "conn-x",
                },
            },
        )
        assert kwargs["base_model_id"] == "anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert kwargs["provider"] == "anthropic"

    def test_uipath_owned_uses_model_name(self):
        kwargs: dict = {}
        apply_backing_model_detection_hints(
            kwargs,
            {
                "modelName": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "byomDetails": None,
            },
        )
        assert kwargs["base_model_id"] == "anthropic.claude-3-5-sonnet-20240620-v1:0"
        assert kwargs["provider"] == "anthropic"

    def test_falls_back_to_model_name(self):
        kwargs: dict = {}
        apply_backing_model_detection_hints(kwargs, {"modelName": "amazon.nova-pro-v1:0"})
        assert kwargs["base_model_id"] == "amazon.nova-pro-v1:0"
        assert kwargs["provider"] == "amazon"

    def test_byo_alias_without_customer_model_sets_no_hints(self):
        kwargs: dict = {}
        apply_backing_model_detection_hints(
            kwargs,
            {
                "modelName": "VeryCustomBedddrockAlias",
                "byomDetails": {"integrationServiceConnectionId": "conn-x"},
            },
        )
        assert "base_model_id" not in kwargs
        assert "provider" not in kwargs

    def test_does_not_override_caller_supplied_hints(self):
        kwargs = {"base_model_id": "amazon.nova-pro-v1:0", "provider": "amazon"}
        apply_backing_model_detection_hints(
            kwargs,
            {"byomDetails": {"customerModel": "anthropic.claude-sonnet-4-5-20250929-v1:0"}},
        )
        assert kwargs["base_model_id"] == "amazon.nova-pro-v1:0"
        assert kwargs["provider"] == "amazon"


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic"),
        ("global.anthropic.claude-sonnet-4-6", "anthropic"),
        ("amazon.nova-pro-v1:0", "amazon"),
        ("AWS - Bedrock", None),
        ("my-claude-sonnet-4-5", None),
        (None, None),
    ],
)
def test_provider_from_model(model_id, expected):
    assert provider_from_model(model_id) == expected
