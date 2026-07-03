"""Unit tests for Bedrock backing-model resolution."""

import pytest
from uipath_langchain_client.clients.bedrock.model_resolution import (
    apply_backing_model_detection_hints,
)


class TestApplyBackingModelDetectionHints:
    @pytest.mark.parametrize(
        "customer_model,expected_provider",
        [
            ("anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic"),
            ("global.anthropic.claude-sonnet-4-6", "anthropic"),
            ("amazon.nova-pro-v1:0", "amazon"),
        ],
    )
    def test_byo_uses_customer_model(self, customer_model, expected_provider):
        kwargs: dict = {}
        apply_backing_model_detection_hints(
            kwargs,
            {
                "modelName": "AWS - Bedrock",
                "byomDetails": {
                    "customerModel": customer_model,
                    "integrationServiceConnectionId": "conn-x",
                },
            },
        )
        assert kwargs["base_model_id"] == customer_model
        assert kwargs["provider"] == expected_provider

    def test_unparseable_customer_model_sets_no_hints(self):
        kwargs: dict = {}
        apply_backing_model_detection_hints(
            kwargs,
            {
                "modelName": "AWS - Bedrock",
                "byomDetails": {"customerModel": "my-claude-sonnet-4-5"},
            },
        )
        assert "base_model_id" not in kwargs
        assert "provider" not in kwargs

    def test_non_byo_model_sets_no_hints(self):
        kwargs: dict = {}
        apply_backing_model_detection_hints(
            kwargs,
            {
                "modelName": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "byomDetails": None,
            },
        )
        assert "base_model_id" not in kwargs
        assert "provider" not in kwargs

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

    def test_caller_base_model_alias_is_not_shadowed_by_base_model_id(self):
        kwargs = {"base_model": "amazon.nova-pro-v1:0"}
        apply_backing_model_detection_hints(
            kwargs,
            {"byomDetails": {"customerModel": "anthropic.claude-sonnet-4-5-20250929-v1:0"}},
        )
        assert "base_model_id" not in kwargs
        assert kwargs["base_model"] == "amazon.nova-pro-v1:0"
