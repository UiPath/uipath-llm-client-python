"""Unit tests for Converse model_settings -> request-shape mapping.

Bedrock Converse has no top-level ``thinking`` field, so a transport-agnostic
``thinking`` from agent.json must be routed into ``additional_model_request_fields``
rather than ``model_kwargs``. These test the pure partition helper against the real
class field set (static — no client construction / network).
"""

from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatBedrockConverse,
    _partition_converse_settings,
)

FIELDS = UiPathChatBedrockConverse.model_fields


def test_field_assumptions() -> None:
    # Documents the invariants the mapping relies on.
    assert "thinking" not in FIELDS
    assert "additional_model_request_fields" in FIELDS


def test_reasoning_bundle_goes_to_passthrough() -> None:
    direct, passthrough = _partition_converse_settings(
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        FIELDS,
        {},
    )
    # Both must nest in additional_model_request_fields — output_config as a
    # top-level Converse field makes the provider 400.
    assert passthrough == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert direct == {}


def test_output_config_is_a_field_but_still_nested() -> None:
    # Guards the regression: output_config IS a field, yet must go to passthrough.
    assert "output_config" in FIELDS
    _, passthrough = _partition_converse_settings(
        {"output_config": {"effort": "high"}}, FIELDS, {}
    )
    assert passthrough == {"output_config": {"effort": "high"}}


def test_explicit_additional_fields_stay_direct() -> None:
    # Backward compatibility: an explicit wrapper is a real field -> set directly.
    settings = {"additional_model_request_fields": {"thinking": {"type": "enabled"}}}
    direct, passthrough = _partition_converse_settings(settings, FIELDS, {})
    assert direct == settings
    assert passthrough == {}


def test_disabled_key_is_dropped() -> None:
    direct, passthrough = _partition_converse_settings(
        {"temperature": 0.5, "thinking": {"type": "adaptive"}},
        FIELDS,
        {"temperature": True},
    )
    assert "temperature" not in direct
    assert "temperature" not in passthrough
    assert passthrough == {"thinking": {"type": "adaptive"}}
