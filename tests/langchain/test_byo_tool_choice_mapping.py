"""Unit tests for reading a BYO configuration's tool choice mapping into ``model_details``.

LLM Gateway's BYO probe stores ``customFieldMappings["ToolChoice"] = "auto"`` on a
configuration whose model rejects a forced tool choice. Discovery returns that
mapping under ``byomDetails``, also for BYO-added models whose ``modelDetails`` is
null, and the agent runtime reads ``model_details["shouldSkipForcedToolChoice"]``.
"""

from typing import Any

import pytest
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.factory import get_chat_model

from uipath.llm_client.settings import UiPathBaseSettings


def _stub_model_info(
    monkeypatch: pytest.MonkeyPatch,
    settings: UiPathBaseSettings,
    *,
    model_details: dict[str, Any] | None,
    custom_field_mappings: dict[str, str] | None,
) -> None:
    def _stub(model_name: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "modelName": model_name,
            "vendor": "AwsBedrock",
            "modelSubscriptionType": "BYOMAdded",
            "modelDetails": model_details,
            "byomDetails": {
                "availableOperationCodes": ["agents-design-eval-deploy"],
                "integrationServiceConnectionId": "b47fd787-6b3e-4f28-a608-22bd154bd186",
                "defaultModel": model_name,
                "customFieldMappings": custom_field_mappings,
            },
        }

    monkeypatch.setattr(settings, "get_model_info", _stub)


def test_factory_skips_forced_tool_choice_for_added_model_with_auto_mapping(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    from uipath_langchain_client.settings import RoutingMode

    _stub_model_info(
        monkeypatch,
        client_settings,
        model_details=None,
        custom_field_mappings={"ToolChoice": "auto"},
    )

    llm = get_chat_model(
        "Test", client_settings=client_settings, routing_mode=RoutingMode.NORMALIZED
    )

    assert llm.model_details == {"shouldSkipForcedToolChoice": True}


def test_validator_skips_forced_tool_choice_for_added_model_with_auto_mapping(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    _stub_model_info(
        monkeypatch,
        client_settings,
        model_details=None,
        custom_field_mappings={"ToolChoice": "auto"},
    )

    llm = UiPathChat(model="Test", settings=client_settings)

    assert llm.model_details == {"shouldSkipForcedToolChoice": True}


def test_factory_keeps_other_model_details_when_adding_the_flag(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    from uipath_langchain_client.settings import RoutingMode

    _stub_model_info(
        monkeypatch,
        client_settings,
        model_details={
            "maxOutputTokens": 64000,
            "shouldSkipTemperature": True,
            "shouldSkipForcedToolChoice": False,
        },
        custom_field_mappings={"MaxTokens": "MaxCompletionTokens", "ToolChoice": "auto"},
    )

    llm = get_chat_model(
        "claude-sonnet-5-5", client_settings=client_settings, routing_mode=RoutingMode.NORMALIZED
    )

    assert llm.model_details == {
        "maxOutputTokens": 64000,
        "shouldSkipTemperature": True,
        "shouldSkipForcedToolChoice": True,
    }
    assert llm.disabled_params is not None
    assert "temperature" in llm.disabled_params


@pytest.mark.parametrize(
    "custom_field_mappings",
    [None, {}, {"MaxTokens": "MaxCompletionTokens"}, {"ToolChoice": "required"}],
)
def test_factory_leaves_model_details_alone_without_auto_mapping(
    monkeypatch: pytest.MonkeyPatch,
    client_settings: UiPathBaseSettings,
    custom_field_mappings: dict[str, str] | None,
) -> None:
    from uipath_langchain_client.settings import RoutingMode

    _stub_model_info(
        monkeypatch,
        client_settings,
        model_details=None,
        custom_field_mappings=custom_field_mappings,
    )

    llm = get_chat_model(
        "Test", client_settings=client_settings, routing_mode=RoutingMode.NORMALIZED
    )

    assert llm.model_details == {}


def test_factory_handles_null_byom_details(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    from uipath_langchain_client.settings import RoutingMode

    def _stub(model_name: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "modelName": model_name,
            "vendor": "AwsBedrock",
            "modelSubscriptionType": "UiPathOwned",
            "modelDetails": {"shouldSkipForcedToolChoice": False},
            "byomDetails": None,
        }

    monkeypatch.setattr(client_settings, "get_model_info", _stub)

    llm = get_chat_model(
        "claude-sonnet-5-5", client_settings=client_settings, routing_mode=RoutingMode.NORMALIZED
    )

    assert llm.model_details == {"shouldSkipForcedToolChoice": False}
