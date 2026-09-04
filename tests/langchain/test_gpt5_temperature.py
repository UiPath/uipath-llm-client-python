"""Temperature survives on gpt-5 variants that default to no reasoning.

Covers both sites langchain-openai strips it, and pins the cases where stripping
is correct. Upstream: langchain-ai/langchain#35423.
"""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from pydantic import SecretStr
from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
)
from uipath_langchain_client.settings import ApiFlavor

from uipath.llm_client.settings import UiPathBaseSettings

MESSAGES = [HumanMessage(content="hi")]


def _build(client_settings: UiPathBaseSettings, **kwargs: Any) -> Any:
    chat_class = kwargs.pop("chat_class", UiPathChatOpenAI)
    return chat_class(
        model=kwargs.pop("model", "gpt-5.4"),
        client_settings=client_settings,
        model_details=kwargs.pop("model_details", {}),
        api_flavor=ApiFlavor.RESPONSES,
        **kwargs,
    )


def _payload(chat: Any) -> dict[str, Any]:
    return chat._get_request_payload(MESSAGES)


@pytest.mark.parametrize("chat_class", [UiPathChatOpenAI, UiPathAzureChatOpenAI])
def test_dotted_gpt5_keeps_temperature(
    chat_class: type, client_settings: UiPathBaseSettings
) -> None:
    chat = _build(client_settings, chat_class=chat_class, temperature=0.64)
    assert chat.temperature == 0.64
    assert _payload(chat)["temperature"] == 0.64


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasoning_effort": "low"},
        {"reasoning": {"effort": "low"}},
        {"model": "gpt-5"},
        {"model": "gpt-5.4-pro"},
    ],
    ids=["effort", "reasoning-dict", "base-gpt-5", "pro"],
)
def test_dropped_where_temperature_is_unsupported(
    overrides: dict[str, Any], client_settings: UiPathBaseSettings
) -> None:
    chat = _build(client_settings, temperature=0.64, **overrides)
    assert chat.temperature is None
    assert "temperature" not in _payload(chat)


def test_discovery_skip_flag_blocks_the_restore(
    client_settings: UiPathBaseSettings,
) -> None:
    chat = _build(client_settings, model_details={"shouldSkipTemperature": True})
    assert "temperature" not in chat._get_request_payload(MESSAGES, temperature=0.5)


def test_langchain_handling_still_delegated(client_settings: UiPathBaseSettings) -> None:
    assert _build(client_settings, model="o1").temperature == 1
    assert _build(client_settings, model="gpt-5-chat", temperature=0.64).temperature == 0.64


def test_upstream_bug_still_present() -> None:
    """Fails when langchain-openai fixes #35423; delete the shim and this file then."""
    plain = ChatOpenAI(
        model="gpt-5.4",
        api_key=SecretStr("x"),
        temperature=0.64,
        use_responses_api=True,
    )
    assert plain.temperature is None
    assert "temperature" not in plain._get_request_payload(MESSAGES)
