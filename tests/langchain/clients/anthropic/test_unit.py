"""LangChain unit tests for Anthropic provider client."""

from typing import Any

import pytest
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AsyncAnthropic,
    AsyncAnthropicBedrock,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic

from uipath.llm_client.settings import ApiFlavor, UiPathBaseSettings, VendorType

ANTHROPIC_CHAT_CLASSES = [UiPathChatAnthropic]


class TestAnthropicChatModel(ChatModelUnitTests):
    @pytest.fixture(autouse=True, params=ANTHROPIC_CHAT_CLASSES)
    def setup_models(self, request: pytest.FixtureRequest, client_settings: UiPathBaseSettings):
        self._completions_class = request.param
        self._completions_kwargs = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "client_settings": client_settings,
            "vendor_type": "awsbedrock",
        }

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self._completions_kwargs

    @pytest.mark.xfail(reason="Skipping serdes test for now")
    def test_serdes(self, *args: Any, **kwargs: Any) -> None: ...


def _build(client_settings: UiPathBaseSettings, **kwargs: Any) -> UiPathChatAnthropic:
    return UiPathChatAnthropic(
        model="anthropic.claude-sonnet-4-6",
        settings=client_settings,
        model_details={},
        **kwargs,
    )


class TestAnthropicMessagesFlavor:
    """AnthropicMessages uses the native Anthropic SDK (model-in-body wire format)
    over the awsbedrock passthrough URL."""

    def test_sets_anthropic_messages_flavor_over_bedrock_url(
        self, client_settings: UiPathBaseSettings
    ):
        chat = _build(
            client_settings,
            vendor_type=VendorType.AWSBEDROCK,
            api_flavor=ApiFlavor.ANTHROPIC_MESSAGES,
        )
        assert chat.api_config.vendor_type == VendorType.AWSBEDROCK
        assert chat.api_config.api_flavor == ApiFlavor.ANTHROPIC_MESSAGES

    def test_uses_native_anthropic_sdk(self, client_settings: UiPathBaseSettings):
        chat = _build(
            client_settings,
            vendor_type=VendorType.AWSBEDROCK,
            api_flavor=ApiFlavor.ANTHROPIC_MESSAGES,
        )
        assert isinstance(chat._anthropic_client, Anthropic)
        assert not isinstance(chat._anthropic_client, AnthropicBedrock)
        assert isinstance(chat._async_anthropic_client, AsyncAnthropic)
        assert not isinstance(chat._async_anthropic_client, AsyncAnthropicBedrock)

    def test_flavor_is_orthogonal_to_vendor_type(self, client_settings: UiPathBaseSettings):
        chat = _build(
            client_settings,
            vendor_type=VendorType.ANTHROPIC,
            api_flavor=ApiFlavor.ANTHROPIC_MESSAGES,
        )
        assert chat.api_config.api_flavor == ApiFlavor.ANTHROPIC_MESSAGES
        assert isinstance(chat._anthropic_client, Anthropic)


class TestVendorDerivedDefaultsUnchanged:
    """Regression guard: omitting api_flavor preserves the prior vendor-derived behavior."""

    def test_bedrock_without_flavor_uses_invoke_and_anthropic_bedrock(
        self, client_settings: UiPathBaseSettings
    ):
        chat = _build(client_settings, vendor_type=VendorType.AWSBEDROCK)
        assert chat.api_config.api_flavor == ApiFlavor.INVOKE
        assert isinstance(chat._anthropic_client, AnthropicBedrock)
        assert isinstance(chat._async_anthropic_client, AsyncAnthropicBedrock)

    def test_anthropic_vendor_defaults_to_anthropic_client(
        self, client_settings: UiPathBaseSettings
    ):
        chat = _build(client_settings, vendor_type=VendorType.ANTHROPIC)
        assert isinstance(chat._anthropic_client, Anthropic)
        assert not isinstance(chat._anthropic_client, AnthropicBedrock)
