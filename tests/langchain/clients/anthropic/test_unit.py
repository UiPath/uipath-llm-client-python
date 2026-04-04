"""LangChain unit tests for Anthropic provider client."""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic

from uipath.llm_client.settings import UiPathBaseSettings

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
