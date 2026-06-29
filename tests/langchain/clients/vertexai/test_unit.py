"""LangChain unit tests for VertexAI provider client."""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from uipath_langchain_client.clients.vertexai.chat_models import UiPathChatAnthropicVertex

from uipath.llm_client.settings import UiPathBaseSettings

VERTEXAI_CHAT_CLASSES = [UiPathChatAnthropicVertex]


class TestVertexAIChatModel(ChatModelUnitTests):
    @pytest.fixture(autouse=True, params=VERTEXAI_CHAT_CLASSES)
    def setup_models(self, request: pytest.FixtureRequest, client_settings: UiPathBaseSettings):
        self._completions_class = request.param
        self._completions_kwargs = {
            "model": "claude-haiku-4-5@20251001",
            "client_settings": client_settings,
        }

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self._completions_kwargs

    @pytest.mark.xfail(reason="Skipping serdes test for now")
    def test_serdes(self, *args: Any, **kwargs: Any) -> None: ...
