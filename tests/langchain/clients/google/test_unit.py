"""LangChain unit tests for Google provider clients."""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests, EmbeddingsUnitTests
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.google.embeddings import UiPathGoogleGenerativeAIEmbeddings

from uipath.llm_client.settings import UiPathBaseSettings

GOOGLE_CHAT_CLASSES = [UiPathChatGoogleGenerativeAI]
GOOGLE_EMBEDDINGS_CLASSES = [UiPathGoogleGenerativeAIEmbeddings]


class TestGoogleChatModel(ChatModelUnitTests):
    @pytest.fixture(autouse=True, params=GOOGLE_CHAT_CLASSES)
    def setup_models(self, request: pytest.FixtureRequest, client_settings: UiPathBaseSettings):
        self._completions_class = request.param
        self._completions_kwargs = {
            "model": "gemini-2.5-flash",
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


class TestGoogleEmbeddings(EmbeddingsUnitTests):
    @pytest.fixture(autouse=True, params=GOOGLE_EMBEDDINGS_CLASSES)
    def setup_models(self, request: pytest.FixtureRequest, client_settings: UiPathBaseSettings):
        self._embeddings_class = request.param
        self._embeddings_kwargs = {
            "model": "PLACEHOLDER",
            "client_settings": client_settings,
        }

    @property
    def embeddings_class(self) -> type[Embeddings]:
        return self._embeddings_class

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return self._embeddings_kwargs
