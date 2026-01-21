from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests, EmbeddingsUnitTests

from tests.langchain.conftest import COMPLETION_CLIENTS_CLASSES, EMBEDDINGS_CLIENTS_CLASSES
from uipath_llm_client.settings import UiPathBaseSettings


class TestCoreChatModel(ChatModelUnitTests):
    @pytest.fixture(autouse=True, params=COMPLETION_CLIENTS_CLASSES)
    def setup_models(self, request: pytest.FixtureRequest, client_settings: UiPathBaseSettings):
        self._completions_class = request.param
        self._completions_kwargs = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
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


class TestCoreEmbeddings(EmbeddingsUnitTests):
    @pytest.fixture(autouse=True, params=EMBEDDINGS_CLIENTS_CLASSES)
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
