import pytest
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.normalized.embeddings import UiPathEmbeddings
from uipath_langchain_client.factory import get_chat_model, get_embedding_model

from tests.langchain.conftest import COMPLETION_MODEL_NAMES, EMBEDDING_MODEL_NAMES
from uipath.llm_client.settings import UiPathBaseSettings


@pytest.mark.vcr
class TestFactoryFunction:
    @pytest.mark.parametrize("model_name", COMPLETION_MODEL_NAMES)
    def test_get_chat_model(self, model_name: str, client_settings: UiPathBaseSettings):
        chat_model = get_chat_model(model_name=model_name, client_settings=client_settings)
        assert chat_model is not None

    @pytest.mark.parametrize("model_name", EMBEDDING_MODEL_NAMES)
    def test_get_embedding_model(self, model_name: str, client_settings: UiPathBaseSettings):
        embedding_model = get_embedding_model(
            model_name=model_name, client_settings=client_settings
        )
        assert embedding_model is not None

    @pytest.mark.parametrize("model_name", COMPLETION_MODEL_NAMES)
    def test_get_chat_model_custom_class(
        self, model_name: str, client_settings: UiPathBaseSettings
    ):
        chat_model = get_chat_model(
            model_name=model_name,
            client_settings=client_settings,
            custom_class=UiPathChat,
        )
        assert chat_model is not None
        assert isinstance(chat_model, UiPathChat)

    @pytest.mark.parametrize("model_name", EMBEDDING_MODEL_NAMES)
    def test_get_embedding_model_custom_class(
        self, model_name: str, client_settings: UiPathBaseSettings
    ):
        embedding_model = get_embedding_model(
            model_name=model_name,
            client_settings=client_settings,
            custom_class=UiPathEmbeddings,
        )
        assert embedding_model is not None
        assert isinstance(embedding_model, UiPathEmbeddings)
