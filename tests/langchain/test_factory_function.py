import pytest
from uipath_langchain_client.factory import get_chat_model, get_embedding_model

from tests.langchain.conftest import COMPLETION_MODEL_NAMES, EMBEDDING_MODEL_NAMES


@pytest.mark.vcr
class TestFactoryFunction:
    @pytest.mark.parametrize("model_name", COMPLETION_MODEL_NAMES)
    def test_get_chat_model(self, model_name: str):
        chat_model = get_chat_model(model_name)
        assert chat_model is not None

    @pytest.mark.parametrize("model_name", EMBEDDING_MODEL_NAMES)
    def test_get_embedding_model(self, model_name: str):
        embedding_model = get_embedding_model(model_name)
        assert embedding_model is not None
