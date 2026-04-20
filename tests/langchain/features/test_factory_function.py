from unittest.mock import MagicMock

import pytest
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.normalized.embeddings import UiPathEmbeddings
from uipath_langchain_client.factory import get_chat_model, get_embedding_model

from tests.langchain.conftest import COMPLETION_MODEL_NAMES, EMBEDDING_MODEL_NAMES
from uipath.llm_client.settings import ApiFlavor, UiPathBaseSettings


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


class TestFactoryDefaultApiFlavor:
    """Unit tests for the default api_flavor picked by the chat factory.

    The factory returns concrete LangChain model classes whose construction is
    non-trivial. Instead of fully instantiating them, we patch the concrete
    classes with a sentinel that captures the kwargs the factory passes.
    """

    def _captured_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_info: dict,
        **factory_kwargs,
    ) -> dict:
        settings = MagicMock()
        settings.get_model_info.return_value = model_info
        captured: dict = {}

        class _StubModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "uipath_langchain_client.clients.openai.chat_models.UiPathChatOpenAI",
            _StubModel,
        )
        monkeypatch.setattr(
            "uipath_langchain_client.clients.openai.chat_models.UiPathAzureChatOpenAI",
            _StubModel,
        )
        get_chat_model(
            model_name=model_info["modelName"],
            client_settings=settings,
            **factory_kwargs,
        )
        return captured

    def test_openai_chat_defaults_to_responses_when_no_flavor_discovered(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """UiPath-owned OpenAI (apiFlavor=null) should default to the Responses API."""
        captured = self._captured_kwargs(
            monkeypatch,
            {
                "modelName": "gpt-4o",
                "vendor": "OpenAi",
                "apiFlavor": None,
                "modelFamily": "OpenAi",
            },
        )
        assert captured["api_flavor"] == ApiFlavor.RESPONSES

    def test_openai_chat_respects_user_api_flavor_override(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit api_flavor from the caller still wins over the default."""
        captured = self._captured_kwargs(
            monkeypatch,
            {
                "modelName": "gpt-4o",
                "vendor": "OpenAi",
                "apiFlavor": None,
                "modelFamily": "OpenAi",
            },
            api_flavor=ApiFlavor.CHAT_COMPLETIONS,
        )
        assert captured["api_flavor"] == ApiFlavor.CHAT_COMPLETIONS

    def test_openai_chat_respects_discovered_byom_chat_completions(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """BYOM-discovered chat-completions still maps to chat-completions."""
        captured = self._captured_kwargs(
            monkeypatch,
            {
                "modelName": "custom-gpt",
                "vendor": "OpenAi",
                "apiFlavor": "OpenAiChatCompletions",
                "modelFamily": None,
            },
        )
        assert captured["api_flavor"] == ApiFlavor.CHAT_COMPLETIONS
