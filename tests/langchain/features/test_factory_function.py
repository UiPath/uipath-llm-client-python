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


class TestFactoryBedrockApiFlavorRouting:
    """The AWSBEDROCK branch routes purely on ``api_flavor``:

    - ``ApiFlavor.INVOKE`` -> ``UiPathChatBedrock``
    - ``ApiFlavor.CONVERSE`` or ``None`` -> ``UiPathChatBedrockConverse``

    Model family (e.g. ANTHROPIC_CLAUDE) no longer influences the choice.
    """

    def _patch_bedrock_classes(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Replace the three bedrock chat classes with sentinels and record which one was built."""
        chosen: dict = {}

        def _make_stub(name: str):
            class _Stub:
                def __init__(self, **kwargs):
                    chosen["class"] = name
                    chosen["kwargs"] = kwargs

            return _Stub

        for name in (
            "UiPathChatBedrockConverse",
            "UiPathChatBedrock",
            "UiPathChatAnthropicBedrock",
        ):
            monkeypatch.setattr(
                f"uipath_langchain_client.clients.bedrock.chat_models.{name}",
                _make_stub(name),
            )
        return chosen

    def _settings_with_model_info(self, model_info: dict):
        settings = MagicMock()
        settings.get_model_info.return_value = model_info
        return settings

    def test_no_api_flavor_uses_bedrock_converse(self, monkeypatch: pytest.MonkeyPatch):
        chosen = self._patch_bedrock_classes(monkeypatch)
        settings = self._settings_with_model_info(
            {
                "modelName": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "vendor": "AwsBedrock",
                "apiFlavor": None,
                "modelFamily": None,
            }
        )
        get_chat_model(
            model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
            client_settings=settings,
        )
        assert chosen["class"] == "UiPathChatBedrockConverse"

    def test_converse_api_flavor_uses_bedrock_converse(self, monkeypatch: pytest.MonkeyPatch):
        chosen = self._patch_bedrock_classes(monkeypatch)
        settings = self._settings_with_model_info(
            {
                "modelName": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "vendor": "AwsBedrock",
                "apiFlavor": None,
                "modelFamily": None,
            }
        )
        get_chat_model(
            model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
            client_settings=settings,
            api_flavor=ApiFlavor.CONVERSE,
        )
        assert chosen["class"] == "UiPathChatBedrockConverse"

    def test_invoke_api_flavor_uses_bedrock_invoke(self, monkeypatch: pytest.MonkeyPatch):
        chosen = self._patch_bedrock_classes(monkeypatch)
        settings = self._settings_with_model_info(
            {
                "modelName": "amazon.titan-text-express-v1",
                "vendor": "AwsBedrock",
                "apiFlavor": None,
                "modelFamily": None,
            }
        )
        get_chat_model(
            model_name="amazon.titan-text-express-v1",
            client_settings=settings,
            api_flavor=ApiFlavor.INVOKE,
        )
        assert chosen["class"] == "UiPathChatBedrock"


class TestFactoryAgentHubConfig:
    """The ``agenthub_config`` factory kwarg overrides ``client_settings.agenthub_config``
    via ``model_copy`` so the caller's instance is not mutated."""

    def _capture_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_info: dict,
        original_settings,
        **factory_kwargs,
    ):
        captured: dict = {}

        class _StubModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "uipath_langchain_client.clients.openai.chat_models.UiPathAzureChatOpenAI",
            _StubModel,
        )
        get_chat_model(
            model_name=model_info["modelName"],
            client_settings=original_settings,
            **factory_kwargs,
        )
        return captured

    def _make_settings(self, agenthub_config: str | None):
        settings = MagicMock()
        settings.get_model_info.return_value = {
            "modelName": "gpt-4o",
            "vendor": "OpenAi",
            "apiFlavor": None,
            "modelFamily": "OpenAi",
        }
        settings.agenthub_config = agenthub_config

        def _model_copy(*, update):
            copied = MagicMock()
            copied.get_model_info.return_value = settings.get_model_info.return_value
            copied.agenthub_config = update.get("agenthub_config", agenthub_config)
            return copied

        settings.model_copy.side_effect = _model_copy
        return settings

    def test_kwarg_overrides_settings_value(self, monkeypatch: pytest.MonkeyPatch):
        original = self._make_settings(agenthub_config="agentsruntime")
        captured = self._capture_settings(
            monkeypatch,
            original.get_model_info.return_value,
            original,
            agenthub_config="agentsplayground",
        )
        assert captured["settings"].agenthub_config == "agentsplayground"
        original.model_copy.assert_called_once_with(update={"agenthub_config": "agentsplayground"})

    def test_caller_settings_not_mutated(self, monkeypatch: pytest.MonkeyPatch):
        original = self._make_settings(agenthub_config="agentsruntime")
        self._capture_settings(
            monkeypatch,
            original.get_model_info.return_value,
            original,
            agenthub_config="agentsplayground",
        )
        assert original.agenthub_config == "agentsruntime"

    def test_no_kwarg_keeps_settings_value(self, monkeypatch: pytest.MonkeyPatch):
        original = self._make_settings(agenthub_config="agentsruntime")
        captured = self._capture_settings(
            monkeypatch,
            original.get_model_info.return_value,
            original,
        )
        assert captured["settings"] is original
        original.model_copy.assert_not_called()
