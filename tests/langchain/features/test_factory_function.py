from unittest.mock import MagicMock

import pytest
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.normalized.embeddings import UiPathEmbeddings
from uipath_langchain_client.factory import (
    get_chat_model,
    get_embedding_model,
)

from tests.langchain.conftest import COMPLETION_MODEL_NAMES, EMBEDDING_MODEL_NAMES
from uipath.llm_client.settings import ApiFlavor, UiPathBaseSettings, VendorType


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
    """The AWSBEDROCK branch routes on ``api_flavor`` and ``model_family``:

    - ``ApiFlavor.INVOKE`` + ``ANTHROPIC_CLAUDE`` -> ``UiPathChatAnthropicBedrock``
    - ``ApiFlavor.INVOKE`` (other families) -> ``UiPathChatBedrock``
    - ``ApiFlavor.CONVERSE`` or ``None`` -> ``UiPathChatBedrockConverse``
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

    def test_invoke_api_flavor_with_anthropic_claude_uses_anthropic_bedrock(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        chosen = self._patch_bedrock_classes(monkeypatch)
        settings = self._settings_with_model_info(
            {
                "modelName": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "vendor": "AwsBedrock",
                "apiFlavor": None,
                "modelFamily": "AnthropicClaude",
            }
        )
        get_chat_model(
            model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
            client_settings=settings,
            api_flavor=ApiFlavor.INVOKE,
        )
        assert chosen["class"] == "UiPathChatAnthropicBedrock"

    def test_converse_api_flavor_with_anthropic_claude_still_uses_converse(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """ANTHROPIC_CLAUDE only diverts to AnthropicBedrock on INVOKE — CONVERSE still wins."""
        chosen = self._patch_bedrock_classes(monkeypatch)
        settings = self._settings_with_model_info(
            {
                "modelName": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "vendor": "AwsBedrock",
                "apiFlavor": None,
                "modelFamily": "AnthropicClaude",
            }
        )
        get_chat_model(
            model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
            client_settings=settings,
            api_flavor=ApiFlavor.CONVERSE,
        )
        assert chosen["class"] == "UiPathChatBedrockConverse"


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


class TestFactoryAnthropicMessagesRouting:
    """AwsBedrock + ``apiFlavor=AnthropicMessages`` routes to ``UiPathChatAnthropic``
    configured for the native Anthropic Messages wire format over the Bedrock
    passthrough URL (not the Bedrock Converse/Invoke clients)."""

    def _capture(
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
            "uipath_langchain_client.clients.anthropic.chat_models.UiPathChatAnthropic",
            _StubModel,
        )
        get_chat_model(
            model_name=model_info["modelName"],
            client_settings=settings,
            **factory_kwargs,
        )
        return captured

    def test_anthropic_messages_routes_to_uipath_chat_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        captured = self._capture(
            monkeypatch,
            {
                "modelName": "anthropic.claude-sonnet-4-6",
                "vendor": "AwsBedrock",
                "apiFlavor": "AnthropicMessages",
                "modelFamily": "AnthropicClaude",
            },
        )
        assert captured["vendor_type"] == VendorType.AWSBEDROCK
        assert captured["api_flavor"] == ApiFlavor.ANTHROPIC_MESSAGES


_BYO_BEDROCK_CONVERSE = {
    "modelName": "AWS - Bedrock",
    "vendor": "AwsBedrock",
    "modelFamily": None,
    "apiFlavor": None,
    "modelSubscriptionType": "BYOMAdded",
    "byomDetails": {
        "customerModel": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "integrationServiceConnectionId": "conn-x",
    },
}
_BYO_BEDROCK_INVOKE = {**_BYO_BEDROCK_CONVERSE, "apiFlavor": "AwsBedrockInvoke"}


class TestBedrockFactoryBaseModel:
    @pytest.fixture()
    def client_settings(self):
        import os
        from unittest.mock import patch

        from uipath.llm_client.settings.llmgateway import LLMGatewaySettings

        env = {
            "LLMGW_URL": "http://test-bedrock",
            "LLMGW_SEMANTIC_ORG_ID": "org",
            "LLMGW_SEMANTIC_TENANT_ID": "tenant",
            "LLMGW_REQUESTING_PRODUCT": "test",
            "LLMGW_REQUESTING_FEATURE": "test",
            "LLMGW_ACCESS_TOKEN": "dummy-token",
        }
        with patch.dict(os.environ, env, clear=True):
            return LLMGatewaySettings()

    @pytest.fixture(autouse=True)
    def _clear_discovery_cache(self):
        UiPathBaseSettings._discovery_cache.clear()
        yield
        UiPathBaseSettings._discovery_cache.clear()

    def _seed(self, client_settings, model_info):
        key = client_settings._discovery_cache_key()
        client_settings._discovery_cache[key] = [model_info]

    def test_converse_byo_alias_gets_backing_base_model(self, client_settings):
        self._seed(client_settings, _BYO_BEDROCK_CONVERSE)
        model = get_chat_model(
            "AWS - Bedrock",
            byo_connection_id="conn-x",
            client_settings=client_settings,
        )
        assert isinstance(model, UiPathChatBedrockConverse)
        assert model.model_id == "AWS - Bedrock"
        assert model.base_model_id == "anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert model.supports_tool_choice_values == ("auto", "any", "tool")

        from langchain_core.tools import tool

        @tool
        def ping() -> str:
            """ping."""
            return "pong"

        model.bind_tools([ping], tool_choice="any")

    def test_converse_direct_construction_takes_explicit_backing_model(self, client_settings):
        self._seed(client_settings, _BYO_BEDROCK_CONVERSE)
        model = UiPathChatBedrockConverse(
            model="AWS - Bedrock",
            settings=client_settings,
            byo_connection_id="conn-x",
            base_model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            provider="anthropic",
        )
        assert model.base_model_id == "anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert model.provider == "anthropic"
        assert model.supports_tool_choice_values == ("auto", "any", "tool")

    def test_invoke_byo_alias_gets_provider(self, client_settings):
        self._seed(client_settings, _BYO_BEDROCK_INVOKE)
        model = get_chat_model(
            "AWS - Bedrock",
            byo_connection_id="conn-x",
            client_settings=client_settings,
        )
        assert isinstance(model, UiPathChatBedrock)
        assert model.model_id == "AWS - Bedrock"
        assert model.base_model_id == "anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert model.provider == "anthropic"
        assert model._get_provider() == "anthropic"


class TestModelSettingsForwarding:
    """get_chat_model forwards model_settings into the chosen client's constructor."""

    def test_factory_forwards_model_settings_to_constructor(self, monkeypatch: pytest.MonkeyPatch):
        settings = MagicMock()
        settings.get_model_info.return_value = {
            "modelName": "gpt-4o",
            "vendor": "OpenAi",
            "apiFlavor": "responses",
            "modelFamily": "OpenAi",
        }
        captured: dict = {}

        class _StubModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "uipath_langchain_client.clients.openai.chat_models.UiPathAzureChatOpenAI",
            _StubModel,
        )
        get_chat_model(
            model_name="gpt-4o",
            client_settings=settings,
            model_settings={"reasoning_effort": "high", "temperature": 1.0},
        )
        assert captured["model_settings"] == {
            "reasoning_effort": "high",
            "temperature": 1.0,
        }


class TestModelSettingsApplied:
    """model_settings is applied during real construction (via the model_validator).

    Native provider keys land as real fields (no per-provider mapping); unknown keys
    route to model_kwargs; keys named in disabled_params are skipped.
    """

    @pytest.fixture()
    def settings(self) -> UiPathBaseSettings:
        import os
        from unittest.mock import patch

        from uipath.llm_client.settings.llmgateway import LLMGatewaySettings

        env = {
            "LLMGW_URL": "http://test",
            "LLMGW_SEMANTIC_ORG_ID": "org",
            "LLMGW_SEMANTIC_TENANT_ID": "tenant",
            "LLMGW_REQUESTING_PRODUCT": "test",
            "LLMGW_REQUESTING_FEATURE": "test",
            "LLMGW_ACCESS_TOKEN": "dummy-token",
        }
        with patch.dict(os.environ, env, clear=True):
            return LLMGatewaySettings()

    def test_openai_native_key_set_unknown_key_to_model_kwargs(self, settings: UiPathBaseSettings):
        from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

        model = UiPathChatOpenAI(
            model="some-openai-model",
            settings=settings,
            model_details={},
            model_settings={"reasoning_effort": "high", "made_up_key": 1},
        )
        assert model.reasoning_effort == "high"
        assert model.model_kwargs == {"made_up_key": 1}

    def test_anthropic_native_keys_set_verbatim(self, settings: UiPathBaseSettings):
        from uipath_langchain_client.clients.anthropic.chat_models import (
            UiPathChatAnthropic,
        )

        model = UiPathChatAnthropic(
            model="anthropic.claude-sonnet-4-6",
            settings=settings,
            model_details={},
            model_settings={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        )
        assert model.thinking == {"type": "adaptive"}
        assert model.output_config == {"effort": "high"}

    def test_bedrock_additional_model_request_fields_set_verbatim(
        self, settings: UiPathBaseSettings
    ):
        UiPathBaseSettings._discovery_cache.clear()
        settings._discovery_cache[settings._discovery_cache_key()] = [
            {
                "modelName": "AWS - Bedrock",
                "vendor": "Bedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelFamily": "Anthropic",
                "modelDetails": {"customerModelName": "anthropic.claude-sonnet-4-5-20250929-v1:0"},
            }
        ]
        amrf = {"thinking": {"type": "enabled", "budget_tokens": 4096}}
        try:
            model = UiPathChatBedrockConverse(
                model="AWS - Bedrock",
                settings=settings,
                byo_connection_id="conn-x",
                base_model="anthropic.claude-sonnet-4-5-20250929-v1:0",
                provider="anthropic",
                model_settings={"additional_model_request_fields": amrf},
            )
        finally:
            UiPathBaseSettings._discovery_cache.clear()
        assert model.additional_model_request_fields == amrf

    def test_bedrock_explicit_amrf_wins_over_model_settings(self, settings: UiPathBaseSettings):
        """An explicitly-passed additional_model_request_fields key wins over a
        colliding model_settings passthrough key."""
        UiPathBaseSettings._discovery_cache.clear()
        settings._discovery_cache[settings._discovery_cache_key()] = [
            {
                "modelName": "AWS - Bedrock",
                "vendor": "Bedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelFamily": "Anthropic",
                "modelDetails": {"customerModelName": "anthropic.claude-sonnet-4-5-20250929-v1:0"},
            }
        ]
        try:
            model = UiPathChatBedrockConverse(
                model="AWS - Bedrock",
                settings=settings,
                byo_connection_id="conn-x",
                base_model="anthropic.claude-sonnet-4-5-20250929-v1:0",
                provider="anthropic",
                additional_model_request_fields={"anthropic_beta": ["explicit"]},
                model_settings={"anthropic_beta": ["from-settings"]},
            )
        finally:
            UiPathBaseSettings._discovery_cache.clear()
        assert model.additional_model_request_fields["anthropic_beta"] == ["explicit"]

    def test_disabled_key_is_skipped(self, settings: UiPathBaseSettings):
        from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

        model = UiPathChatOpenAI(
            model="some-openai-model",
            settings=settings,
            model_details={},
            disabled_params={"temperature": None},
            model_settings={"temperature": 0.2},
        )
        assert model.temperature is None

    def test_string_value_coerced_to_field_type(self, settings: UiPathBaseSettings):
        """Values arrive as untyped JSON (UI forms, gateway data); a field key must go
        through pydantic assignment validation, not raw setattr, so '8192' becomes 8192
        instead of a string serialized into the request body."""
        from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

        model = UiPathChatOpenAI(
            model="some-openai-model",
            settings=settings,
            model_details={},
            model_settings={"max_tokens": "8192"},
        )
        assert model.max_tokens == 8192
        assert isinstance(model.max_tokens, int)

    def test_alias_key_sets_field_not_model_kwargs(self, settings: UiPathBaseSettings):
        """A key matching only a field alias ('timeout' -> request_timeout) must set the
        field instead of leaking into model_kwargs as an unknown completion parameter."""
        from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

        model = UiPathChatOpenAI(
            model="some-openai-model",
            settings=settings,
            model_details={},
            model_settings={"timeout": 42.0},
        )
        assert model.request_timeout == 42.0
        assert "timeout" not in (model.model_kwargs or {})

    def test_multiple_field_keys_do_not_pollute_model_kwargs(self, settings: UiPathBaseSettings):
        """Regression: applying settings via pydantic assignment validation re-ran
        LangChain's ``build_extra`` validator, which swept the cached uipath httpx
        clients out of ``__dict__`` into ``model_kwargs`` — from where they would
        be sent as completion parameters."""
        from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

        model = UiPathChatOpenAI(
            model="some-openai-model",
            settings=settings,
            model_details={},
            model_settings={"max_tokens": "8192", "timeout": 30},
        )
        assert model.max_tokens == 8192
        assert model.request_timeout == 30
        assert "uipath_sync_client" not in (model.model_kwargs or {})
        assert "uipath_async_client" not in (model.model_kwargs or {})

    def test_invalid_field_value_raises_at_construction(self, settings: UiPathBaseSettings):
        """A value pydantic can't coerce fails fast with a clear error instead of being
        stored raw and rejected by the provider at request time."""
        from pydantic import ValidationError
        from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

        with pytest.raises(ValidationError):
            UiPathChatOpenAI(
                model="some-openai-model",
                settings=settings,
                model_details={},
                model_settings={"max_tokens": "not-a-number"},
            )
