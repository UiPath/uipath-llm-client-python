"""LangChain unit tests for Bedrock provider clients."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_tests.unit_tests import ChatModelUnitTests, EmbeddingsUnitTests
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.bedrock.embeddings import UiPathBedrockEmbeddings

from uipath.llm_client.settings import UiPathBaseSettings
from uipath.llm_client.utils.model_family import (
    CLAUDE_OPUS_4_UNSUPPORTED_SAMPLING_PARAMS,
    is_claude_opus_4_or_above,
)

BEDROCK_CHAT_CLASSES = [UiPathChatAnthropicBedrock, UiPathChatBedrock, UiPathChatBedrockConverse]
BEDROCK_EMBEDDINGS_CLASSES = [UiPathBedrockEmbeddings]


class TestBedrockChatModel(ChatModelUnitTests):
    @pytest.fixture(autouse=True, params=BEDROCK_CHAT_CLASSES)
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


class TestSamplingParamStripping:
    """UiPathBaseChatModel must strip sampling params for models whose modelDetails
    declare ``shouldSkipTemperature: True`` — both at instantiation (ctor kwargs) and
    at invocation time (``invoke(..., temperature=...)``)."""

    @pytest.fixture()
    def opus4_client(self, client_settings: UiPathBaseSettings) -> UiPathChatAnthropicBedrock:
        # model_details bypasses the discovery network call at init.
        return UiPathChatAnthropicBedrock(
            model="anthropic.claude-opus-4-7",
            settings=client_settings,
            model_details={"shouldSkipTemperature": True},
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        )

    def test_is_claude_opus_4_or_above(self) -> None:
        # Name-based fallback for models not found in discovery.
        assert is_claude_opus_4_or_above("anthropic.claude-opus-4-7")
        assert is_claude_opus_4_or_above("claude-opus-4-7-20250514")
        assert not is_claude_opus_4_or_above("anthropic.claude-opus-4-5-20251101-v1:0")
        assert not is_claude_opus_4_or_above("anthropic.claude-opus-4-6-v1")
        assert not is_claude_opus_4_or_above("anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert not is_claude_opus_4_or_above("anthropic.claude-haiku-4-5-20251001-v1:0")

    def test_instance_fields_nulled_at_init(self, opus4_client: UiPathChatAnthropicBedrock) -> None:
        # Model validator on UiPathBaseChatModel nulled the sampling fields and
        # discarded them from __pydantic_fields_set__ so downstream payload builders
        # treat them as unset.
        for param in CLAUDE_OPUS_4_UNSUPPORTED_SAMPLING_PARAMS:
            if param in type(opus4_client).model_fields:
                assert getattr(opus4_client, param) is None
                assert param not in opus4_client.model_fields_set

    def test_invocation_kwargs_stripped(self, opus4_client: UiPathChatAnthropicBedrock) -> None:
        # llm.invoke("...", temperature=0.5) — the kwargs must not reach the SDK call.
        with patch.object(opus4_client, "_client") as mock_client:
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(type="text", text="hi")],
                stop_reason="end_turn",
                usage=MagicMock(input_tokens=10, output_tokens=5),
                model="anthropic.claude-opus-4-7",
                id="msg_123",
            )
            opus4_client.invoke(
                [HumanMessage(content="hi")],
                temperature=0.5,
                top_k=10,
                top_p=0.5,
            )
        call_kwargs = mock_client.messages.create.call_args.kwargs
        for param in CLAUDE_OPUS_4_UNSUPPORTED_SAMPLING_PARAMS:
            assert param not in call_kwargs, f"{param} must be stripped"

    def test_supported_model_keeps_params(self, client_settings: UiPathBaseSettings) -> None:
        haiku = UiPathChatAnthropicBedrock(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            settings=client_settings,
            model_details={"shouldSkipTemperature": False},
            temperature=0.5,
        )
        assert haiku.temperature == 0.5
        assert "temperature" in haiku.model_fields_set


class TestBedrockEmbeddings(EmbeddingsUnitTests):
    @pytest.fixture(autouse=True, params=BEDROCK_EMBEDDINGS_CLASSES)
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
