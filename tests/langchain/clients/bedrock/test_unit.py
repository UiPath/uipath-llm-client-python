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


class TestClaudeOpus4SamplingParamFiltering:
    """UiPathChatAnthropicBedrock must strip temperature/top_k/top_p for claude-opus-4+ models."""

    @pytest.fixture()
    def opus4_client(self, client_settings: UiPathBaseSettings) -> UiPathChatAnthropicBedrock:
        return UiPathChatAnthropicBedrock(
            model="anthropic.claude-opus-4-7",
            settings=client_settings,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        )

    def test_is_claude_opus_4_or_above(self) -> None:
        # Matched — sampling params stripped
        assert is_claude_opus_4_or_above("anthropic.claude-opus-4-7")
        assert is_claude_opus_4_or_above("claude-opus-4-5-20250514")
        assert is_claude_opus_4_or_above("anthropic.claude-opus-4-5-20251101-v1:0")
        assert is_claude_opus_4_or_above("anthropic.claude-opus-4-6-v1")
        # Not matched — sampling params pass through unchanged
        assert not is_claude_opus_4_or_above("anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert not is_claude_opus_4_or_above("anthropic.claude-haiku-4-5-20251001-v1:0")

    def test_unsupported_params_stripped_from_payload(
        self, opus4_client: UiPathChatAnthropicBedrock
    ) -> None:
        with patch.object(opus4_client, "_client") as mock_client:
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(type="text", text="hi")],
                stop_reason="end_turn",
                usage=MagicMock(input_tokens=10, output_tokens=5),
                model="anthropic.claude-opus-4-7",
                id="msg_123",
            )
            opus4_client.invoke([HumanMessage(content="hi")])
        call_kwargs = mock_client.messages.create.call_args.kwargs
        for param in CLAUDE_OPUS_4_UNSUPPORTED_SAMPLING_PARAMS:
            assert param not in call_kwargs, f"{param} must be stripped for claude-opus-4"

    def test_sampling_params_kept_for_other_models(
        self, client_settings: UiPathBaseSettings
    ) -> None:
        haiku = UiPathChatAnthropicBedrock(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            settings=client_settings,
            temperature=0.5,
        )
        with patch.object(haiku, "_client") as mock_client:
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(type="text", text="hi")],
                stop_reason="end_turn",
                usage=MagicMock(input_tokens=10, output_tokens=5),
                model="anthropic.claude-haiku-4-5-20251001-v1:0",
                id="msg_123",
            )
            haiku.invoke([HumanMessage(content="hi")])
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5, "temperature must be kept for haiku"


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
