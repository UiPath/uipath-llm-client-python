"""LangChain integration tests for Google provider clients."""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from tests.langchain.integration_tests import UiPathChatModelIntegrationTests
from uipath.llm_client.settings import PlatformSettings


@pytest.mark.asyncio
@pytest.mark.vcr
class TestGoogleIntegrationChatModel(UiPathChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        _, model_kwargs = completions_config
        model_name = model_kwargs.get("model", "")
        test_name = request.node.originalname
        is_gemini_3 = "gemini-3" in model_name.lower()

        # Gemini GoogleGenerativeAI: tool_message_histories / tool_message_error_status
        if is_gemini_3 and test_name in [
            "test_tool_message_histories_string_content",
            "test_tool_message_histories_list_content",
            "test_tool_message_error_status",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported for Gemini 3 models")

        # Gemini GoogleGenerativeAI: thought signature invalid in multi-turn with thinking
        if test_name in [
            "test_image_tool_message",
            "test_pdf_tool_message",
        ]:
            pytest.skip(f"Skipping {test_name}: thought signature invalid in multi-turn")

        # Gemini GoogleGenerativeAI: parallel tool calling
        if test_name in [
            "test_parallel_and_sequential_tool_calling",
            "test_parallel_and_sequential_tool_calling_async",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported for Gemini models")


@pytest.mark.asyncio
@pytest.mark.vcr
class TestGoogleIntegrationEmbeddings(EmbeddingsIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, embeddings_config: tuple[type[Embeddings], dict[str, Any]]):
        self._embeddings_class, self._embeddings_kwargs = embeddings_config

    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        embeddings_config: tuple[type[Embeddings], dict[str, Any]],
    ) -> None:
        _, model_kwargs = embeddings_config
        client_settings = model_kwargs.get("client_settings")

        if isinstance(client_settings, PlatformSettings):
            pytest.skip("Platform embeddings endpoint only supports OpenAI-compatible models")

    @property
    def embeddings_class(self) -> type[Embeddings]:
        return self._embeddings_class

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return self._embeddings_kwargs
