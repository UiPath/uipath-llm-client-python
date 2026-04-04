"""LangChain integration tests for Google provider clients."""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests, EmbeddingsIntegrationTests

from uipath.llm_client.settings import PlatformSettings


@pytest.mark.asyncio
@pytest.mark.vcr
class TestGoogleIntegrationChatModel(ChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, completions_config: tuple[type[BaseChatModel], dict[str, Any]]):
        self._completions_class, self.completions_kwargs = completions_config

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_pdf_tool_message(self) -> bool:
        return True

    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        model_class, model_kwargs = completions_config
        model_name = model_kwargs.get("model", "")
        test_name = request.node.originalname
        is_gemini_3 = "gemini-3" in model_name.lower()

        # Useless framework tests
        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE", "test_unicode_tool_call_integration"]:
            pytest.skip(f"Skipping {test_name}: not relevant")

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

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self.completions_kwargs

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    def test_stream(self, model: BaseChatModel) -> None:
        chunks: list[AIMessageChunk] = []
        full: AIMessageChunk | None = None
        for chunk in model.stream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            chunks.append(chunk)
            full = chunk if full is None else full + chunk
        assert len(chunks) > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        text_blocks = [block for block in full.content_blocks if block["type"] == "text"]
        assert len(text_blocks) == 1

        last_chunk = chunks[-1]
        assert last_chunk.chunk_position == "last", (
            f"Final chunk must have chunk_position='last', got {last_chunk.chunk_position!r}"
        )

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    async def test_astream(self, model: BaseChatModel) -> None:
        chunks: list[AIMessageChunk] = []
        full: AIMessageChunk | None = None
        async for chunk in model.astream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            chunks.append(chunk)
            full = chunk if full is None else full + chunk
        assert len(chunks) > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        text_blocks = [block for block in full.content_blocks if block["type"] == "text"]
        assert len(text_blocks) == 1

        last_chunk = chunks[-1]
        assert last_chunk.chunk_position == "last", (
            f"Final chunk must have chunk_position='last', got {last_chunk.chunk_position!r}"
        )


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
