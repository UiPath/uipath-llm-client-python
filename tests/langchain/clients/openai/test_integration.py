"""LangChain integration tests for OpenAI provider clients."""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests, EmbeddingsIntegrationTests

from tests.langchain.utils import search_accommodation, search_attractions, search_flights


@pytest.mark.asyncio
@pytest.mark.vcr
class TestOpenAIIntegrationChatModel(ChatModelIntegrationTests):
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

        # Useless framework tests
        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE", "test_unicode_tool_call_integration"]:
            pytest.skip(f"Skipping {test_name}: not relevant")

        # GPT-5 / responses_api: stop_sequence not supported
        if ("gpt-5" in model_name.lower() or "use_responses_api" in model_kwargs) and test_name in [
            "test_stop_sequence",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported for GPT-5 / responses API")

        # GPT models: pdf/image tool messages not supported
        if "gpt" in model_name.lower() and test_name in [
            "test_pdf_inputs",
            "test_pdf_tool_message",
            "test_image_tool_message",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported for GPT models")

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

    def test_parallel_and_sequential_tool_calling(self, model: BaseChatModel) -> None:
        """Test parallel tool calling - model should call multiple tools at once."""
        tools = [search_accommodation, search_flights, search_attractions]
        prompt = (
            "I want to plan a trip to Paris from New York. "
            "I need to find flights for March 15th, accommodation from March 15th to March 20th, and things to do there.",
            "Search for accomodations, flights and attractions in parallel. Don't repeat the same tool call.",
        )
        model_with_tools_parallel = model.bind_tools(
            tools, tool_choice="any", parallel_tool_calls=True
        )
        model_with_tools_sequential = model.bind_tools(
            tools, tool_choice="any", parallel_tool_calls=False
        )

        parallel_response = model_with_tools_parallel.invoke(prompt)
        sequential_response = model_with_tools_sequential.invoke(prompt)

        assert parallel_response.tool_calls is not None
        assert sequential_response.tool_calls is not None
        assert len(parallel_response.tool_calls) == len(tools), (
            f"Expected multiple different tools to be called in parallel, got: {parallel_response.tool_calls}"
        )
        assert len(sequential_response.tool_calls) == 1, (
            f"Expected only one tool to be called in sequential mode, got: {sequential_response.tool_calls}"
        )

    async def test_parallel_and_sequential_tool_calling_async(self, model: BaseChatModel) -> None:
        """Test parallel and sequential tool calling async."""
        tools = [search_accommodation, search_flights, search_attractions]
        prompt = (
            "I want to plan a trip to Paris from New York. "
            "I need to find flights for March 15th, accommodation from March 15th to March 20th, and things to do there.",
            "Search for accomodations, flights and attractions in parallel. Don't repeat the same tool call.",
        )
        model_with_tools_parallel = model.bind_tools(
            tools, tool_choice="any", parallel_tool_calls=True
        )
        model_with_tools_sequential = model.bind_tools(
            tools, tool_choice="any", parallel_tool_calls=False
        )

        parallel_response = await model_with_tools_parallel.ainvoke(prompt)
        sequential_response = await model_with_tools_sequential.ainvoke(prompt)

        assert parallel_response.tool_calls is not None
        assert sequential_response.tool_calls is not None
        assert len(parallel_response.tool_calls) == len(tools), (
            f"Expected multiple different tools to be called in parallel, got: {parallel_response.tool_calls}"
        )
        assert len(sequential_response.tool_calls) == 1, (
            f"Expected only one tool to be called in sequential mode, got: {sequential_response.tool_calls}"
        )


@pytest.mark.asyncio
@pytest.mark.vcr
class TestOpenAIIntegrationEmbeddings(EmbeddingsIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, embeddings_config: tuple[type[Embeddings], dict[str, Any]]):
        self._embeddings_class, self._embeddings_kwargs = embeddings_config

    @property
    def embeddings_class(self) -> type[Embeddings]:
        return self._embeddings_class

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return self._embeddings_kwargs
