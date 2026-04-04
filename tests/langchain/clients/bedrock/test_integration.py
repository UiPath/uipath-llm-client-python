"""LangChain integration tests for Bedrock provider clients."""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)

from tests.langchain.utils import search_accommodation, search_attractions, search_flights


@pytest.mark.asyncio
@pytest.mark.vcr
class TestBedrockIntegrationChatModel(ChatModelIntegrationTests):
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
        test_name = request.node.originalname
        has_thinking = "thinking" in model_kwargs

        # Useless framework tests
        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE", "test_unicode_tool_call_integration"]:
            pytest.skip(f"Skipping {test_name}: not relevant")

        # Claude + thinking: tool_choice forces tool use, incompatible with thinking
        if has_thinking and test_name in [
            "test_structured_few_shot_examples",
            "test_tool_calling",
            "test_tool_calling_async",
            "test_tool_calling_with_no_arguments",
            "test_tool_choice",
            "test_tool_message_error_status",
            "test_bind_runnables_as_tools",
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
            "test_parallel_and_sequential_tool_calling",
            "test_parallel_and_sequential_tool_calling_async",
        ]:
            pytest.skip(
                f"Skipping {test_name}: thinking may not be enabled when tool_choice forces tool use"
            )

        # Claude + thinking: extended thinking requires specific conversation history
        if has_thinking and test_name in [
            "test_tool_message_histories_list_content",
            "test_tool_message_histories_string_content",
        ]:
            pytest.skip(
                f"Skipping {test_name}: extended thinking requires a specific conversation history"
            )

        # Claude via Bedrock: image URL sources not supported
        if model_class in (UiPathChatAnthropicBedrock, UiPathChatBedrock) and test_name in [
            "test_image_inputs",
        ]:
            pytest.skip(f"Skipping {test_name}: URL image sources not supported via gateway")

        # UiPathChatBedrock: PDF/file inputs not supported
        if model_class == UiPathChatBedrock and test_name in [
            "test_pdf_inputs",
        ]:
            pytest.skip(f"Skipping {test_name}: file blocks not supported on this client")

        # UiPathChatBedrockConverse: parallel tool calling not supported
        if model_class == UiPathChatBedrockConverse and test_name in [
            "test_parallel_and_sequential_tool_calling",
            "test_parallel_and_sequential_tool_calling_async",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported by Bedrock Converse API")

        # UiPathChatBedrockConverse: image/pdf serialization issues
        if model_class == UiPathChatBedrockConverse and test_name in [
            "test_image_inputs",
            "test_pdf_inputs",
            "test_image_tool_message",
            "test_pdf_tool_message",
        ]:
            pytest.skip(f"Skipping {test_name}: serialization not supported on Bedrock Converse")

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
        """Test parallel tool calling for Claude Bedrock models."""
        tools = [search_accommodation, search_flights, search_attractions]
        prompt = (
            "I want to plan a trip to Paris from New York. "
            "I need to find flights for March 15th, accommodation from March 15th to March 20th, and things to do there.",
            "Search for accomodations, flights and attractions in parallel. Don't repeat the same tool call.",
        )
        model_with_tools_parallel = model.bind_tools(
            tools,
            tool_choice={"type": "any", "disable_parallel_tool_use": False},  # type: ignore
        )
        model_with_tools_sequential = model.bind_tools(
            tools,
            tool_choice={"type": "any", "disable_parallel_tool_use": True},  # type: ignore
        )

        parallel_response = model_with_tools_parallel.invoke(prompt)
        sequential_response = model_with_tools_sequential.invoke(prompt)

        assert parallel_response.tool_calls is not None
        assert sequential_response.tool_calls is not None
        assert len(parallel_response.tool_calls) == len(tools)
        assert len(sequential_response.tool_calls) == 1

    async def test_parallel_and_sequential_tool_calling_async(self, model: BaseChatModel) -> None:
        """Test parallel and sequential tool calling async for Bedrock."""
        tools = [search_accommodation, search_flights, search_attractions]
        prompt = (
            "I want to plan a trip to Paris from New York. "
            "I need to find flights for March 15th, accommodation from March 15th to March 20th, and things to do there.",
            "Search for accomodations, flights and attractions in parallel. Don't repeat the same tool call.",
        )
        model_with_tools_parallel = model.bind_tools(
            tools,
            tool_choice={"type": "any", "disable_parallel_tool_use": False},  # type: ignore
        )
        model_with_tools_sequential = model.bind_tools(
            tools,
            tool_choice={"type": "any", "disable_parallel_tool_use": True},  # type: ignore
        )

        parallel_response = await model_with_tools_parallel.ainvoke(prompt)
        sequential_response = await model_with_tools_sequential.ainvoke(prompt)

        assert parallel_response.tool_calls is not None
        assert sequential_response.tool_calls is not None
        assert len(parallel_response.tool_calls) == len(tools)
        assert len(sequential_response.tool_calls) == 1
