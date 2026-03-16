from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests, EmbeddingsIntegrationTests
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.vertexai.chat_models import UiPathChatAnthropicVertex

from tests.langchain.utils import search_accommodation, search_attractions, search_flights


@pytest.mark.asyncio
@pytest.mark.vcr
class TestIntegrationChatModel(ChatModelIntegrationTests):
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
        has_thinking = "thinking" in model_kwargs
        is_claude = "claude" in model_name.lower()
        is_gemini = "gemini" in model_name.lower()
        is_gemini_3 = "gemini-3" in model_name.lower()
        is_vertex_claude = "@" in model_name and is_claude

        # Useless framework tests
        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE", "test_unicode_tool_call_integration"]:
            pytest.skip(f"Skipping {test_name}: not relevant")

        # Claude + thinking: tool_choice forces tool use, incompatible with thinking
        if (
            is_claude
            and has_thinking
            and test_name
            in [
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
            ]
        ):
            pytest.skip(
                f"Skipping {test_name}: thinking may not be enabled when tool_choice forces tool use"
            )

        # Claude + thinking: extended thinking requires specific conversation history
        if (
            is_claude
            and has_thinking
            and test_name
            in [
                "test_tool_message_histories_list_content",
                "test_tool_message_histories_string_content",
            ]
        ):
            pytest.skip(
                f"Skipping {test_name}: extended thinking requires a specific conversation history"
            )

        # UiPathChatAnthropicVertex: ls_structured_output_format not implemented
        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(f"Skipping {test_name}: ls_structured_output not supported on this client")

        # UiPathChatAnthropicVertex: content_blocks not populated correctly with output_version v1
        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_tool_calling",
        ]:
            pytest.skip(f"Skipping {test_name}: content_blocks not supported on this client")

        # UiPathChatAnthropicVertex: system message must be at beginning
        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_double_messages_conversation",
        ]:
            pytest.skip(
                f"Skipping {test_name}: system message must be at beginning of message list"
            )

        # UiPathChatAnthropicVertex: agent_loop fails with non_standard content tag
        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_agent_loop",
        ]:
            pytest.skip(f"Skipping {test_name}: fails due to non_standard content tag")

        # UiPathChatAnthropic: structured output / tool calling content_blocks issues
        if model_class == UiPathChatAnthropic and test_name in [
            "test_tool_calling",
            "test_tool_calling_async",
            "test_tool_calling_with_no_arguments",
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(f"Skipping {test_name}: content_blocks not currently supported")

        # Claude via Vertex AI: streaming bugged (502 / empty content)
        if is_vertex_claude and test_name in [
            "test_stream",
            "test_astream",
            "test_stream_time",
            "test_usage_metadata_streaming",
        ]:
            pytest.skip(f"Skipping {test_name}: currently bugged on Vertex AI")

        # Claude via Vertex/Bedrock (Anthropic SDK): image URL sources not supported
        if (
            is_claude
            and model_class
            in [
                UiPathChatAnthropicVertex,
                UiPathChatAnthropic,
                UiPathChatAnthropicBedrock,
                UiPathChatBedrock,
            ]
            and test_name
            in [
                "test_image_inputs",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: URL image sources not supported via gateway")

        # Claude via Vertex/Bedrock: PDF/file inputs not supported
        if (
            is_claude
            and model_class
            in [
                UiPathChatAnthropicVertex,
                UiPathChatBedrock,
            ]
            and test_name
            in [
                "test_pdf_inputs",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: file blocks not supported on this client")

        # UiPathChatAnthropicVertex: image/pdf tool message
        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_image_tool_message",
            "test_pdf_tool_message",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported on this client")

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

        # Gemini GoogleGenerativeAI: tool_message_histories / tool_message_error_status
        if (
            model_class == UiPathChatGoogleGenerativeAI
            and is_gemini_3
            and test_name
            in [
                "test_tool_message_histories_string_content",
                "test_tool_message_histories_list_content",
                "test_tool_message_error_status",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: not supported for Gemini 3 models")

        # Gemini GoogleGenerativeAI: thought signature invalid in multi-turn with thinking
        if (
            model_class == UiPathChatGoogleGenerativeAI
            and is_gemini
            and test_name
            in [
                "test_image_tool_message",
                "test_pdf_tool_message",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: thought signature invalid in multi-turn")

        # Gemini GoogleGenerativeAI: parallel tool calling
        if (
            model_class == UiPathChatGoogleGenerativeAI
            and is_gemini
            and test_name
            in [
                "test_parallel_and_sequential_tool_calling",
                "test_parallel_and_sequential_tool_calling_async",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: not supported for Gemini models")

        # UiPathChat (normalized) + Gemini: tool operations fail
        if (
            model_class == UiPathChat
            and is_gemini
            and test_name
            in [
                "test_tool_calling",
                "test_tool_calling_async",
                "test_tool_choice",
                "test_structured_few_shot_examples",
                "test_tool_message_error_status",
                "test_tool_message_histories_list_content",
                "test_tool_message_histories_string_content",
                "test_pdf_inputs",
                "test_image_tool_message",
                "test_pdf_tool_message",
                "test_parallel_and_sequential_tool_calling",
                "test_parallel_and_sequential_tool_calling_async",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: not supported for Gemini on normalized API")

        # UiPathChat (normalized) + Gemini 3: agent_loop
        if (
            model_class == UiPathChat
            and is_gemini_3
            and test_name
            in [
                "test_agent_loop",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: not supported for Gemini 3 on normalized API")

        # UiPathChat (normalized) + Claude via Vertex: structured output / tool calling
        if (
            model_class == UiPathChat
            and is_vertex_claude
            and test_name
            in [
                "test_tool_calling",
                "test_tool_calling_async",
                "test_tool_calling_with_no_arguments",
                "test_structured_output",
                "test_structured_output_async",
                "test_structured_output_pydantic_2_v1",
                "test_pdf_inputs",
                "test_image_inputs",
                "test_parallel_and_sequential_tool_calling",
                "test_parallel_and_sequential_tool_calling_async",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: not supported for Claude via Vertex on normalized")

        # UiPathChat (normalized) + Claude via Bedrock: image/pdf/parallel
        if (
            model_class == UiPathChat
            and "anthropic." in model_name.lower()
            and test_name
            in [
                "test_image_inputs",
                "test_pdf_inputs",
                "test_parallel_and_sequential_tool_calling",
                "test_parallel_and_sequential_tool_calling_async",
            ]
        ):
            pytest.skip(f"Skipping {test_name}: not supported for Claude via Bedrock on normalized")

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
        model_name = getattr(model, "model_name", "") or getattr(model, "model", "")
        if "gpt" in model_name.lower():
            model_with_tools_parallel = model.bind_tools(
                tools, tool_choice="any", parallel_tool_calls=True
            )
            model_with_tools_sequential = model.bind_tools(
                tools, tool_choice="any", parallel_tool_calls=False
            )
        elif "claude" in model_name.lower():
            model_with_tools_parallel = model.bind_tools(
                tools,
                tool_choice={"type": "any", "disable_parallel_tool_use": False},  # type: ignore
            )
            model_with_tools_sequential = model.bind_tools(
                tools,
                tool_choice={"type": "any", "disable_parallel_tool_use": True},  # type: ignore
            )
        elif "gemini" in model_name.lower():
            model_with_tools_parallel = model.bind_tools(
                tools,
                tool_config={
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tool.name for tool in tools],
                    }
                },
            )
            model_with_tools_sequential = model.bind_tools(
                tools,
                tool_config={
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tools[0].name],
                    }
                },
            )
        else:
            pytest.skip("Parallel tool calling is not supported for this model")

        parallel_response = model_with_tools_parallel.invoke(prompt)
        sequential_response = model_with_tools_sequential.invoke(prompt)

        # Verify tool calls were made
        assert parallel_response.tool_calls is not None
        assert sequential_response.tool_calls is not None
        assert len(parallel_response.tool_calls) == len(tools), (
            f"Expected multiple different tools to be called in parallel, got: {parallel_response.tool_calls}"
        )
        assert len(sequential_response.tool_calls) == 1, (
            f"Expected only one tool to be called in sequential mode, got: {sequential_response.tool_calls}"
        )

    async def test_parallel_and_sequential_tool_calling_async(self, model: BaseChatModel) -> None:
        """Test parallel and sequential tool calling async - compare both modes."""
        tools = [search_accommodation, search_flights, search_attractions]
        prompt = (
            "I want to plan a trip to Paris from New York. "
            "I need to find flights for March 15th, accommodation from March 15th to March 20th, and things to do there.",
            "Search for accomodations, flights and attractions in parallel. Don't repeat the same tool call.",
        )
        model_name = getattr(model, "model_name", "") or getattr(model, "model", "")
        if "gpt" in model_name.lower():
            model_with_tools_parallel = model.bind_tools(
                tools, tool_choice="any", parallel_tool_calls=True
            )
            model_with_tools_sequential = model.bind_tools(
                tools, tool_choice="any", parallel_tool_calls=False
            )
        elif "claude" in model_name.lower():
            model_with_tools_parallel = model.bind_tools(
                tools,
                tool_choice={"type": "any", "disable_parallel_tool_use": False},  # type: ignore
            )
            model_with_tools_sequential = model.bind_tools(
                tools,
                tool_choice={"type": "any", "disable_parallel_tool_use": True},  # type: ignore
            )
        elif "gemini" in model_name.lower():
            model_with_tools_parallel = model.bind_tools(
                tools,
                tool_config={
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tool.name for tool in tools],
                    }
                },
            )
            model_with_tools_sequential = model.bind_tools(
                tools,
                tool_config={
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tools[0].name],
                    }
                },
            )
        else:
            pytest.skip("Parallel tool calling is not supported for this model")

        parallel_response = await model_with_tools_parallel.ainvoke(prompt)
        sequential_response = await model_with_tools_sequential.ainvoke(prompt)

        # Verify tool calls were made
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
class TestIntegrationEmbeddings(EmbeddingsIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, embeddings_config: tuple[type[Embeddings], dict[str, Any]]):
        self._embeddings_class, self._embeddings_kwargs = embeddings_config

    @property
    def embeddings_class(self) -> type[Embeddings]:
        return self._embeddings_class

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return self._embeddings_kwargs
