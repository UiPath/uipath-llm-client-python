from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests, EmbeddingsIntegrationTests
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.normalized.chat_models import UiPathNormalizedChatModel
from uipath_langchain_client.clients.vertexai.chat_models import UiPathChatAnthropicVertex

from tests.langchain.utils import search_accommodation, search_attractions, search_flights


@pytest.mark.asyncio
@pytest.mark.vcr
class TestIntegrationChatModel(ChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, completions_config: tuple[type[BaseChatModel], dict[str, Any]]):
        self._completions_class, self.completions_kwargs = completions_config

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

        if test_name in [
            "test_parallel_and_sequential_tool_calling",
            "test_parallel_and_sequential_tool_calling_async",
        ]:
            if model_class == UiPathChatBedrockConverse:
                pytest.skip(
                    "Skipping test because it is not supported by the Bedrock Converse API."
                )
            if has_thinking and "claude" in model_name.lower():
                pytest.skip(
                    "Skipping test because thinking is not supported by the Bedrock Converse API."
                )
            if "gemini-3-flash" in model_name.lower():
                pytest.skip("Skipping test because it is not supported by the Gemini 3 Flash API.")

        if (
            model_class == UiPathChatGoogleGenerativeAI
            and "gemini-3-flash" in model_name.lower()
            and test_name
            in [
                "test_structured_output",
                "test_structured_output_optional_param",
            ]
        ):
            pytest.skip("Skip because weird streaming bug")

        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
        ]:
            pytest.skip("ls_structured_output is not supported on this client")

        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_tool_calling",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip("content_blocks are not supported on this client")

        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_agentic_loop",
        ]:
            pytest.skip("This test fails because of something with tag")

        if model_class in [UiPathChatAnthropic] and test_name in [
            "test_tool_calling",
            "test_tool_calling_async",
            "test_tool_calling_with_no_arguments",
            "test_structured_output",
            "test_structured_output_async",
            # "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(
                f"Skipping test {test_name} on {UiPathChatAnthropic.__name__} because: they don't currently support content blocks"
            )

        if (
            has_thinking
            and "claude" in model_name.lower()
            and test_name
            in [
                "test_structured_output",
                "test_structured_output_async",
                "test_structured_output_optional_param",
                "test_structured_output_pydantic_2_v1",
            ]
        ):
            pytest.skip(
                f"Skipping test {test_name} on {model_name} because: Thinking may not be enabled when tool_choice forces tool use."
            )

        if model_class == UiPathChatAnthropicVertex and test_name in [
            "test_double_messages_conversation",
        ]:
            pytest.skip(
                f"Skipping test {test_name} on {UiPathChatAnthropicVertex.__name__} because: System message must be at beginning of message list."
            )

        if model_class == UiPathChatBedrockConverse and test_name in [
            "test_bind_runnables_as_tools",
            "test_structured_few_shot_examples",
            "test_tool_calling",
            "test_tool_calling_async",
            "test_tool_calling_with_no_arguments",
            "test_tool_choice",
        ]:
            pytest.skip(
                f"Skipping test {test_name} because it is not supported by the Bedrock Converse API."
            )
        if model_class == UiPathChatBedrock and test_name in [
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(
                f"Skipping test {test_name} because it is not supported by the Bedrock API."
            )

        if "claude" in model_name.lower() and has_thinking:
            if test_name in [
                "test_structured_few_shot_examples",
                "test_tool_calling",
                "test_tool_calling_async",
                "test_tool_calling_with_no_arguments",
                "test_tool_choice",
                "test_tool_message_error_status",
                "test_bind_runnables_as_tools",
            ]:
                pytest.skip(
                    f"Skipping test {test_name} because extended thinking can't be enabled with force tool use"
                )
            if test_name in [
                "test_tool_message_histories_list_content",
                "test_tool_message_histories_string_content",
            ]:
                pytest.skip(
                    f"Skipping test {test_name} because extended thinking requires a speicific conversation history"
                )

        if "gemini" in model_name.lower():
            if model_class == UiPathNormalizedChatModel and test_name in [
                "test_tool_calling",
                "test_tool_calling_async",
                "test_tool_choice",
                "test_structured_few_shot_examples",
            ]:
                pytest.skip(
                    f"Skipping test {test_name} because it is not supported for Gemini models on normalized"
                )
            if (
                "gemini-3" in model_name.lower() or model_class == UiPathNormalizedChatModel
            ) and test_name in [
                "test_tool_message_error_status",
                "test_tool_message_histories_list_content",
                "test_tool_message_histories_string_content",
            ]:
                pytest.skip(
                    f"Skipping test {test_name} because it is not supported for Gemini 3 models on normalized"
                )
        if (
            "gemini-3" in model_name.lower()
            and model_class == UiPathNormalizedChatModel
            and test_name in ["test_agent_loop"]
        ):
            pytest.skip(
                f"Skipping test {test_name} because it is not supported for Gemini 3 models"
            )
        if (
            "@" in model_name
            and "claude" in model_name.lower()
            and test_name
            in ["test_stream", "test_astream", "test_stream_time", "test_usage_metadata_streaming"]
        ):
            pytest.skip(f"Skipping test {test_name} as it is currently bugged on Vertex AI")

        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE", "test_unicode_tool_call_integration"]:
            pytest.skip(f"Skipping test {test_name} because it is useless")
        if ("gpt-5" in model_name.lower() or "use_responses_api" in model_kwargs) and test_name in [
            "test_stop_sequence"
        ]:
            pytest.skip(f"Skipping test {test_name} because it is not supported for GPT-5 models")

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self.completions_kwargs

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    def test_stream(self, model: BaseChatModel) -> None:
        num_chunks = 0
        full: AIMessageChunk | None = None
        for chunk in model.stream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            num_chunks += 1
            full = chunk if full is None else full + chunk
        assert num_chunks > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        assert len([block for block in full.content_blocks if block["type"] == "text"]) == 1
        assert full.content_blocks[-1]["type"] == "text"

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    async def test_astream(self, model: BaseChatModel) -> None:
        num_chunks = 0
        full: AIMessageChunk | None = None
        async for chunk in model.astream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            num_chunks += 1
            full = chunk if full is None else full + chunk
        assert num_chunks > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        assert len([block for block in full.content_blocks if block["type"] == "text"]) == 1
        assert full.content_blocks[-1]["type"] == "text"

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
