"""LangChain integration tests for Bedrock provider clients."""

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)

from tests.langchain.integration_tests import UiPathChatModelIntegrationTests


@pytest.mark.asyncio
@pytest.mark.vcr
class TestBedrockIntegrationChatModel(UiPathChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        model_class, model_kwargs = completions_config
        test_name = request.node.originalname
        has_thinking = "thinking" in model_kwargs

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

        # File-input matrix: structured output uses tool_choice; thinking is incompatible.
        if test_name in ("test_file_inputs", "test_file_inputs_async") and has_thinking:
            pytest.skip("Structured output forces tool_choice, which is incompatible with thinking")

    def _bind_parallel_and_sequential(
        self, model: BaseChatModel, tools: Iterable[Any]
    ) -> tuple[Runnable, Runnable]:
        tools_list = list(tools)
        if isinstance(model, UiPathChatAnthropicBedrock):
            # UiPathChatAnthropicBedrock uses parallel_tool_calls as a top-level param.
            return (
                model.bind_tools(tools_list, tool_choice="any", parallel_tool_calls=True),
                model.bind_tools(tools_list, tool_choice="any", parallel_tool_calls=False),
            )
        return (
            model.bind_tools(
                tools_list,
                tool_choice={"type": "any", "disable_parallel_tool_use": False},  # type: ignore
            ),
            model.bind_tools(
                tools_list,
                tool_choice={"type": "any", "disable_parallel_tool_use": True},  # type: ignore
            ),
        )
