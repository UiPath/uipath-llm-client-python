"""LangChain integration tests for VertexAI provider client."""

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from tests.langchain.integration_tests import UiPathChatModelIntegrationTests


@pytest.mark.asyncio
@pytest.mark.vcr
class TestVertexAIIntegrationChatModel(UiPathChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        _, model_kwargs = completions_config
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

        # UiPathChatAnthropicVertex: ls_structured_output_format not implemented
        if test_name in [
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(f"Skipping {test_name}: ls_structured_output not supported on this client")

        # UiPathChatAnthropicVertex: content_blocks not populated correctly
        if test_name in [
            "test_tool_calling",
        ]:
            pytest.skip(f"Skipping {test_name}: content_blocks not supported on this client")

        # UiPathChatAnthropicVertex: system message must be at beginning
        if test_name in [
            "test_double_messages_conversation",
        ]:
            pytest.skip(
                f"Skipping {test_name}: system message must be at beginning of message list"
            )

        # UiPathChatAnthropicVertex: agent_loop fails with non_standard content tag
        if test_name in [
            "test_agent_loop",
        ]:
            pytest.skip(f"Skipping {test_name}: fails due to non_standard content tag")

        # Claude via Vertex AI: streaming bugged (502 / empty content)
        if test_name in [
            "test_stream",
            "test_astream",
            "test_stream_time",
            "test_usage_metadata_streaming",
        ]:
            pytest.skip(f"Skipping {test_name}: currently bugged on Vertex AI")

        # Claude via Vertex: image URL sources not supported
        if test_name in [
            "test_image_inputs",
        ]:
            pytest.skip(f"Skipping {test_name}: URL image sources not supported via gateway")

        # Claude via Vertex: PDF/file inputs not supported
        if test_name in [
            "test_pdf_inputs",
        ]:
            pytest.skip(f"Skipping {test_name}: file blocks not supported on this client")

        # Image/pdf tool message not supported
        if test_name in [
            "test_image_tool_message",
            "test_pdf_tool_message",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported on this client")

        # File-input matrix: structured output not supported on this client (matches
        # the test_structured_output skip above).
        if test_name in ("test_file_inputs", "test_file_inputs_async"):
            pytest.skip("Structured output is not currently supported on this client")

    def _bind_parallel_and_sequential(
        self, model: BaseChatModel, tools: Iterable[Any]
    ) -> tuple[Runnable, Runnable]:
        tools_list = list(tools)
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
