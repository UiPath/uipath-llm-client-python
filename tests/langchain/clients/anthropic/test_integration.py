"""LangChain integration tests for UiPathChatAnthropic client.

Tests UiPathChatAnthropic with both vertexai and awsbedrock vendor_types.
"""

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from tests.langchain.file_fixtures import IMAGE_FORMATS, PDF_FORMATS
from tests.langchain.integration_tests import UiPathChatModelIntegrationTests


@pytest.mark.asyncio
@pytest.mark.vcr
class TestAnthropicIntegrationChatModel(UiPathChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        _, model_kwargs = completions_config
        model_name = model_kwargs.get("model", "")
        test_name = request.node.originalname
        has_thinking = "thinking" in model_kwargs
        is_vertex = model_kwargs.get("vendor_type") == "vertexai" or "@" in model_name
        callspec = getattr(request.node, "callspec", None)
        fmt = callspec.params.get("fmt") if callspec else None

        # Claude via Vertex AI: streaming bugged (502 / empty content)
        if is_vertex and test_name in [
            "test_stream",
            "test_astream",
            "test_stream_time",
            "test_usage_metadata_streaming",
        ]:
            pytest.skip(f"Skipping {test_name}: currently bugged on Vertex AI")

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

        # UiPathChatAnthropic: structured output / tool calling content_blocks issues
        if test_name in [
            "test_tool_calling",
            "test_tool_calling_async",
            "test_tool_calling_with_no_arguments",
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(f"Skipping {test_name}: content_blocks not currently supported")

        # Claude: image URL sources not supported via gateway
        if test_name in [
            "test_image_inputs",
        ]:
            pytest.skip(f"Skipping {test_name}: URL image sources not supported via gateway")

        # File-input matrix: structured output forces tool_choice (incompatible with
        # thinking); image/PDF blocks don't round-trip through the Anthropic gateway.
        if test_name in ("test_file_inputs", "test_file_inputs_async"):
            if has_thinking:
                pytest.skip(
                    "Structured output forces tool_choice, which is incompatible with thinking"
                )
            if fmt in (IMAGE_FORMATS | PDF_FORMATS):
                pytest.skip(
                    "Image/PDF content blocks are not supported via the Anthropic gateway path"
                )

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
