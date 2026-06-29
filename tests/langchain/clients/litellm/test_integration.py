"""LangChain integration tests for LiteLLM provider clients.

Tests UiPathChatLiteLLM across multiple providers using the langchain-tests framework.
"""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from tests.langchain.integration_tests import UiPathChatModelIntegrationTests


@pytest.mark.asyncio
@pytest.mark.vcr
class TestLiteLLMIntegrationChatModel(UiPathChatModelIntegrationTests):
    @property
    def supports_pdf_inputs(self) -> bool:
        return False

    @property
    def supports_pdf_tool_message(self) -> bool:
        return False

    @property
    def tool_choice_value(self) -> str:
        return "required"

    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        _, model_kwargs = completions_config
        model_name = model_kwargs.get("model", "")
        test_name = request.node.originalname
        is_claude = "claude" in model_name.lower()
        is_gemini = "gemini" in model_name.lower()
        is_bedrock = model_name.startswith("anthropic.")
        is_vertex_claude = "@" in model_name and is_claude

        # Streaming tests — Bedrock invoke streaming returns 500 from gateway
        # ("Unable to extract claim sub_type from token")
        if is_bedrock and test_name in [
            "test_stream",
            "test_astream",
            "test_stream_time",
            "test_usage_metadata_streaming",
        ]:
            pytest.skip(
                f"Skipping {test_name}: Bedrock streaming not supported via gateway S2S auth"
            )

        # Bedrock/Vertex Claude: structured output with tool_choice can be unreliable
        if (is_bedrock or is_vertex_claude) and test_name in [
            "test_structured_few_shot_examples",
            "test_structured_output",
            "test_structured_output_optional_param",
        ]:
            pytest.skip(
                f"Skipping {test_name}: structured output via tool_choice unreliable on {model_name}"
            )

        # Gemini: some tool calling tests incompatible
        if is_gemini and test_name in [
            "test_tool_message_histories_string_content",
            "test_tool_message_histories_list_content",
        ]:
            pytest.skip(
                f"Skipping {test_name}: tool message history format incompatible with Gemini via litellm"
            )

        # Bedrock streaming: tool_calling/tool_choice tests use stream() which
        # fails with gateway 500 on Bedrock invoke
        if is_bedrock and test_name in [
            "test_tool_calling",
            "test_tool_calling_async",
            "test_tool_calling_with_no_arguments",
        ]:
            pytest.skip(
                f"Skipping {test_name}: Bedrock streaming not supported via gateway S2S auth"
            )

        # tool_choice: gateway rejects function-name tool_choice (only none/auto/required)
        if test_name == "test_tool_choice":
            pytest.skip(
                f"Skipping {test_name}: gateway only supports none/auto/required tool_choice"
            )

        # stop param: litellm rejects 'stop' for newer OpenAI models
        if test_name == "test_stop_sequence" and "gpt-5" in model_name:
            pytest.skip(f"Skipping {test_name}: litellm rejects stop param for {model_name}")

        # structured_output: ChatLiteLLM.with_structured_output doesn't propagate
        # ls_structured_output_format metadata that langchain-tests expects
        if test_name in [
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(
                f"Skipping {test_name}: ChatLiteLLM.with_structured_output missing "
                "ls_structured_output_format (upstream langchain-litellm issue)"
            )

        # Parallel tool calling has historically not been exercised on the litellm
        # client (the previous class didn't override the test); leave the assertion
        # off until it has dedicated coverage.
        if test_name in (
            "test_parallel_and_sequential_tool_calling",
            "test_parallel_and_sequential_tool_calling_async",
        ):
            pytest.skip(f"Skipping {test_name}: not yet exercised on the LiteLLM client")

        # File-input matrix: reuses with_structured_output, which is unreliable on
        # ChatLiteLLM for the same reason `test_structured_output` is skipped.
        if test_name in ("test_file_inputs", "test_file_inputs_async"):
            pytest.skip(
                "Structured output via ChatLiteLLM is not currently exercised "
                "(upstream langchain-litellm issue)"
            )


@pytest.mark.asyncio
@pytest.mark.vcr
class TestLiteLLMIntegrationEmbeddings(EmbeddingsIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, embeddings_config: tuple[type[Embeddings], dict[str, Any]]):
        self._embeddings_class, self.embeddings_kwargs = embeddings_config

    @property
    def embeddings_class(self) -> type[Embeddings]:
        return self._embeddings_class

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return self.embeddings_kwargs
