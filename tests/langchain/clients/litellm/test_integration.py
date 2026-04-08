"""LangChain integration tests for LiteLLM provider clients.

Tests UiPathChatLiteLLM across multiple providers using the langchain-tests framework.
"""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests, EmbeddingsIntegrationTests


@pytest.mark.asyncio
@pytest.mark.vcr
class TestLiteLLMIntegrationChatModel(ChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, completions_config: tuple[type[BaseChatModel], dict[str, Any]]):
        self._completions_class, self.completions_kwargs = completions_config

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self.completions_kwargs

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
        return False

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

        # Skip framework-internal tests
        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE", "test_unicode_tool_call_integration"]:
            pytest.skip(f"Skipping {test_name}: not relevant")

        # Streaming tests — litellm streaming through UiPath gateway can be flaky
        if test_name in [
            "test_stream",
            "test_astream",
            "test_stream_time",
            "test_usage_metadata_streaming",
        ]:
            pytest.skip(f"Skipping {test_name}: streaming not stable via litellm + UiPath gateway")

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

    @property
    def tool_choice_value(self) -> str:
        return "required"


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
