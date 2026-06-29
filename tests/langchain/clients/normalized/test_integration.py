"""LangChain integration tests for Normalized provider clients.

UiPathChat is tested with ALL models across providers (GPT, Gemini, Claude).
This includes thinking configurations from all providers.
"""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.integration_tests import EmbeddingsIntegrationTests
from uipath_langchain_client.clients.normalized.embeddings import UiPathEmbeddings

from tests.langchain.file_fixtures import IMAGE_FORMATS, PDF_FORMATS
from tests.langchain.integration_tests import UiPathChatModelIntegrationTests
from uipath.llm_client.settings import PlatformSettings


@pytest.mark.asyncio
@pytest.mark.vcr
class TestNormalizedIntegrationChatModel(UiPathChatModelIntegrationTests):
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
        is_claude = "claude" in model_name.lower()
        is_gemini = "gemini" in model_name.lower()
        is_gemini_3 = "gemini-3" in model_name.lower()
        is_vertex_claude = "@" in model_name and is_claude
        is_bedrock_claude = "anthropic." in model_name.lower()
        callspec = getattr(request.node, "callspec", None)
        fmt = callspec.params.get("fmt") if callspec else None

        # Claude via Vertex AI: streaming bugged (502 / empty content)
        if is_vertex_claude and test_name in [
            "test_stream",
            "test_astream",
            "test_stream_time",
            "test_usage_metadata_streaming",
        ]:
            pytest.skip(f"Skipping {test_name}: currently bugged on Vertex AI")

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

        # UiPathChat (normalized) + Gemini: tool operations fail
        if is_gemini and test_name in [
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
        ]:
            pytest.skip(f"Skipping {test_name}: not supported for Gemini on normalized API")

        # UiPathChat (normalized) + Gemini 3: agent_loop
        if is_gemini_3 and test_name in [
            "test_agent_loop",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported for Gemini 3 on normalized API")

        # UiPathChat (normalized) + Claude via Vertex: structured output / tool calling
        if is_vertex_claude and test_name in [
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
        ]:
            pytest.skip(f"Skipping {test_name}: not supported for Claude via Vertex on normalized")

        # UiPathChat (normalized) + Claude via Bedrock: image/pdf/parallel
        if is_bedrock_claude and test_name in [
            "test_image_inputs",
            "test_pdf_inputs",
            "test_parallel_and_sequential_tool_calling",
            "test_parallel_and_sequential_tool_calling_async",
        ]:
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

        # File-input matrix skips on the normalized API
        if test_name in ("test_file_inputs", "test_file_inputs_async"):
            if has_thinking:
                pytest.skip(
                    "Structured output forces tool_choice, incompatible with Claude thinking"
                )
            if "gpt" in model_name.lower() and fmt in PDF_FORMATS:
                pytest.skip("PDF inputs not supported for GPT on the normalized API")
            if is_gemini and fmt in PDF_FORMATS:
                pytest.skip("PDF inputs not supported for Gemini on the normalized API")
            if (is_vertex_claude or is_bedrock_claude) and fmt in (IMAGE_FORMATS | PDF_FORMATS):
                pytest.skip(
                    "Image/PDF content blocks not supported for Claude via "
                    "Vertex/Bedrock on the normalized API"
                )

    def test_parallel_and_sequential_tool_calling(self, model: BaseChatModel) -> None:
        """Test parallel tool calling - normalized API delegates to provider."""
        pytest.skip("Parallel tool calling is not supported for normalized API")

    async def test_parallel_and_sequential_tool_calling_async(self, model: BaseChatModel) -> None:
        """Test parallel tool calling async - normalized API delegates to provider."""
        pytest.skip("Parallel tool calling is not supported for normalized API")


@pytest.mark.asyncio
@pytest.mark.vcr
class TestNormalizedIntegrationEmbeddings(EmbeddingsIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, embeddings_config: tuple[type[Embeddings], dict[str, Any]]):
        self._embeddings_class, self._embeddings_kwargs = embeddings_config

    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        embeddings_config: tuple[type[Embeddings], dict[str, Any]],
    ) -> None:
        model_class, model_kwargs = embeddings_config
        client_settings = model_kwargs.get("client_settings")
        if model_class == UiPathEmbeddings and isinstance(client_settings, PlatformSettings):
            pytest.skip(
                "Normalized embeddings are not supported on UiPath Platform (AgentHub/Orchestrator)"
            )

    @property
    def embeddings_class(self) -> type[Embeddings]:
        return self._embeddings_class

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return self._embeddings_kwargs
