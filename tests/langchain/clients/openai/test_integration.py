"""LangChain integration tests for OpenAI provider clients."""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from tests.langchain.file_fixtures import PDF_FORMATS
from tests.langchain.integration_tests import UiPathChatModelIntegrationTests


@pytest.mark.asyncio
@pytest.mark.vcr
class TestOpenAIIntegrationChatModel(UiPathChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        _, model_kwargs = completions_config
        model_name = model_kwargs.get("model", "")
        test_name = request.node.originalname
        callspec = getattr(request.node, "callspec", None)
        fmt = callspec.params.get("fmt") if callspec else None

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

        # File-input matrix: PDF file blocks require the Responses API on Azure OpenAI.
        if (
            test_name in ("test_file_inputs", "test_file_inputs_async")
            and fmt in PDF_FORMATS
            and not model_kwargs.get("use_responses_api")
        ):
            pytest.skip("PDF inputs require the OpenAI Responses API")


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
