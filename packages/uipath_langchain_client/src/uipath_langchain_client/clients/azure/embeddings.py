from typing import Self

from pydantic import model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel

    from azure.ai.inference import EmbeddingsClient
    from azure.ai.inference.aio import EmbeddingsClient as EmbeddingsClientAsync
except ImportError as e:
    raise ImportError(
        "The 'azure' extra is required to use UiPathAzureAIEmbeddingsModel. "
        "Install it with: uv add uipath-langchain-client[azure]"
    ) from e


class UiPathAzureAIEmbeddingsModel(UiPathBaseLLMClient, AzureAIEmbeddingsModel):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="azure",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    endpoint: str | None = "PLACEHOLDER"
    credentials: str | None = "PLACEHOLDER"

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        # TODO: finish implementation once we have a proper model in UiPath API
        self._client = EmbeddingsClient(
            endpoint="PLACEHOLDER",
            credentials="PLACEHOLDER",
            model=self.model_name,
            **self.client_kwargs,
        )
        self._async_client = EmbeddingsClientAsync(
            endpoint="PLACEHOLDER",
            credentials="PLACEHOLDER",
            model=self.model_name,
            **self.client_kwargs,
        )
        return self
