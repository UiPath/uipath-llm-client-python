from typing import Self

from pydantic import Field, model_validator

from uipath_langchain_client.base_client import UiPathBaseEmbeddings
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from azure.core.credentials import AzureKeyCredential, TokenCredential
    from azure.core.credentials_async import AsyncTokenCredential
    from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'azure' extra is required to use UiPathAzureAIEmbeddingsModel. "
        "Install it with: uv add uipath-langchain-client[azure]"
    ) from e


class UiPathAzureAIEmbeddingsModel(UiPathBaseEmbeddings, AzureAIOpenAIApiEmbeddingsModel):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="azure",
        freeze_base_url=True,
    )

    # Override fields to avoid env var lookup / validation errors at instantiation
    model: str = Field(default="", alias="model_name")
    endpoint: str | None = Field(default="PLACEHOLDER")
    credential: str | AzureKeyCredential | TokenCredential | AsyncTokenCredential | None = Field(default="PLACEHOLDER")

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = OpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        ).embeddings
        self.async_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        ).embeddings
        return self
