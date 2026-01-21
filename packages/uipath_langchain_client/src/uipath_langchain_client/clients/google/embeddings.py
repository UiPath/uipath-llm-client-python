from typing import Self

from pydantic import Field, SecretStr, model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

    from google.genai.client import Client
    from google.genai.types import HttpOptions
except ImportError as e:
    raise ImportError(
        "The 'google' extra is required to use UiPathGoogleGenerativeAIEmbeddings. "
        "Install it with: uv add uipath-langchain-client[google]"
    ) from e


class UiPathGoogleGenerativeAIEmbeddings(UiPathBaseLLMClient, GoogleGenerativeAIEmbeddings):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="vertexai",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model: str = Field(default="", alias="model_name")
    google_api_key: SecretStr | None = Field(default=SecretStr("PLACEHOLDER"))

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = Client(
            vertexai=True,
            api_key="PLACEHOLDER",
            http_options=HttpOptions(
                timeout=None,  # handled by the UiPath client
                retry_options=None,  # handled by the UiPath client
                base_url=str(self.uipath_sync_client.base_url),
                headers=dict(self.uipath_sync_client.headers),
                httpx_client=self.uipath_sync_client,
                httpx_async_client=self.uipath_async_client,
            ),
        )
        return self
