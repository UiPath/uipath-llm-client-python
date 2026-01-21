from collections.abc import Awaitable, Callable
from typing import Self

from pydantic import Field, SecretStr, model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings

    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required to use UiPathOpenAIEmbeddings and UiPathAzureOpenAIEmbeddings. "
        "Install it with: uv add uipath-langchain-client[openai]"
    ) from e


class UiPathOpenAIEmbeddings(UiPathBaseLLMClient, OpenAIEmbeddings):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="openai",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model: str = Field(default="", alias="model_name")
    openai_api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] = Field(
        alias="api_key", default=SecretStr("PLACEHOLDER")
    )

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = OpenAI(
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        ).embeddings
        self.async_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        ).embeddings
        return self


class UiPathAzureOpenAIEmbeddings(UiPathBaseLLMClient, AzureOpenAIEmbeddings):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="openai",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model: str = Field(default="", alias="model_name")
    azure_endpoint: str | None = Field(default="PLACEHOLDER")
    openai_api_version: str | None = Field(default="PLACEHOLDER", alias="api_version")
    openai_api_key: SecretStr | None = Field(default=SecretStr("PLACEHOLDER"), alias="api_key")

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = AzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        ).embeddings
        self.async_client = AsyncAzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        ).embeddings
        return self
