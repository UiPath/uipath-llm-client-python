from collections.abc import Awaitable, Callable
from typing import Self

from pydantic import Field, SecretStr, model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI

    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required to use UiPathChatOpenAI and UiPathAzureChatOpenAI. "
        "Install it with: uv add uipath-langchain-client[openai]"
    ) from e


class UiPathChatOpenAI(UiPathBaseLLMClient, ChatOpenAI):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="openai",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    openai_api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] = Field(
        alias="api_key", default=SecretStr("PLACEHOLDER")
    )

    @model_validator(mode="after")
    def setup_uipath_api_flavor_and_version(self) -> Self:
        self.api_config.api_version = "2025-03-01-preview"
        if self._use_responses_api({}):
            self.api_config.api_flavor = "responses"
        else:
            self.api_config.api_flavor = "chat-completions"
        return self

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.root_client = OpenAI(
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        )
        self.root_async_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        )
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        return self


class UiPathAzureChatOpenAI(UiPathBaseLLMClient, AzureChatOpenAI):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="openai",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    azure_endpoint: str | None = Field(default="PLACEHOLDER")
    openai_api_version: str | None = Field(default="PLACEHOLDER", alias="api_version")
    openai_api_key: SecretStr | None = Field(default=SecretStr("PLACEHOLDER"), alias="api_key")

    @model_validator(mode="after")
    def setup_uipath_api_flavor_and_version(self) -> Self:
        self.api_config.api_version = "2025-03-01-preview"
        if self._use_responses_api({}):
            self.api_config.api_flavor = "responses"
        else:
            self.api_config.api_flavor = "chat-completions"
        return self

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.root_client = AzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        )
        self.root_async_client = AsyncAzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        )
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        return self
