from typing import Self

from httpx import Request
from pydantic import Field, model_validator

from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.clients.openai.utils import fix_url_and_api_flavor_header
from uipath_langchain_client.settings import (
    ApiFlavor,
    ApiType,
    RoutingMode,
    UiPathAPIConfig,
    VendorType,
)

try:
    from azure.core.credentials import AzureKeyCredential, TokenCredential
    from azure.core.credentials_async import AsyncTokenCredential
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'azure' extra is required to use UiPathAzureAIChatCompletionsModel. "
        "Install it with: uv add uipath-langchain-client[azure]"
    ) from e


class UiPathAzureAIChatCompletionsModel(UiPathBaseChatModel, AzureAIOpenAIApiChatModel):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.AZURE,
        freeze_base_url=False,
    )
    api_flavor: ApiFlavor | str | None = None

    # Override fields to avoid env var lookup / validation errors at instantiation
    endpoint: str | None = Field(default="PLACEHOLDER")
    credential: str | AzureKeyCredential | TokenCredential | AsyncTokenCredential | None = Field(
        default="PLACEHOLDER"
    )

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        if self.api_flavor is not None:
            self.api_config.api_flavor = self.api_flavor
        base_url = str(self.uipath_sync_client.base_url).rstrip("/")
        locked_flavor = str(self.api_config.api_flavor) if self.api_config.api_flavor else None

        def on_request(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        async def on_request_async(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        self.uipath_sync_client.event_hooks["request"].append(on_request)
        self.uipath_async_client.event_hooks["request"].append(on_request_async)

        self.root_client = OpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        )
        self.root_async_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        )
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        return self
