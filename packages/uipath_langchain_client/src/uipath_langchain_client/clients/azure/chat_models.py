from typing import Self

from httpx import URL, Request
from pydantic import Field, model_validator

from uipath_langchain_client.base_client import UiPathBaseChatModel
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

    # Override fields to avoid env var lookup / validation errors at instantiation
    endpoint: str | None = Field(default="PLACEHOLDER")
    credential: str | AzureKeyCredential | TokenCredential | AsyncTokenCredential | None = Field(
        default="PLACEHOLDER"
    )

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        base_url = str(self.uipath_sync_client.base_url).rstrip("/")

        def fix_url_and_api_flavor_header(request: Request):
            url_suffix = str(request.url).split(base_url)[-1]
            if "responses" in url_suffix:
                request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.RESPONSES.value
            else:
                request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.CHAT_COMPLETIONS.value
            request.url = URL(base_url)

        async def fix_url_and_api_flavor_header_async(request: Request):
            url_suffix = str(request.url).split(base_url)[-1]
            if "responses" in url_suffix:
                request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.RESPONSES.value
            else:
                request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.CHAT_COMPLETIONS.value
            request.url = URL(base_url)

        self.uipath_sync_client.event_hooks["request"].append(fix_url_and_api_flavor_header)
        self.uipath_async_client.event_hooks["request"].append(fix_url_and_api_flavor_header_async)

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
