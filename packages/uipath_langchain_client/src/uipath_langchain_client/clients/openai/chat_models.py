from collections.abc import Awaitable, Callable
from typing import Self

from httpx import Request
from pydantic import Field, SecretStr, model_validator

from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.clients.openai.utils import fix_url_and_api_flavor_header
from uipath_langchain_client.settings import (
    ApiType,
    RoutingMode,
    UiPathAPIConfig,
    VendorType,
)

try:
    from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required to use UiPathChatOpenAI and UiPathAzureChatOpenAI. "
        "Install it with: uv add uipath-langchain-client[openai]"
    ) from e


class UiPathChatOpenAI(UiPathBaseChatModel, ChatOpenAI):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.OPENAI,
        api_version="2025-03-01-preview",
        freeze_base_url=False,
    )

    # Override fields to avoid errors when instantiating the class
    openai_api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] = Field(
        alias="api_key", default=SecretStr("PLACEHOLDER")
    )

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
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
            base_url=base_url,
        )
        self.root_async_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_async_client,
            base_url=base_url,
        )
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        return self


class UiPathAzureChatOpenAI(UiPathBaseChatModel, AzureChatOpenAI):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.OPENAI,
        api_version="2025-03-01-preview",
        freeze_base_url=False,
    )

    # Override fields to avoid errors when instantiating the class
    azure_endpoint: str | None = Field(default="PLACEHOLDER")
    openai_api_version: str | None = Field(default="PLACEHOLDER", alias="api_version")
    openai_api_key: SecretStr | None = Field(default=SecretStr("PLACEHOLDER"), alias="api_key")

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        base_url = str(self.uipath_sync_client.base_url).rstrip("/")
        locked_flavor = str(self.api_config.api_flavor) if self.api_config.api_flavor else None

        def on_request(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        async def on_request_async(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        self.uipath_sync_client.event_hooks["request"].append(on_request)
        self.uipath_async_client.event_hooks["request"].append(on_request_async)

        self.root_client = AzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        )
        self.root_async_client = AsyncAzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        )
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        return self
