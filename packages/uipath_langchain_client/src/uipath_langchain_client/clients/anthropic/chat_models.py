from functools import cached_property
from typing import Any, Self

from pydantic import Field, model_validator
from typing_extensions import override

from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.settings import (
    ApiFlavor,
    ApiType,
    RoutingMode,
    UiPathAPIConfig,
    VendorType,
)

try:
    from anthropic import (
        Anthropic,
        AnthropicBedrock,
        AnthropicFoundry,
        AnthropicVertex,
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )
    from langchain_anthropic.chat_models import ChatAnthropic
except ImportError as e:
    raise ImportError(
        "The 'anthropic' extra is required to use UiPathChatAnthropic. "
        "Install it with: uv add uipath-langchain-client[anthropic]"
    ) from e


class UiPathChatAnthropic(UiPathBaseChatModel, ChatAnthropic):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.ANTHROPIC,
        freeze_base_url=True,
    )
    vendor_type: VendorType = VendorType.ANTHROPIC

    @model_validator(mode="after")
    def setup_api_flavor_and_version(self) -> Self:
        self.api_config.vendor_type = self.vendor_type
        match self.vendor_type:
            case VendorType.VERTEXAI:
                self.api_config.api_flavor = ApiFlavor.ANTHROPIC_CLAUDE
                self.api_config.api_version = "v1beta1"
            case VendorType.AWSBEDROCK:
                self.api_config.api_flavor = ApiFlavor.INVOKE
            case _:
                raise ValueError(
                    "anthropic and azure vendors are currently not supported by UiPath"
                )
        return self

    # Override fields to avoid typing issues and fix stuff
    stop_sequences: list[str] | None = Field(default=None, alias="stop")
    model: str = Field(default="", alias="model_name")
    default_request_timeout: float | None = None

    @cached_property
    def _anthropic_client(
        self,
    ) -> Anthropic | AnthropicVertex | AnthropicBedrock | AnthropicFoundry:
        match self.vendor_type:
            case VendorType.ANTHROPIC:
                return Anthropic(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_sync_client,
                )
            case VendorType.AZURE:
                return AnthropicFoundry(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_sync_client,
                )
            case VendorType.VERTEXAI:
                return AnthropicVertex(
                    region="PLACEHOLDER",
                    project_id="PLACEHOLDER",
                    access_token="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_sync_client,
                )
            case VendorType.AWSBEDROCK:
                return AnthropicBedrock(
                    aws_access_key="PLACEHOLDER",
                    aws_secret_key="PLACEHOLDER",
                    aws_region="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_sync_client,
                )
            case _:
                raise ValueError("Anthropic models are currently not hosted on any other provider")

    @cached_property
    def _async_anthropic_client(
        self,
    ) -> AsyncAnthropic | AsyncAnthropicVertex | AsyncAnthropicBedrock | AsyncAnthropicFoundry:
        match self.vendor_type:
            case VendorType.ANTHROPIC:
                return AsyncAnthropic(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_async_client,
                )
            case VendorType.AZURE:
                return AsyncAnthropicFoundry(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_async_client,
                )
            case VendorType.VERTEXAI:
                return AsyncAnthropicVertex(
                    region="PLACEHOLDER",
                    project_id="PLACEHOLDER",
                    access_token="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_async_client,
                )
            case VendorType.AWSBEDROCK:
                return AsyncAnthropicBedrock(
                    aws_access_key="PLACEHOLDER",
                    aws_secret_key="PLACEHOLDER",
                    aws_region="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    max_retries=0,  # handled by the UiPathBaseChatModel
                    http_client=self.uipath_async_client,
                )
            case _:
                raise ValueError("Anthropic models are currently not hosted on any other provider")

    @override
    def _create(self, payload: dict) -> Any:
        if "betas" in payload:
            return self._anthropic_client.beta.messages.create(**payload)
        return self._anthropic_client.messages.create(**payload)

    @override
    async def _acreate(self, payload: dict) -> Any:
        if "betas" in payload:
            return await self._async_anthropic_client.beta.messages.create(**payload)
        return await self._async_anthropic_client.messages.create(**payload)
