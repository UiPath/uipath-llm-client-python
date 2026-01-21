from functools import cached_property
from typing import Any, Literal, Self, override

from pydantic import Field, model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_anthropic.chat_models import ChatAnthropic

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
except ImportError as e:
    raise ImportError(
        "The 'anthropic' extra is required to use UiPathChatAnthropic. "
        "Install it with: uv add uipath-langchain-client[anthropic]"
    ) from e


class UiPathChatAnthropic(UiPathBaseLLMClient, ChatAnthropic):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="anthropic",
        freeze_base_url=True,
    )
    vendor_type: Literal["anthropic", "azure", "vertexai", "awsbedrock"] = "awsbedrock"

    @model_validator(mode="after")
    def setup_api_flavor_and_version(self) -> Self:
        match self.vendor_type:
            case "vertexai":
                self.api_config.api_flavor = "anthropic-claude"
                self.api_config.api_version = "v1beta1"
            case "awsbedrock":
                self.api_config.api_flavor = "invoke"
            case _:
                raise ValueError("Those vendors are currently not supported")
        self.api_config.vendor_type = self.vendor_type
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
            case "azure":
                return AnthropicFoundry(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_sync_client,
                )
            case "vertexai":
                return AnthropicVertex(
                    region="PLACEHOLDER",
                    project_id="PLACEHOLDER",
                    access_token="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_sync_client,
                )
            case "awsbedrock":
                return AnthropicBedrock(
                    aws_access_key="PLACEHOLDER",
                    aws_secret_key="PLACEHOLDER",
                    aws_region="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_sync_client,
                )
            case "anthropic":
                return Anthropic(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_sync_client.base_url),
                    default_headers=dict(self.uipath_sync_client.headers),
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_sync_client,
                )

    @cached_property
    def _async_anthropic_client(
        self,
    ) -> AsyncAnthropic | AsyncAnthropicVertex | AsyncAnthropicBedrock | AsyncAnthropicFoundry:
        match self.vendor_type:
            case "azure":
                return AsyncAnthropicFoundry(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_async_client,
                )
            case "vertexai":
                return AsyncAnthropicVertex(
                    region="PLACEHOLDER",
                    project_id="PLACEHOLDER",
                    access_token="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_async_client,
                )
            case "awsbedrock":
                return AsyncAnthropicBedrock(
                    aws_access_key="PLACEHOLDER",
                    aws_secret_key="PLACEHOLDER",
                    aws_region="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_async_client,
                )
            case _:
                return AsyncAnthropic(
                    api_key="PLACEHOLDER",
                    base_url=str(self.uipath_async_client.base_url),
                    default_headers=dict(self.uipath_async_client.headers),
                    timeout=None,  # handled by the UiPathBaseLLMClient
                    max_retries=1,  # handled by the UiPathBaseLLMClient
                    http_client=self.uipath_async_client,
                )

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
