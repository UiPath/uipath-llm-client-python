from typing import Self

from pydantic import model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from anthropic import AnthropicVertex, AsyncAnthropicVertex
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex
except ImportError as e:
    raise ImportError(
        "The 'vertexai' extra is required to use UiPathChatAnthropicVertex. "
        "Install it with: uv add uipath-langchain-client[vertexai]"
    ) from e


class UiPathChatAnthropicVertex(UiPathBaseLLMClient, ChatAnthropicVertex):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="vertexai",
        api_flavor="anthropic-claude",
        freeze_base_url=True,
    )

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = AnthropicVertex(
            region="PLACEHOLDER",
            project_id="PLACEHOLDER",
            access_token="PLACEHOLDER",
            base_url=str(self.uipath_sync_client.base_url),
            default_headers=self.uipath_sync_client.headers,
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        )
        self.async_client = AsyncAnthropicVertex(
            region="PLACEHOLDER",
            project_id="PLACEHOLDER",
            access_token="PLACEHOLDER",
            base_url=str(self.uipath_async_client.base_url),
            default_headers=self.uipath_async_client.headers,
            timeout=None,  # handled by the UiPath client
            max_retries=1,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        )
        return self
