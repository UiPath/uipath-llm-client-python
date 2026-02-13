from typing import Self

from pydantic import Field, SecretStr, model_validator

from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from fireworks.client.api_client import FireworksClient as FireworksClientV1
    from langchain_fireworks.chat_models import ChatFireworks
except ImportError as e:
    raise ImportError(
        "The 'fireworks' extra is required to use UiPathChatFireworks. "
        'Install it with: uv add "uipath-langchain-client[fireworks]"'
    ) from e


class UiPathChatFireworks(UiPathBaseLLMClient, ChatFireworks):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="openai",
        api_flavor="chat-completions",
        api_version="2025-03-01-preview",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    fireworks_api_base: str | None = Field(alias="base_url", default="PLACEHOLDER")
    fireworks_api_key: SecretStr = Field(default=SecretStr("PLACEHOLDER"), alias="api_key")

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        fireworks_client_v1 = FireworksClientV1(
            api_key=self.fireworks_api_key.get_secret_value(),
            base_url=self.fireworks_api_base,
        )
        fireworks_client_v1._client = self.uipath_sync_client
        fireworks_client_v1._async_client = self.uipath_async_client
        self.client._client = fireworks_client_v1
        self.async_client._client = fireworks_client_v1
        return self
