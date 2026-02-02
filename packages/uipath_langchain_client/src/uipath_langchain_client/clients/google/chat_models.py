from typing import Self

from httpx import URL, Request
from pydantic import Field, SecretStr, model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    from google.genai.client import Client
    from google.genai.types import HttpOptions
except ImportError as e:
    raise ImportError(
        "The 'google' extra is required to use UiPathChatGoogleGenerativeAI. "
        "Install it with: uv add uipath-langchain-client[google]"
    ) from e


class UiPathChatGoogleGenerativeAI(UiPathBaseLLMClient, ChatGoogleGenerativeAI):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="vertexai",
        api_flavor="generate-content",
        api_version="v1beta1",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model: str = Field(default="", alias="model_name")
    google_api_key: SecretStr | None = Field(default=SecretStr("PLACEHOLDER"))

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        def fix_url_for_streaming(request: Request):
            if request.headers.get("X-UiPath-Streaming-Enabled") == "true":
                request.url = URL(request.url).copy_add_param("alt", "sse")

        async def fix_url_for_streaming_async(request: Request):
            if request.headers.get("X-UiPath-Streaming-Enabled") == "true":
                request.url = URL(request.url).copy_add_param("alt", "sse")

        self.uipath_sync_client.event_hooks["request"].append(fix_url_for_streaming)
        self.uipath_async_client.event_hooks["request"].append(fix_url_for_streaming_async)

        self.client = Client(
            vertexai=True,
            api_key="PLACEHOLDER",
            http_options=HttpOptions(
                base_url=str(self.uipath_sync_client.base_url),
                headers=dict(self.uipath_sync_client.headers),
                timeout=None,  # handled by the UiPath client
                retry_options=None,  # handled by the UiPath client
                httpx_client=self.uipath_sync_client,
                httpx_async_client=self.uipath_async_client,
            ),
        )
        return self
