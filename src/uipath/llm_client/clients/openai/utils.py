from httpx import URL, Request

from uipath.llm_client.httpx_client import build_routing_headers
from uipath.llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType


class OpenAIRequestHandler:
    def __init__(
        self,
        model_name: str,
        client_settings: UiPathBaseSettings,
        byo_connection_id: str | None = None,
    ):
        self.model_name = model_name
        self.client_settings = client_settings
        self.byo_connection_id = byo_connection_id
        self.base_api_config = UiPathAPIConfig(
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type=VendorType.OPENAI,
            api_version="2025-03-01-preview",
            freeze_base_url=False,
        )

    def fix_url_and_headers(self, request: Request):
        if request.url.path.endswith("/completions"):
            api_config = self.base_api_config.model_copy(
                update={"api_flavor": ApiFlavor.CHAT_COMPLETIONS, "api_type": ApiType.COMPLETIONS}
            )
            request.headers.update(
                build_routing_headers(
                    model_name=self.model_name,
                    byo_connection_id=self.byo_connection_id,
                    api_config=api_config,
                )
            )
            request.url = URL(
                self.client_settings.build_base_url(
                    model_name=self.model_name, api_config=api_config
                )
            )
        elif request.url.path.endswith("/responses"):
            api_config = self.base_api_config.model_copy(
                update={"api_flavor": ApiFlavor.RESPONSES, "api_type": ApiType.COMPLETIONS}
            )
            request.headers.update(
                build_routing_headers(
                    model_name=self.model_name,
                    byo_connection_id=self.byo_connection_id,
                    api_config=api_config,
                )
            )
            request.url = URL(
                self.client_settings.build_base_url(
                    model_name=self.model_name, api_config=api_config
                )
            )
        elif request.url.path.endswith("/embeddings"):
            api_config = self.base_api_config.model_copy(update={"api_type": ApiType.EMBEDDINGS})
            request.headers.update(
                build_routing_headers(
                    model_name=self.model_name,
                    byo_connection_id=self.byo_connection_id,
                    api_config=api_config,
                )
            )
            request.url = URL(
                self.client_settings.build_base_url(
                    model_name=self.model_name, api_config=api_config
                )
            )
        else:
            raise ValueError(f"Unsupported API endpoint: {request.url.path}")

    async def fix_url_and_headers_async(self, request: Request):
        self.fix_url_and_headers(request)
