import logging
from collections.abc import Mapping, Sequence

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType
from uipath.llm_client.utils.retry import RetryConfig

try:
    from google.genai.client import Client
    from google.genai.types import HttpOptions
except ImportError as e:
    raise ImportError(
        "The 'google' extra is required to use UiPathGoogleClient. "
        "Install it with: uv add uipath-llm-client[google]"
    ) from e


class UiPathGoogle(Client):
    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        default_headers: Mapping[str, str] | None = None,
        captured_headers: Sequence[str] = ("x-uipath-",),
        timeout: float | None = None,
        max_retries: int = 0,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type=VendorType.VERTEXAI,
            api_flavor=ApiFlavor.GENERATE_CONTENT,
            api_version="v1beta1",
            freeze_base_url=True,
        )
        merged_headers = {
            **(default_headers or {}),
            **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
        }
        base_url = client_settings.build_base_url(model_name=model_name, api_config=api_config)
        auth = client_settings.build_auth_pipeline()

        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=timeout,
            max_retries=max_retries,
            captured_headers=captured_headers,
            retry_config=retry_config,
            base_url=base_url,
            headers=merged_headers,
            logger=logger,
            auth=auth,
        )
        httpx_async_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=timeout,
            max_retries=max_retries,
            captured_headers=captured_headers,
            retry_config=retry_config,
            base_url=base_url,
            headers=merged_headers,
            logger=logger,
            auth=auth,
        )
        super().__init__(
            api_key="PLACEHOLDER",
            http_options=HttpOptions(
                base_url=str(httpx_client.base_url),
                headers=dict(httpx_client.headers),
                retry_options=None,  # handled by the UiPath client
                httpx_client=httpx_client,
                httpx_async_client=httpx_async_client,
            ),
        )
