import logging
from typing import Any

from uipath_llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath_llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath_llm_client.utils.retry import RetryConfig

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
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = UiPathAPIConfig(
            api_type="completions",
            client_type="passthrough",
            vendor_type="vertexai",
            api_flavor="generate-content",
            api_version="v1beta1",
            freeze_base_url=True,
        )
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        httpx_async_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            api_key="PLACEHOLDER",
            http_options=HttpOptions(
                base_url=str(httpx_client.base_url),
                headers=dict(httpx_client.headers),
                timeout=None,  # handled by the UiPath client
                retry_options=None,  # handled by the UiPath client
                httpx_client=httpx_client,
                httpx_async_client=httpx_async_client,
            ),
        )
