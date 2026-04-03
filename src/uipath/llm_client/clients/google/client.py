import logging
from collections.abc import Mapping, Sequence

from uipath.llm_client.clients.utils import build_httpx_async_client, build_httpx_client
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
    """Google GenAI client routed through UiPath LLM Gateway.

    Args:
        model_name: The Google model name (e.g., "gemini-2.5-flash").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        captured_headers: Response header prefixes to capture (case-insensitive).
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        captured_headers: Sequence[str] = ("x-uipath-",),
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
        httpx_client = build_httpx_client(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            client_settings=client_settings,
            api_config=api_config,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
        )
        httpx_async_client = build_httpx_async_client(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            client_settings=client_settings,
            api_config=api_config,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
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
