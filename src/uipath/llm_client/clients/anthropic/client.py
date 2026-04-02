"""UiPath Anthropic client wrappers for routing through UiPath LLM Gateway.

This module provides Anthropic client variants that route requests through
the UiPath LLM Gateway while preserving the full Anthropic SDK interface.

Example:
    >>> from uipath.llm_client.clients.anthropic import UiPathAnthropic
    >>>
    >>> client = UiPathAnthropic(model_name="claude-3-5-sonnet-20241022")
    >>> response = client.messages.create(
    ...     model="claude-3-5-sonnet-20241022",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=1024,
    ... )
"""

import logging
from collections.abc import Mapping, Sequence

from uipath.llm_client.clients.utils import build_httpx_async_client, build_httpx_client
from uipath.llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import ApiType, RoutingMode, VendorType
from uipath.llm_client.utils.retry import RetryConfig

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
except ImportError as e:
    raise ImportError(
        "The 'anthropic' extra is required to use UiPath Anthropic clients. "
        "Install it with: uv add uipath-llm-client[anthropic]"
    ) from e


def _build_api_config(vendor_type: str | VendorType = VendorType.ANTHROPIC) -> UiPathAPIConfig:
    """Build standard API config for Anthropic clients."""
    return UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=vendor_type,
        freeze_base_url=True,
    )


class UiPathAnthropic(Anthropic):
    """Anthropic client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name (e.g., "claude-3-5-sonnet-20241022").
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
        super().__init__(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )


class UiPathAsyncAnthropic(AsyncAnthropic):
    """Async Anthropic client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name (e.g., "claude-3-5-sonnet-20241022").
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
        super().__init__(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_async_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )


class UiPathAnthropicBedrock(AnthropicBedrock):
    """Anthropic Bedrock client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name.
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
        super().__init__(
            aws_access_key="PLACEHOLDER",
            aws_secret_key="PLACEHOLDER",
            aws_region="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(vendor_type=VendorType.AWSBEDROCK),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )


class UiPathAsyncAnthropicBedrock(AsyncAnthropicBedrock):
    """Async Anthropic Bedrock client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name.
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
        super().__init__(
            aws_access_key="PLACEHOLDER",
            aws_secret_key="PLACEHOLDER",
            aws_region="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_async_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(vendor_type=VendorType.AWSBEDROCK),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )


class UiPathAnthropicVertex(AnthropicVertex):
    """Anthropic Vertex client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name.
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
        super().__init__(
            region="PLACEHOLDER",
            project_id="PLACEHOLDER",
            access_token="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(vendor_type=VendorType.VERTEXAI),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )


class UiPathAsyncAnthropicVertex(AsyncAnthropicVertex):
    """Async Anthropic Vertex client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name.
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
        super().__init__(
            region="PLACEHOLDER",
            project_id="PLACEHOLDER",
            access_token="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_async_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(vendor_type=VendorType.VERTEXAI),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )


class UiPathAnthropicFoundry(AnthropicFoundry):
    """Anthropic Foundry (Azure) client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name.
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
        super().__init__(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(vendor_type=VendorType.AZURE),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )


class UiPathAsyncAnthropicFoundry(AsyncAnthropicFoundry):
    """Async Anthropic Foundry (Azure) client routed through UiPath LLM Gateway.

    Args:
        model_name: The Anthropic model name.
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
        super().__init__(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=build_httpx_async_client(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=_build_api_config(vendor_type=VendorType.AZURE),
                client_settings=client_settings,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                captured_headers=captured_headers,
                retry_config=retry_config,
                logger=logger,
            ),
        )
