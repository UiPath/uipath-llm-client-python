"""UiPath-configured HTTPX clients with retry, logging, streaming support and others.

This module provides customized httpx Client and AsyncClient subclasses that include:
- Default UiPath LLM Gateway headers (timeout, full 4xx response)
- Automatic retry logic with configurable backoff
- Request/response logging with timing information
- Streaming header injection (X-UiPath-Streaming-Enabled)
- Optional URL freezing to prevent vendor SDK URL mutations

Example:
    >>> from uipath_llm_client.httpx_client import UiPathHttpxClient
    >>> from uipath_llm_client.settings import UiPathAPIConfig
    >>>
    >>> client = UiPathHttpxClient(
    ...     base_url="https://cloud.uipath.com/org/tenant/llmgateway_/",
    ...     model_name="gpt-4o",
    ...     api_config=UiPathAPIConfig(...),
    ...     max_retries=3,
    ... )
    >>> response = client.post("/chat/completions", json={"messages": [...]})
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from httpx import (
    URL,
    AsyncBaseTransport,
    AsyncClient,
    BaseTransport,
    Client,
    Headers,
    Request,
    Response,
)
from httpx._types import HeaderTypes
from uipath._utils._ssl_context import get_httpx_client_kwargs

from uipath_llm_client.settings import (
    UiPathAPIConfig,
)
from uipath_llm_client.utils.exceptions import patch_raise_for_status
from uipath_llm_client.utils.logging import LoggingConfig
from uipath_llm_client.utils.retry import (
    RetryableAsyncHTTPTransport,
    RetryableHTTPTransport,
    RetryConfig,
)


def build_routing_headers(
    *,
    model_name: str | None = None,
    byo_connection_id: str | None = None,
    api_config: UiPathAPIConfig | None = None,
) -> Mapping[str, str]:
    """Build UiPath LLM Gateway routing headers based on configuration.

    Args:
        api_config: UiPath API configuration.
        model_name: LLM model name (required for normalized API).
        byo_connection_id: Bring Your Own connection ID.

    Returns:
        Headers mapping for routing requests through the gateway.
    """
    headers: dict[str, str] = {}
    if api_config is not None:
        if api_config.client_type == "normalized" and model_name is not None:
            headers["X-UiPath-LlmGateway-NormalizedApi-ModelName"] = model_name
        elif api_config.client_type == "passthrough" and api_config.api_type == "completions":
            if api_config.api_flavor is not None:
                headers["X-UiPath-LlmGateway-ApiFlavor"] = api_config.api_flavor
            if api_config.api_version is not None:
                headers["X-UiPath-LlmGateway-ApiVersion"] = api_config.api_version
    if byo_connection_id is not None:
        headers["X-UiPath-LlmGateway-ByoConnectionId"] = byo_connection_id
    return headers


class UiPathHttpxClient(Client):
    """Synchronous HTTP client configured for UiPath LLM services.

    Extends httpx.Client with:
    - Default UiPath headers (server timeout, full 4xx responses)
    - Automatic retry on transient failures (429, 5xx)
    - Request/response duration logging
    - Streaming header injection (X-UiPath-Streaming-Enabled)
    - Optional URL freezing to prevent vendor SDK mutations

    Headers are merged in order: default headers -> api_config headers -> user headers.
    Later headers override earlier ones with the same key.

    Attributes:
        model_name: The LLM model name (for logging purposes).
        api_config: UiPath API configuration settings.
    """

    _streaming_header: str = "X-UiPath-Streaming-Enabled"
    _default_headers: Mapping[str, str] = {
        "X-UiPath-LLMGateway-TimeoutSeconds": "30",  # server side timeout, default is 10, maximum is 300
        "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",  # allow full 4xx responses (default is false)
    }

    def __init__(
        self,
        *,
        model_name: str | None = None,
        byo_connection_id: str | None = None,
        api_config: UiPathAPIConfig | None = None,
        max_retries: int | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        """Initialize the UiPath HTTP client.

        Args:
            model_name: LLM model name for logging context.
            byo_connection_id: Bring Your Own connection ID for custom model deployments.
            api_config: UiPath API configuration (api_type, vendor_type, etc.).
                Provides additional headers via build_headers() and controls URL
                freezing via freeze_base_url attribute.
            max_retries: Maximum retry attempts for failed requests. Defaults to 1.
            retry_config: Custom retry configuration (backoff, retryable status codes).
            logger: Logger instance for request/response logging.
            **kwargs: Additional arguments passed to httpx.Client (e.g., base_url,
                timeout, auth, headers, transport, event_hooks).
        """
        self.model_name = model_name
        self.byo_connection_id = byo_connection_id
        self.api_config = api_config

        # Extract httpx.Client params that we need to modify
        headers: HeaderTypes | None = kwargs.pop("headers", None)
        transport: BaseTransport | None = kwargs.pop("transport", None)
        event_hooks: Mapping[str, list[Callable[..., Any]]] | None = kwargs.pop("event_hooks", None)

        # Merge headers: default -> api_config -> user provided
        merged_headers = Headers(self._default_headers)
        merged_headers.update(
            build_routing_headers(
                api_config=api_config, model_name=model_name, byo_connection_id=byo_connection_id
            )
        )
        if headers is not None:
            merged_headers.update(headers)

        self._freeze_base_url = self.api_config is not None and self.api_config.freeze_base_url

        # Setup retry transport if not provided
        if transport is None:
            transport = RetryableHTTPTransport(
                max_retries=max_retries or 1,
                retry_config=retry_config,
                logger=logger,
            )

        # Setup logging hooks
        logging_config = LoggingConfig(
            logger=logger,
            model_name=model_name,
            api_config=api_config,
        )
        if event_hooks is None:
            event_hooks = {
                "request": [],
                "response": [],
            }
        event_hooks["request"].append(logging_config.log_request_duration)
        event_hooks["response"].append(logging_config.log_response_duration)
        event_hooks["response"].append(logging_config.log_error)

        # setup ssl context
        kwargs.update(get_httpx_client_kwargs())

        super().__init__(
            headers=merged_headers, transport=transport, event_hooks=event_hooks, **kwargs
        )

    def send(self, request: Request, *, stream: bool = False, **kwargs: Any) -> Response:
        """Send an HTTP request with UiPath-specific modifications.

        Injects the streaming header and optionally freezes the URL before sending.

        Args:
            request: The HTTP request to send.
            stream: Whether to stream the response body.
            **kwargs: Additional arguments passed to the parent send method.

        Returns:
            Response with patched raise_for_status() that raises UiPath exceptions.
        """
        if self._freeze_base_url:
            request.url = URL(self.base_url)
        request.headers[self._streaming_header] = str(stream).lower()
        response = super().send(request, stream=stream, **kwargs)
        return patch_raise_for_status(response)


class UiPathHttpxAsyncClient(AsyncClient):
    """Asynchronous HTTP client configured for UiPath LLM services.

    Extends httpx.AsyncClient with:
    - Default UiPath headers (server timeout, full 4xx responses)
    - Automatic retry on transient failures (429, 5xx)
    - Request/response duration logging
    - Streaming header injection (X-UiPath-Streaming-Enabled)
    - Optional URL freezing to prevent vendor SDK mutations

    Headers are merged in order: default headers -> api_config headers -> user headers.
    Later headers override earlier ones with the same key.

    Attributes:
        model_name: The LLM model name (for logging purposes).
        api_config: UiPath API configuration settings.
    """

    _streaming_header: str = "X-UiPath-Streaming-Enabled"
    _default_headers: Mapping[str, str] = {
        "X-UiPath-LLMGateway-TimeoutSeconds": "30",  # server side timeout, default is 10, maximum is 300
        "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",  # allow full 4xx responses (default is false)
    }

    def __init__(
        self,
        *,
        model_name: str | None = None,
        byo_connection_id: str | None = None,
        api_config: UiPathAPIConfig | None = None,
        max_retries: int | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        """Initialize the UiPath async HTTP client.

        Args:
            model_name: LLM model name for logging context.
            byo_connection_id: Bring Your Own connection ID for custom model deployments.
            api_config: UiPath API configuration (api_type, vendor_type, etc.).
                Provides additional headers via build_headers() and controls URL
                freezing via freeze_base_url attribute.
            max_retries: Maximum retry attempts for failed requests. Defaults to 1.
            retry_config: Custom retry configuration (backoff, retryable status codes).
            logger: Logger instance for request/response logging.
            **kwargs: Additional arguments passed to httpx.AsyncClient (e.g., base_url,
                timeout, auth, headers, transport, event_hooks).
        """
        self.model_name = model_name
        self.byo_connection_id = byo_connection_id
        self.api_config = api_config

        # Extract httpx.AsyncClient params that we need to modify
        headers: HeaderTypes | None = kwargs.pop("headers", None)
        transport: AsyncBaseTransport | None = kwargs.pop("transport", None)
        event_hooks: Mapping[str, list[Callable[..., Any]]] | None = kwargs.pop("event_hooks", None)

        # Merge headers: default -> api_config -> user provided
        merged_headers = Headers(self._default_headers)
        merged_headers.update(
            build_routing_headers(
                api_config=api_config, model_name=model_name, byo_connection_id=byo_connection_id
            )
        )
        if headers is not None:
            merged_headers.update(headers)

        self._freeze_base_url = self.api_config is not None and self.api_config.freeze_base_url

        # Setup retry transport if not provided
        if transport is None:
            transport = RetryableAsyncHTTPTransport(
                max_retries=max_retries or 1,
                retry_config=retry_config,
                logger=logger,
            )

        # Setup logging hooks
        logging_config = LoggingConfig(
            logger=logger,
            model_name=model_name,
            api_config=api_config,
        )
        if event_hooks is None:
            event_hooks = {
                "request": [],
                "response": [],
            }
        event_hooks["request"].append(logging_config.alog_request_duration)
        event_hooks["response"].append(logging_config.alog_response_duration)
        event_hooks["response"].append(logging_config.alog_error)

        # setup ssl context
        kwargs.update(get_httpx_client_kwargs())

        super().__init__(
            headers=merged_headers, transport=transport, event_hooks=event_hooks, **kwargs
        )

    async def send(self, request: Request, *, stream: bool = False, **kwargs: Any) -> Response:
        """Send an HTTP request asynchronously with UiPath-specific modifications.

        Injects the streaming header and optionally freezes the URL before sending.

        Args:
            request: The HTTP request to send.
            stream: Whether to stream the response body.
            **kwargs: Additional arguments passed to the parent send method.

        Returns:
            Response with patched raise_for_status() that raises UiPath exceptions.
        """
        if self._freeze_base_url:
            request.url = URL(self.base_url)
        request.headers[self._streaming_header] = str(stream).lower()
        response = await super().send(request, stream=stream, **kwargs)
        return patch_raise_for_status(response)
