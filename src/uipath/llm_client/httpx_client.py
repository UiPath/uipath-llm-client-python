"""UiPath-configured HTTPX clients with retry, logging, streaming support and others.

This module provides customized httpx Client and AsyncClient subclasses that include:
- Default UiPath LLM Gateway headers (timeout, full 4xx response)
- Automatic retry logic with configurable backoff
- Request/response logging with timing information
- Streaming header injection (X-UiPath-Streaming-Enabled)
- Optional URL freezing to prevent vendor SDK URL mutations

Example:
    >>> from uipath.llm_client.httpx_client import UiPathHttpxClient
    >>> from uipath.llm_client.settings import UiPathAPIConfig
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
from collections.abc import Callable, Sequence
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

from uipath.llm_client.settings.base import UiPathAPIConfig
from uipath.llm_client.utils.exceptions import patch_raise_for_status
from uipath.llm_client.utils.headers import (
    build_routing_headers,
    extract_matching_headers,
    get_dynamic_request_headers,
    set_captured_response_headers,
)
from uipath.llm_client.utils.logging import LoggingConfig
from uipath.llm_client.utils.retry import (
    RetryableAsyncHTTPTransport,
    RetryableHTTPTransport,
    RetryConfig,
)
from uipath.llm_client.utils.ssl_config import get_httpx_ssl_client_kwargs


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
    _default_headers: dict[str, str] = {
        "X-UiPath-LLMGateway-TimeoutSeconds": "295",  # server side timeout, default is 10, maximum is 300
        # "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",  # allow full 4xx responses (default is false) — removed from default to avoid PII leakage in logs
    }

    def __init__(
        self,
        *,
        model_name: str | None = None,
        byo_connection_id: str | None = None,
        api_config: UiPathAPIConfig | None = None,
        captured_headers: Sequence[str] = ("x-uipath-",),
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
            captured_headers: Case-insensitive header name prefixes to capture from
                responses. Captured headers are stored in a ContextVar and can be
                retrieved with get_captured_response_headers(). Defaults to ("x-uipath-",).
            max_retries: Maximum retry attempts for failed requests. Defaults to 0
                (retries disabled). Set to a positive integer to enable retries.
            retry_config: Custom retry configuration (backoff, retryable status codes).
            logger: Logger instance for request/response logging.
            **kwargs: Additional arguments passed to httpx.Client (e.g., base_url,
                timeout, auth, headers, transport, event_hooks).
        """
        self.model_name = model_name
        self.byo_connection_id = byo_connection_id
        self.api_config = api_config
        self._captured_headers = tuple(captured_headers)

        # Extract httpx.Client params that we need to modify
        headers: HeaderTypes | None = kwargs.pop("headers", None)
        transport: BaseTransport | None = kwargs.pop("transport", None)
        event_hooks: dict[str, list[Callable[..., Any]]] | None = kwargs.pop("event_hooks", None)

        # Merge headers: default -> api_config -> user provided
        merged_headers = Headers(self._default_headers)
        merged_headers.update(
            build_routing_headers(
                model_name=model_name, byo_connection_id=byo_connection_id, api_config=api_config
            )
        )
        if headers is not None:
            merged_headers.update(headers)

        self._freeze_base_url = self.api_config is not None and self.api_config.freeze_base_url

        # Setup retry transport if not provided
        if transport is None:
            transport = RetryableHTTPTransport(
                max_retries=max_retries if max_retries is not None else 0,
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
        event_hooks.setdefault("request", []).append(logging_config.log_request_duration)
        event_hooks.setdefault("response", []).append(logging_config.log_response_duration)
        event_hooks["response"].append(logging_config.log_error)

        # setup ssl context
        kwargs.update(get_httpx_ssl_client_kwargs())

        super().__init__(
            headers=merged_headers, transport=transport, event_hooks=event_hooks, **kwargs
        )

    def send(self, request: Request, *, stream: bool = False, **kwargs: Any) -> Response:
        """Send an HTTP request with UiPath-specific modifications.

        Injects the streaming header and optionally freezes the URL before sending.
        Captures matching response headers into a ContextVar for later retrieval.

        Args:
            request: The HTTP request to send.
            stream: Whether to stream the response body.
            **kwargs: Additional arguments passed to the parent send method.

        Returns:
            Response with patched raise_for_status() that raises UiPath exceptions.
        """
        if self._freeze_base_url:
            request.url = URL(str(self.base_url).rstrip("/"))
        request.headers[self._streaming_header] = str(stream).lower()
        dynamic_headers = get_dynamic_request_headers()
        if dynamic_headers:
            request.headers.update(dynamic_headers)
        response = super().send(request, stream=stream, **kwargs)
        if self._captured_headers:
            captured = extract_matching_headers(response.headers, self._captured_headers)
            if captured:
                set_captured_response_headers(captured)
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
    _default_headers: dict[str, str] = {
        "X-UiPath-LLMGateway-TimeoutSeconds": "295",  # server side timeout, default is 10, maximum is 300
        # "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",  # allow full 4xx responses (default is false) — removed from default to avoid PII leakage in logs
    }

    def __init__(
        self,
        *,
        model_name: str | None = None,
        byo_connection_id: str | None = None,
        api_config: UiPathAPIConfig | None = None,
        captured_headers: Sequence[str] = ("x-uipath-",),
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
            captured_headers: Case-insensitive header name prefixes to capture from
                responses. Captured headers are stored in a ContextVar and can be
                retrieved with get_captured_response_headers(). Defaults to ("x-uipath-",).
            max_retries: Maximum retry attempts for failed requests. Defaults to 0
                (retries disabled). Set to a positive integer to enable retries.
            retry_config: Custom retry configuration (backoff, retryable status codes).
            logger: Logger instance for request/response logging.
            **kwargs: Additional arguments passed to httpx.AsyncClient (e.g., base_url,
                timeout, auth, headers, transport, event_hooks).
        """
        self.model_name = model_name
        self.byo_connection_id = byo_connection_id
        self.api_config = api_config
        self._captured_headers = tuple(captured_headers)

        # Extract httpx.AsyncClient params that we need to modify
        headers: HeaderTypes | None = kwargs.pop("headers", None)
        transport: AsyncBaseTransport | None = kwargs.pop("transport", None)
        event_hooks: dict[str, list[Callable[..., Any]]] | None = kwargs.pop("event_hooks", None)

        # Merge headers: default -> api_config -> user provided
        merged_headers = Headers(self._default_headers)
        merged_headers.update(
            build_routing_headers(
                model_name=model_name, byo_connection_id=byo_connection_id, api_config=api_config
            )
        )
        if headers is not None:
            merged_headers.update(headers)

        self._freeze_base_url = self.api_config is not None and self.api_config.freeze_base_url

        # Setup retry transport if not provided
        if transport is None:
            transport = RetryableAsyncHTTPTransport(
                max_retries=max_retries if max_retries is not None else 0,
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
        event_hooks.setdefault("request", []).append(logging_config.alog_request_duration)
        event_hooks.setdefault("response", []).append(logging_config.alog_response_duration)
        event_hooks["response"].append(logging_config.alog_error)

        # setup ssl context
        kwargs.update(get_httpx_ssl_client_kwargs())

        super().__init__(
            headers=merged_headers, transport=transport, event_hooks=event_hooks, **kwargs
        )

    async def send(self, request: Request, *, stream: bool = False, **kwargs: Any) -> Response:
        """Send an HTTP request asynchronously with UiPath-specific modifications.

        Injects the streaming header and optionally freezes the URL before sending.
        Captures matching response headers into a ContextVar for later retrieval.

        Args:
            request: The HTTP request to send.
            stream: Whether to stream the response body.
            **kwargs: Additional arguments passed to the parent send method.

        Returns:
            Response with patched raise_for_status() that raises UiPath exceptions.
        """
        if self._freeze_base_url:
            request.url = URL(str(self.base_url).rstrip("/"))
        request.headers[self._streaming_header] = str(stream).lower()
        dynamic_headers = get_dynamic_request_headers()
        if dynamic_headers:
            request.headers.update(dynamic_headers)
        response = await super().send(request, stream=stream, **kwargs)
        if self._captured_headers:
            captured = extract_matching_headers(response.headers, self._captured_headers)
            if captured:
                set_captured_response_headers(captured)
        return patch_raise_for_status(response)
