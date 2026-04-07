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
import ssl
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from httpx import (
    URL,
    AsyncBaseTransport,
    AsyncClient,
    BaseTransport,
    Client,
    Headers,
    Limits,
    Request,
    Response,
)
from httpx._config import DEFAULT_LIMITS, DEFAULT_MAX_REDIRECTS, DEFAULT_TIMEOUT_CONFIG
from httpx._types import (
    AuthTypes,
    CertTypes,
    CookieTypes,
    HeaderTypes,
    ProxyTypes,
    QueryParamTypes,
    TimeoutTypes,
)

from uipath.llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
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

# Sentinel to distinguish "not provided" from an explicit ``None`` / ``False``.
_UNSET: Any = object()


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

    """

    _streaming_header: str = "X-UiPath-Streaming-Enabled"
    _default_headers: dict[str, str] = {
        "X-UiPath-LLMGateway-TimeoutSeconds": "295",  # server side timeout, default is 10, maximum is 300
        "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",  # allow full 4xx responses (default is false) — removed from default to avoid PII leakage in logs
    }

    def __init__(
        self,
        *,
        # UiPath-specific
        model_name: str | None = None,
        byo_connection_id: str | None = None,
        api_config: UiPathAPIConfig | None = None,
        client_settings: UiPathBaseSettings | None = None,
        captured_headers: Sequence[str] = ("x-uipath-",),
        max_retries: int | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        # httpx.Client
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: ssl.SSLContext | str | bool = _UNSET,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: Mapping[str, BaseTransport | None] | None = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = _UNSET,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Mapping[str, list[typing.Callable[..., Any]]] | None = None,
        base_url: URL | str = "",
        transport: BaseTransport | None = None,
        default_encoding: str | typing.Callable[[bytes], str] = "utf-8",
    ):
        """Initialize the UiPath HTTP client.

        Args:
            model_name: LLM model name for logging context.
            byo_connection_id: Bring Your Own connection ID for custom model deployments.
            client_settings: UiPath backend settings.  When provided the auth
                pipeline, base URL (if *api_config* is also set), and auth headers
                are derived automatically — you do not need to set *auth* or
                *base_url* manually.
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
            auth: HTTP authentication (same as httpx.Client).  Derived from
                *client_settings* when not provided explicitly.
            params: Default query parameters (same as httpx.Client).
            headers: Additional headers merged after UiPath defaults.
            cookies: Default cookies (same as httpx.Client).
            verify: SSL verification; defaults to UiPath's SSL config (system certs
                via truststore/certifi, or disabled via UIPATH_DISABLE_SSL_VERIFY).
            cert: Client-side TLS certificate (same as httpx.Client).
            trust_env: Read proxy/SSL env vars (same as httpx.Client).
            http1: Enable HTTP/1.1 (same as httpx.Client).
            http2: Enable HTTP/2 (same as httpx.Client).
            proxy: Proxy URL (same as httpx.Client).
            mounts: Transport mount map (same as httpx.Client).
            timeout: Request timeout (same as httpx.Client).
            follow_redirects: Follow redirects; defaults to True (UiPath default).
            limits: Connection pool limits (same as httpx.Client).
            max_redirects: Maximum number of redirects (same as httpx.Client).
            event_hooks: Event hooks; UiPath logging hooks are always appended.
            base_url: Base URL for requests (same as httpx.Client).  Derived from
                *client_settings* + *api_config* when not provided explicitly.
            transport: Custom transport; when None a retryable transport is created.
            default_encoding: Default text encoding (same as httpx.Client).
        """
        self._captured_headers = tuple(captured_headers)

        # Derive auth, base_url, and extra headers from client_settings
        if client_settings is not None:
            if auth is None:
                auth = client_settings.build_auth_pipeline()
            if api_config is not None:
                if not base_url:
                    base_url = client_settings.build_base_url(
                        model_name=model_name, api_config=api_config
                    )

        # Merge headers: default -> routing -> settings auth -> caller
        merged_headers = Headers(self._default_headers)
        merged_headers.update(
            build_routing_headers(
                model_name=model_name, byo_connection_id=byo_connection_id, api_config=api_config
            )
        )
        if client_settings is not None and api_config is not None:
            merged_headers.update(
                client_settings.build_auth_headers(model_name=model_name, api_config=api_config)
            )
        if headers is not None:
            merged_headers.update(headers)

        self._freeze_base_url = api_config is not None and api_config.freeze_base_url

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
        mutable_hooks: dict[str, list[Callable[..., Any]]] = (
            {k: list(v) for k, v in event_hooks.items()} if event_hooks is not None else {}
        )
        mutable_hooks.setdefault("request", []).append(logging_config.log_request_duration)
        mutable_hooks.setdefault("response", []).append(logging_config.log_response_duration)
        mutable_hooks["response"].append(logging_config.log_error)

        # Apply UiPath SSL defaults only when the caller did not provide explicit values
        ssl_defaults = get_httpx_ssl_client_kwargs()
        if verify is _UNSET:
            verify = ssl_defaults.get("verify", True)
        if follow_redirects is _UNSET:
            follow_redirects = ssl_defaults.get("follow_redirects", True)

        super().__init__(
            auth=auth,
            params=params,
            headers=merged_headers,
            cookies=cookies,
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            proxy=proxy,
            mounts=mounts,
            timeout=timeout,
            follow_redirects=follow_redirects,
            limits=limits,
            max_redirects=max_redirects,
            event_hooks=mutable_hooks,
            base_url=base_url,
            transport=transport,
            default_encoding=default_encoding,
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

    """

    _streaming_header: str = "X-UiPath-Streaming-Enabled"
    _default_headers: dict[str, str] = {
        "X-UiPath-LLMGateway-TimeoutSeconds": "295",  # server side timeout, default is 10, maximum is 300
        "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",  # allow full 4xx responses (default is false) — removed from default to avoid PII leakage in logs
    }

    def __init__(
        self,
        *,
        # UiPath-specific
        model_name: str | None = None,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        api_config: UiPathAPIConfig | None = None,
        captured_headers: Sequence[str] = ("x-uipath-",),
        max_retries: int | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        # httpx.AsyncClient
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: ssl.SSLContext | str | bool = _UNSET,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: Mapping[str, AsyncBaseTransport | None] | None = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = _UNSET,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Mapping[str, list[typing.Callable[..., Any]]] | None = None,
        base_url: URL | str = "",
        transport: AsyncBaseTransport | None = None,
        default_encoding: str | typing.Callable[[bytes], str] = "utf-8",
    ):
        """Initialize the UiPath async HTTP client.

        See :class:`UiPathHttpxClient` for full parameter documentation.
        """
        self._captured_headers = tuple(captured_headers)

        # Derive auth, base_url, and extra headers from client_settings
        if client_settings is not None:
            if auth is None:
                auth = client_settings.build_auth_pipeline()
            if api_config is not None:
                if not base_url:
                    base_url = client_settings.build_base_url(
                        model_name=model_name, api_config=api_config
                    )

        # Merge headers: default -> routing -> settings auth -> caller
        merged_headers = Headers(self._default_headers)
        merged_headers.update(
            build_routing_headers(
                model_name=model_name, byo_connection_id=byo_connection_id, api_config=api_config
            )
        )
        if client_settings is not None and api_config is not None:
            merged_headers.update(
                client_settings.build_auth_headers(model_name=model_name, api_config=api_config)
            )
        if headers is not None:
            merged_headers.update(headers)

        self._freeze_base_url = api_config is not None and api_config.freeze_base_url

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
        mutable_hooks: dict[str, list[Callable[..., Any]]] = (
            {k: list(v) for k, v in event_hooks.items()} if event_hooks is not None else {}
        )
        mutable_hooks.setdefault("request", []).append(logging_config.alog_request_duration)
        mutable_hooks.setdefault("response", []).append(logging_config.alog_response_duration)
        mutable_hooks["response"].append(logging_config.alog_error)

        # Apply UiPath SSL defaults only when the caller did not provide explicit values
        ssl_defaults = get_httpx_ssl_client_kwargs()
        if verify is _UNSET:
            verify = ssl_defaults.get("verify", True)
        if follow_redirects is _UNSET:
            follow_redirects = ssl_defaults.get("follow_redirects", True)

        super().__init__(
            auth=auth,
            params=params,
            headers=merged_headers,
            cookies=cookies,
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            proxy=proxy,
            mounts=mounts,
            timeout=timeout,
            follow_redirects=follow_redirects,
            limits=limits,
            max_redirects=max_redirects,
            event_hooks=mutable_hooks,
            base_url=base_url,
            transport=transport,
            default_encoding=default_encoding,
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
