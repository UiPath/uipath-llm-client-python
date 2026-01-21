"""
UiPath LLM Client - Core HTTP Client Module

This module provides the base HTTP client for interacting with UiPath's LLM services.
It handles authentication, request/response formatting, retry logic, and logging.

The UiPathBaseLLMClient class is designed to be used as a mixin with framework-specific
chat models (e.g., LangChain, LlamaIndex) to provide UiPath connectivity.

Example:
    >>> from uipath_llm_client import UiPathBaseLLMClient, UiPathAPIConfig
    >>> from uipath_langchain_client.settings import get_default_client_settings
    >>>
    >>> client = UiPathBaseLLMClient(
    ...     model="gpt-4o-2024-11-20",
    ...     api_config=UiPathAPIConfig(
    ...         api_type="completions",
    ...         client_type="passthrough",
    ...         vendor_type="openai",
    ...     ),
    ...     client_settings=get_default_client_settings(),
    ... )
    >>> response = client.uipath_request(request_body={"messages": [...]})
"""

import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from functools import cached_property
from typing import Any, Literal

from httpx import URL, Response
from pydantic import BaseModel, ConfigDict, Field

from uipath_langchain_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath_llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath_llm_client.utils.retry import RetryConfig


class UiPathBaseLLMClient(BaseModel):
    """Base HTTP client for interacting with UiPath's LLM services.

    Provides the underlying HTTP transport layer with support for:
        - Authentication and token management
        - Request URL and header formatting
        - Retry logic with configurable backoff
        - Request/response logging

    This class is typically used as a mixin with framework-specific chat models
    (e.g., LangChain, LlamaIndex) to provide UiPath connectivity.

    Attributes:
        model_name: Name of the LLM model to use (aliased as "model")
        byo_connection_id: Optional connection ID for Bring Your Own (BYO) models enrolled
            in LLMGateway. When provided, routes requests to your custom-enrolled model.
        api_config: API configuration (api_type, client_type, vendor_type, etc.)
        client_settings: Client configuration (base URL, auth headers, etc.)
        default_headers: Additional headers to include in requests
        request_timeout: Client-side request timeout in seconds
        retry_config: Configuration for retry behavior on failed requests
        logger: Logger instance for request/response logging
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_alias=True,
        validate_by_name=True,
        validate_default=True,
    )

    model_name: str = Field(
        alias="model", description="the LLM model name (completions or embeddings)"
    )
    byo_connection_id: str | None = Field(
        default=None,
        description="Bring Your Own (BYO) connection ID for custom models enrolled in LLMGateway. "
        "Use this when you have enrolled your own model deployment and received a connection ID.",
    )

    api_config: UiPathAPIConfig = Field(
        ...,
        description="Settings for the UiPath API",
    )
    client_settings: UiPathBaseSettings = Field(
        default_factory=get_default_client_settings,
        description="Settings for the UiPath client (defaults based on UIPATH_LLM_BACKEND env var)",
    )
    default_headers: Mapping[str, str] | None = Field(
        default={
            "X-UiPath-LLMGateway-TimeoutSeconds": "30",  # server side timeout, default is 10, maximum is 300
            "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",  # allow full 4xx responses (default is false)
        },
        description="Default request headers to include in requests",
    )

    request_timeout: int | None = Field(
        default=None,
        description="Client-side request timeout in seconds",
    )
    max_retries: int = Field(
        default=1,
        description="Maximum number of retries for failed requests",
    )
    retry_config: RetryConfig | None = Field(
        default=None,
        description="Retry configuration for failed requests",
    )
    logger: logging.Logger | None = Field(
        default=None,
        description="Logger for request/response logging",
    )

    @cached_property
    def uipath_sync_client(self) -> UiPathHttpxClient:
        """Here we instantiate a synchronous HTTP client with the proper authentication pipeline, retry logic, logging etc."""
        return UiPathHttpxClient(
            model_name=self.model_name,
            byo_connection_id=self.byo_connection_id,
            api_config=self.api_config,
            auth=self.client_settings.build_auth_pipeline(),
            base_url=self.client_settings.build_base_url(
                model_name=self.model_name, api_config=self.api_config
            ),
            headers={
                **(self.default_headers or {}),
                **self.client_settings.build_auth_headers(
                    model_name=self.model_name, api_config=self.api_config
                ),
            },
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_config=self.retry_config,
            logger=self.logger,
        )

    @cached_property
    def uipath_async_client(self) -> UiPathHttpxAsyncClient:
        """Here we instantiate an asynchronous HTTP client with the proper authentication pipeline, retry logic, logging etc."""
        return UiPathHttpxAsyncClient(
            model_name=self.model_name,
            byo_connection_id=self.byo_connection_id,
            api_config=self.api_config,
            auth=self.client_settings.build_auth_pipeline(),
            base_url=self.client_settings.build_base_url(
                model_name=self.model_name, api_config=self.api_config
            ),
            headers={
                **(self.default_headers or {}),
                **self.client_settings.build_auth_headers(
                    model_name=self.model_name, api_config=self.api_config
                ),
            },
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_config=self.retry_config,
            logger=self.logger,
        )

    def uipath_request(
        self,
        method: str = "POST",
        url: URL | str = "/",
        *,
        request_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Make a synchronous HTTP request to the UiPath API.

        Args:
            method: HTTP method (GET, POST, etc.). Defaults to "POST".
            url: Request URL path. Defaults to "/".
            request_body: JSON request body to send.
            **kwargs: Additional arguments passed to httpx.Client.request().

        Returns:
            httpx.Response: The HTTP response from the API.

        Raises:
            UiPathAPIError: On HTTP 4xx/5xx responses (raised by transport layer).
        """
        return self.uipath_sync_client.request(method, url, json=request_body, **kwargs)

    async def uipath_arequest(
        self,
        method: Literal["POST", "GET"] = "POST",
        url: str = "/",
        *,
        request_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Make an asynchronous HTTP request to the UiPath API."""
        return await self.uipath_async_client.request(method, url, json=request_body, **kwargs)

    def uipath_stream(
        self,
        method: Literal["POST", "GET"] = "POST",
        url: str = "/",
        *,
        request_body: dict[str, Any] | None = None,
        stream_type: Literal["text", "bytes", "lines", "raw"] = "lines",
        **kwargs: Any,
    ) -> Iterator[str | bytes]:
        """Make a synchronous streaming HTTP request to the UiPath API.

        Args:
            method: HTTP method (POST or GET). Defaults to "POST".
            url: Request URL path. Defaults to "/".
            request_body: JSON request body to send.
            stream_type: Type of stream iteration:
                - "text": Yield decoded text chunks
                - "bytes": Yield raw byte chunks
                - "lines": Yield complete lines (default, best for SSE)
                - "raw": Yield raw response data
            **kwargs: Additional arguments passed to httpx.Client.stream().

        Yields:
            str | bytes: Chunks of the streaming response.
        """
        with self.uipath_sync_client.stream(method, url, json=request_body, **kwargs) as response:
            match stream_type:
                case "text":
                    for chunk in response.iter_text():
                        yield chunk
                case "bytes":
                    for chunk in response.iter_bytes():
                        yield chunk
                case "lines":
                    for chunk in response.iter_lines():
                        yield chunk
                case "raw":
                    for chunk in response.iter_raw():
                        yield chunk

    async def uipath_astream(
        self,
        method: Literal["POST", "GET"] = "POST",
        url: str = "/",
        *,
        request_body: dict[str, Any] | None = None,
        stream_type: Literal["text", "bytes", "lines", "raw"] = "lines",
        **kwargs: Any,
    ) -> AsyncIterator[str | bytes]:
        """Make an asynchronous streaming HTTP request to the UiPath API.

        Args:
            method: HTTP method (POST or GET). Defaults to "POST".
            url: Request URL path. Defaults to "/".
            request_body: JSON request body to send.
            stream_type: Type of stream iteration:
                - "text": Yield decoded text chunks
                - "bytes": Yield raw byte chunks
                - "lines": Yield complete lines (default, best for SSE)
                - "raw": Yield raw response data
            **kwargs: Additional arguments passed to httpx.AsyncClient.stream().

        Yields:
            str | bytes: Chunks of the streaming response.
        """
        async with self.uipath_async_client.stream(
            method, url, json=request_body, **kwargs
        ) as response:
            match stream_type:
                case "text":
                    async for chunk in response.aiter_text():
                        yield chunk
                case "bytes":
                    async for chunk in response.aiter_bytes():
                        yield chunk
                case "lines":
                    async for chunk in response.aiter_lines():
                        yield chunk
                case "raw":
                    async for chunk in response.aiter_raw():
                        yield chunk
