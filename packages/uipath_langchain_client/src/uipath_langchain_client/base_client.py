"""
UiPath LLM Client - Core HTTP Client Module

This module provides the base HTTP client for interacting with UiPath's LLM services.
It handles authentication, request/response formatting, retry logic, and logging.

The UiPathBaseLLMClient class is designed to be used as a mixin with framework-specific
chat models (e.g., LangChain, LlamaIndex) to provide UiPath connectivity.

Example:
    >>> from uipath_langchain_client.base_client import UiPathBaseLLMClient
    >>> from uipath_langchain_client.settings import UiPathAPIConfig, get_default_client_settings
    >>>
    >>> client = UiPathBaseLLMClient(
    ...     model="gpt-4o-2024-11-20",
    ...     api_config=UiPathAPIConfig(
    ...         api_type=ApiType.COMPLETIONS,
    ...         routing_mode=RoutingMode.PASSTHROUGH,
    ...         vendor_type="openai",
    ...     ),
    ...     client_settings=get_default_client_settings(),
    ... )
    >>> response = client.uipath_request(request_body={"messages": [...]})
"""

import logging
from abc import ABC
from collections.abc import AsyncGenerator, Generator, Mapping, Sequence
from functools import cached_property
from typing import Any, ClassVar, Literal, Self

from httpx import URL, Response
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from uipath.llm_client.httpx_client import (
    UiPathHttpxAsyncClient,
    UiPathHttpxClient,
)
from uipath.llm_client.utils.headers import (
    UIPATH_DEFAULT_REQUEST_HEADERS,
    get_captured_response_headers,
    set_captured_response_headers,
)
from uipath_langchain_client._sampling import strip_disabled_sampling_kwargs
from uipath_langchain_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath_langchain_client.utils import RetryConfig


class UiPathBaseLLMClient(BaseModel, ABC):
    """Base HTTP client for interacting with UiPath's LLM services.

    Provides the underlying HTTP transport layer with support for:
        - Authentication and token management
        - Request URL and header formatting
        - Retry logic with configurable backoff
        - Request/response logging

    This class is typically used as a mixin with framework-specific chat models

    Attributes:
        model_name: Name of the LLM model to use (aliased as "model")
        byo_connection_id: Optional connection ID for Bring Your Own (BYO) models enrolled
            in LLMGateway. When provided, routes requests to your custom-enrolled model.
        api_config: API configuration (api_type, routing_mode, vendor_type, etc.)
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

    class_default_headers: ClassVar[dict[str, str]] = UIPATH_DEFAULT_REQUEST_HEADERS

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
        alias="settings",
        default_factory=get_default_client_settings,
        description="Settings for the UiPath client (defaults based on UIPATH_LLM_SERVICE env var)",
    )

    model_details: dict[str, Any] | None = Field(
        default=None,
        description="Per-model capability flags sourced from the discovery endpoint "
        "(e.g. {'shouldSkipTemperature': True}). The factory forwards it; direct "
        "instantiation lazy-resolves it from client_settings on first construction.",
    )

    default_headers: Mapping[str, str] | None = Field(
        default=None,
        description="Caller-supplied request headers. Merged on top of `class_default_headers`; "
        "user values win on key collisions. Does not remove built-in defaults.",
    )
    captured_headers: tuple[str, ...] = Field(
        default=("x-uipath-",),
        description="Case-insensitive response header prefixes to capture from LLM Gateway responses. "
        "Captured headers appear in response_metadata under the 'headers' key. "
        "Set to an empty tuple to disable.",
    )

    request_timeout: float | None = Field(
        alias="timeout",
        validation_alias=AliasChoices("timeout", "request_timeout", "default_request_timeout"),
        default=None,
        description="Client-side request timeout in seconds",
    )
    max_retries: int = Field(
        default=0,
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

    @model_validator(mode="after")
    def _resolve_model_details(self) -> Self:
        # Populate model_details eagerly so direct instantiation behaves the
        # same as the factory path. get_available_models is class-cached inside
        # the settings layer, so at most one discovery HTTP call fires per
        # process regardless of how many chat/embedding models are built.
        # Placed on UiPathBaseLLMClient (not just the chat subclass) because
        # model_details is meaningful for embedding wrappers too.
        if self.model_details is None:
            try:
                info = self.client_settings.get_model_info(
                    self.model_name,
                    byo_connection_id=self.byo_connection_id,
                )
                self.model_details = info.get("modelDetails") or {}
            except Exception:
                self.model_details = {}
        return self

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
                **self.class_default_headers,
                **(self.default_headers or {}),
                **self.client_settings.build_auth_headers(
                    model_name=self.model_name, api_config=self.api_config
                ),
            },
            captured_headers=self.captured_headers,
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
                **self.class_default_headers,
                **(self.default_headers or {}),
                **self.client_settings.build_auth_headers(
                    model_name=self.model_name, api_config=self.api_config
                ),
            },
            captured_headers=self.captured_headers,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_config=self.retry_config,
            logger=self.logger,
        )

    def uipath_request(
        self,
        method: Literal["POST", "GET"] = "POST",
        url: URL | str = "/",
        *,
        request_body: dict[str, Any] | None = None,
        raise_status_error: bool = False,
        **kwargs: Any,
    ) -> Response:
        """Make a synchronous HTTP request to the UiPath API.

        Args:
            method: HTTP method (POST or GET). Defaults to "POST".
            url: Request URL path. Defaults to "/".
            request_body: JSON request body to send.
            raise_status_error: If True, raises UiPathAPIError on non-2xx responses.
            **kwargs: Additional arguments passed to httpx.Client.request().

        Returns:
            httpx.Response: The HTTP response from the API.

        Raises:
            UiPathAPIError: On HTTP 4xx/5xx responses when raise_status_error is True,
                or raised by the transport layer.
        """
        response = self.uipath_sync_client.request(method, url, json=request_body, **kwargs)
        if raise_status_error:
            response.raise_for_status()
        return response

    async def uipath_arequest(
        self,
        method: Literal["POST", "GET"] = "POST",
        url: URL | str = "/",
        *,
        request_body: dict[str, Any] | None = None,
        raise_status_error: bool = False,
        **kwargs: Any,
    ) -> Response:
        """Make an asynchronous HTTP request to the UiPath API.

        Args:
            method: HTTP method (POST or GET). Defaults to "POST".
            url: Request URL path. Defaults to "/".
            request_body: JSON request body to send.
            raise_status_error: If True, raises UiPathAPIError on non-2xx responses.
            **kwargs: Additional arguments passed to httpx.AsyncClient.request().

        Returns:
            httpx.Response: The HTTP response from the API.

        Raises:
            UiPathAPIError: On HTTP 4xx/5xx responses when raise_status_error is True,
                or raised by the transport layer.
        """
        response = await self.uipath_async_client.request(method, url, json=request_body, **kwargs)
        if raise_status_error:
            response.raise_for_status()
        return response

    def uipath_stream(
        self,
        method: Literal["POST", "GET"] = "POST",
        url: str = "/",
        *,
        request_body: dict[str, Any] | None = None,
        stream_type: Literal["text", "bytes", "lines", "raw"] = "lines",
        raise_status_error: bool = False,
        **kwargs: Any,
    ) -> Generator[str | bytes, None, None]:
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
            raise_status_error: If True, raises UiPathAPIError on non-2xx responses.
            **kwargs: Additional arguments passed to httpx.Client.stream().

        Yields:
            str | bytes: Chunks of the streaming response.
        """
        with self.uipath_sync_client.stream(method, url, json=request_body, **kwargs) as response:
            if raise_status_error:
                response.raise_for_status()
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
        raise_status_error: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[str | bytes, None]:
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
            raise_status_error: If True, raises UiPathAPIError on non-2xx responses.
            **kwargs: Additional arguments passed to httpx.AsyncClient.stream().

        Yields:
            str | bytes: Chunks of the streaming response.
        """
        async with self.uipath_async_client.stream(
            method, url, json=request_body, **kwargs
        ) as response:
            if raise_status_error:
                response.raise_for_status()
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


class UiPathBaseChatModel(UiPathBaseLLMClient, BaseChatModel):
    """Base chat model that captures LLM Gateway response headers into response_metadata.

    Wraps _generate/_agenerate/_stream/_astream to automatically read captured headers
    from the ContextVar (populated by the httpx client's send()) and inject them into
    the AIMessage's response_metadata under the 'headers' key.

    Dynamic request headers are injected via UiPathDynamicHeadersCallback: set
    ``run_inline = True`` (already the default) so LangChain calls
    ``on_chat_model_start`` in the same coroutine as ``_agenerate``, ensuring the
    ContextVar is visible when ``httpx.send()`` fires.

    Passthrough clients that delegate to vendor SDKs should inherit from this class
    so that headers are captured transparently.
    """

    def _strip_sampling(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop sampling kwargs the model's ``modelDetails`` flags as unsupported."""
        return strip_disabled_sampling_kwargs(
            kwargs,
            model_details=self.model_details,
            model_name=self.model_name,
            logger=self.logger,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs = self._strip_sampling(kwargs)
        set_captured_response_headers({})
        try:
            result = self._uipath_generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self._inject_gateway_headers(result.generations)
            return result
        finally:
            set_captured_response_headers({})

    def _uipath_generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override in subclasses to provide the core (non-wrapped) generate logic."""
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs = self._strip_sampling(kwargs)
        set_captured_response_headers({})
        try:
            result = await self._uipath_agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            self._inject_gateway_headers(result.generations)
            return result
        finally:
            set_captured_response_headers({})

    async def _uipath_agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override in subclasses to provide the core (non-wrapped) async generate logic."""
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Generator[ChatGenerationChunk, None, None]:
        kwargs = self._strip_sampling(kwargs)
        set_captured_response_headers({})
        try:
            first = True
            for chunk in self._uipath_stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if first:
                    self._inject_gateway_headers([chunk])
                    first = False
                yield chunk
        finally:
            set_captured_response_headers({})

    def _uipath_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Generator[ChatGenerationChunk, None, None]:
        """Override in subclasses to provide the core (non-wrapped) stream logic."""
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        kwargs = self._strip_sampling(kwargs)
        set_captured_response_headers({})
        try:
            first = True
            async for chunk in self._uipath_astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if first:
                    self._inject_gateway_headers([chunk])
                    first = False
                yield chunk
        finally:
            set_captured_response_headers({})

    async def _uipath_astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        """Override in subclasses to provide the core (non-wrapped) async stream logic."""
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk

    def _inject_gateway_headers(self, generations: Sequence[ChatGeneration]) -> None:
        """Inject captured gateway headers into each generation's response_metadata."""
        if not self.captured_headers:
            return
        headers = get_captured_response_headers()
        if not headers:
            return
        for generation in generations:
            generation.message.response_metadata["headers"] = headers


class UiPathBaseEmbeddings(UiPathBaseLLMClient, Embeddings):
    pass
