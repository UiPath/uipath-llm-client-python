"""Normalized (provider-agnostic) LLM client for UiPath LLM services.

This module provides a lightweight HTTP client that speaks directly to UiPath's
normalized API endpoint, offering a consistent OpenAI-compatible interface across
all supported LLM providers (OpenAI, Google Gemini, Anthropic, etc.).

Unlike the vendor-specific clients (UiPathOpenAI, UiPathAnthropic, UiPathGoogle),
this client requires no vendor SDK — it makes HTTP requests directly using the
UiPath httpx transport layer.

Example:
    >>> from uipath.llm_client.clients.normalized import UiPathNormalizedClient
    >>>
    >>> client = UiPathNormalizedClient(model_name="gpt-4o-2024-11-20")
    >>>
    >>> # Chat completions
    >>> response = client.completions.create(
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> print(response["choices"][0]["message"]["content"])
    >>>
    >>> # Embeddings
    >>> embed_client = UiPathNormalizedClient(model_name="text-embedding-3-large")
    >>> result = embed_client.embeddings.create(input=["Hello world"])
    >>> print(result["data"][0]["embedding"])
"""

import json
import logging
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.utils.retry import RetryConfig


class NormalizedCompletions:
    """Chat completions sub-resource for :class:`UiPathNormalizedClient`.

    Provides sync/async chat completion and streaming methods backed by
    UiPath's normalized (provider-agnostic) completions endpoint.

    Accessed via ``client.completions``.
    """

    def __init__(
        self,
        model_name: str,
        sync_client: UiPathHttpxClient,
        async_client: UiPathHttpxAsyncClient,
    ) -> None:
        self._model_name = model_name
        self._sync_client = sync_client
        self._async_client = async_client

    def _build_body(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self._model_name, "messages": messages}
        body.update(kwargs)
        return body

    def create(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a synchronous chat completion request.

        Args:
            messages: List of message dicts in OpenAI format, e.g.
                ``[{"role": "user", "content": "Hello!"}]``.
            **kwargs: Additional parameters forwarded to the API:
                - temperature (float): Sampling temperature (0.0–2.0).
                - max_tokens (int): Maximum tokens in the response.
                - tools (list[dict]): Tool definitions for function calling.
                - tool_choice (str | dict): Tool selection strategy.
                - response_format (dict): Structured output format (JSON schema).
                - stop (list[str]): Stop sequences.

        Returns:
            The API response as a dict with OpenAI-compatible structure.
            Key fields: ``choices[0].message.content``, ``usage``.
        """
        body = self._build_body(messages, **kwargs)
        response = self._sync_client.post("", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def acreate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send an asynchronous chat completion request.

        Args:
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional parameters forwarded to the API (see ``create()``).

        Returns:
            The API response as a dict with OpenAI-compatible structure.
        """
        body = self._build_body(messages, **kwargs)
        response = await self._async_client.post("", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream a synchronous chat completion, yielding parsed SSE chunks.

        The ``X-UiPath-Streaming-Enabled: true`` header is set automatically by
        the underlying httpx client when streaming is requested.

        Args:
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional parameters forwarded to the API (see ``create()``).

        Yields:
            Parsed JSON dicts for each SSE chunk from the response stream.
        """
        body = self._build_body(messages, **kwargs)
        with self._sync_client.stream("POST", "", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def astream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream an asynchronous chat completion, yielding parsed SSE chunks.

        The ``X-UiPath-Streaming-Enabled: true`` header is set automatically by
        the underlying httpx async client when streaming is requested.

        Args:
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional parameters forwarded to the API (see ``create()``).

        Yields:
            Parsed JSON dicts for each SSE chunk from the response stream.
        """
        body = self._build_body(messages, **kwargs)
        async with self._async_client.stream("POST", "", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class NormalizedEmbeddings:
    """Embeddings sub-resource for :class:`UiPathNormalizedClient`.

    Provides sync/async embedding methods backed by UiPath's normalized
    (provider-agnostic) embeddings endpoint.

    Accessed via ``client.embeddings``.

    Response structure::

        {
            "data": [{"embedding": [0.1, 0.2, ...], "index": 0}],
            "usage": {"prompt_tokens": N, "total_tokens": N}
        }
    """

    def __init__(
        self,
        model_name: str,
        sync_client: UiPathHttpxClient,
        async_client: UiPathHttpxAsyncClient,
    ) -> None:
        self._model_name = model_name
        self._sync_client = sync_client
        self._async_client = async_client

    def _build_body(self, input: str | list[str], **kwargs: Any) -> dict[str, Any]:
        # The normalized embeddings endpoint resolves the model from routing headers;
        # the body only needs "input" (matching UiPathEmbeddings in the langchain client).
        # The API requires input to always be a list.
        body: dict[str, Any] = {"input": [input] if isinstance(input, str) else input}
        body.update(kwargs)
        return body

    def create(
        self,
        input: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate embeddings synchronously.

        Args:
            input: A single string or list of strings to embed.
            **kwargs: Additional parameters forwarded to the API
                (e.g., ``encoding_format``).

        Returns:
            The API response dict. Extract vectors via
            ``response["data"][i]["embedding"]``.
        """
        body = self._build_body(input, **kwargs)
        response = self._sync_client.post("", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def acreate(
        self,
        input: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate embeddings asynchronously.

        Args:
            input: A single string or list of strings to embed.
            **kwargs: Additional parameters forwarded to the API (see ``create()``).

        Returns:
            The API response dict. Extract vectors via
            ``response["data"][i]["embedding"]``.
        """
        body = self._build_body(input, **kwargs)
        response = await self._async_client.post("", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]


class UiPathNormalizedClient:
    """Provider-agnostic LLM client using UiPath's normalized API.

    Routes requests through UiPath's normalized endpoint, which provides a consistent
    OpenAI-compatible interface across all supported LLM providers (OpenAI, Google
    Gemini, Anthropic on Bedrock/Vertex, etc.).

    Unlike the vendor-specific clients (UiPathOpenAI, UiPathAnthropic, UiPathGoogle),
    this client does not require or wrap a vendor SDK. It communicates directly with
    the UiPath normalized HTTP endpoint using the shared httpx transport layer.

    Sub-resources:
        - ``completions``: Chat completion methods (``create``, ``acreate``, ``stream``,
          ``astream``).
        - ``embeddings``: Embedding methods (``create``, ``acreate``).

    Args:
        model_name: The model identifier (e.g., "gpt-4o-2024-11-20", "gemini-2.5-flash",
            "anthropic.claude-haiku-4-5-20251001-v1:0").
        byo_connection_id: Bring Your Own connection ID for custom model deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        default_headers: Additional headers included in every request.
        captured_headers: Case-insensitive response header name prefixes to capture.
            Captured headers are stored in a ContextVar and can be retrieved with
            ``get_captured_response_headers()``. Defaults to ``("x-uipath-",)``.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum number of retries for failed requests. Defaults to 0.
        retry_config: Custom retry configuration (backoff strategy, retryable errors).
        logger: Logger instance for request/response logging.

    Example:
        >>> client = UiPathNormalizedClient(model_name="gpt-4o-2024-11-20")
        >>> response = client.completions.create(
        ...     messages=[{"role": "user", "content": "What is 2+2?"}],
        ...     temperature=0.0,
        ... )
        >>> print(response["choices"][0]["message"]["content"])
        >>>
        >>> embed_client = UiPathNormalizedClient(model_name="text-embedding-3-large")
        >>> result = embed_client.embeddings.create(input="Hello world")
        >>> print(result["data"][0]["embedding"])
    """

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
        self.model_name = model_name
        self.byo_connection_id = byo_connection_id

        client_settings = client_settings or get_default_client_settings()

        auth = client_settings.build_auth_pipeline()

        def _make_httpx_clients(
            api_config: UiPathAPIConfig,
        ) -> tuple[UiPathHttpxClient, UiPathHttpxAsyncClient]:
            merged_headers = {
                **(default_headers or {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            }
            base_url = client_settings.build_base_url(model_name=model_name, api_config=api_config)
            sync_client = UiPathHttpxClient(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=api_config,
                timeout=timeout,
                max_retries=max_retries,
                retry_config=retry_config,
                captured_headers=captured_headers,
                base_url=base_url,
                headers=merged_headers,
                logger=logger,
                auth=auth,
            )
            async_client = UiPathHttpxAsyncClient(
                model_name=model_name,
                byo_connection_id=byo_connection_id,
                api_config=api_config,
                timeout=timeout,
                max_retries=max_retries,
                retry_config=retry_config,
                captured_headers=captured_headers,
                base_url=base_url,
                headers=merged_headers,
                logger=logger,
                auth=auth,
            )
            return sync_client, async_client

        completions_api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
            freeze_base_url=True,
        )
        completions_sync, completions_async = _make_httpx_clients(completions_api_config)
        self.completions = NormalizedCompletions(model_name, completions_sync, completions_async)

        embeddings_api_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.NORMALIZED,
            freeze_base_url=True,
        )
        embeddings_sync, embeddings_async = _make_httpx_clients(embeddings_api_config)
        self.embeddings = NormalizedEmbeddings(model_name, embeddings_sync, embeddings_async)
