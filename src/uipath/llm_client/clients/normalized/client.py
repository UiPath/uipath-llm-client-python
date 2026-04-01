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
    >>> # Structured output with Pydantic
    >>> from pydantic import BaseModel
    >>> class Capital(BaseModel):
    ...     capital: str
    ...     country: str
    >>> response = client.completions.create(
    ...     messages=[{"role": "user", "content": "Capital of France?"}],
    ...     output_format=Capital,
    ... )
    >>>
    >>> # Embeddings
    >>> embed_client = UiPathNormalizedClient(model_name="text-embedding-3-large")
    >>> result = embed_client.embeddings.create(input=["Hello world"])
    >>> print(result["data"][0]["embedding"])
"""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.utils.retry import RetryConfig

# Type alias for tool definitions accepted by the normalized API.
# Each tool can be:
#   - A dict in the flat OpenAI function format: {"name": ..., "description": ..., "parameters": ...}
#   - A Pydantic BaseModel subclass (auto-converted to the flat format)
#   - A callable with type annotations (auto-converted via docstring + signature)
ToolType = dict[str, Any] | type[BaseModel] | Callable[..., Any]

# Type alias for structured output format.
# Can be:
#   - A Pydantic BaseModel subclass (auto-converted to json_schema response_format)
#   - A dict (passed through as-is to the API's response_format field)
OutputFormatType = type[BaseModel] | dict[str, Any]


def _pydantic_to_tool(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model class to the flat normalized API tool format."""
    schema = model.model_json_schema()
    # Remove pydantic-internal keys that aren't part of JSON Schema
    schema.pop("title", None)
    return {
        "name": model.__name__,
        "description": model.__doc__ or model.__name__,
        "parameters": schema,
    }


def _callable_to_tool(func: Callable[..., Any]) -> dict[str, Any]:
    """Convert a callable to the flat normalized API tool format using its signature."""
    import inspect

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        prop: dict[str, Any] = {}
        if param.annotation is not inspect.Parameter.empty:
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
            prop["type"] = type_map.get(param.annotation, "string")
        else:
            prop["type"] = "string"
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": func.__doc__ or func.__name__,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _resolve_tool(tool: ToolType) -> dict[str, Any]:
    """Convert a tool definition to the flat normalized API format."""
    if isinstance(tool, dict):
        return tool
    if isinstance(tool, type) and issubclass(tool, BaseModel):
        return _pydantic_to_tool(tool)
    if callable(tool):
        return _callable_to_tool(tool)
    raise TypeError(f"Unsupported tool type: {type(tool)}")


def _make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add ``additionalProperties: false`` to all object schemas.

    Required by the ``strict: true`` mode of the normalized API's ``json_schema``
    response format.
    """
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for value in schema.values():
        if isinstance(value, dict):
            _make_strict_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _make_strict_schema(item)
    return schema


def _resolve_output_format(output_format: OutputFormatType) -> dict[str, Any]:
    """Convert an output format spec to the API's response_format field."""
    if isinstance(output_format, dict):
        return output_format
    if isinstance(output_format, type) and issubclass(output_format, BaseModel):
        schema = output_format.model_json_schema()
        schema.pop("title", None)
        _make_strict_schema(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output_format.__name__,
                "strict": True,
                "schema": schema,
            },
        }
    raise TypeError(f"Unsupported output_format type: {type(output_format)}")


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

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolType] | None = None,
        output_format: OutputFormatType | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self._model_name, "messages": messages}

        if tools is not None:
            body["tools"] = [_resolve_tool(t) for t in tools]
        if output_format is not None:
            body["response_format"] = _resolve_output_format(output_format)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if n is not None:
            body["n"] = n
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty

        body.update(kwargs)
        return body

    def create(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolType] | None = None,
        output_format: OutputFormatType | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a synchronous chat completion request.

        Args:
            messages: List of message dicts in OpenAI format, e.g.
                ``[{"role": "user", "content": "Hello!"}]``.
            tools: Tool definitions for function calling. Each element can be:
                - A dict in flat format: ``{"name": ..., "description": ..., "parameters": ...}``
                - A Pydantic ``BaseModel`` subclass (auto-converted)
                - A callable with type annotations (auto-converted)
            output_format: Structured output format. Can be:
                - A Pydantic ``BaseModel`` subclass (auto-converted to ``json_schema``)
                - A dict passed as-is to ``response_format``
            tool_choice: Tool selection strategy (e.g., ``"auto"``, ``"required"``,
                or ``{"type": "tool", "name": "..."}``).
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling probability mass.
            stop: Stop sequence(s) to end generation.
            n: Number of completions to generate.
            presence_penalty: Penalty for repeated tokens (-2.0 to 2.0).
            frequency_penalty: Penalty based on token frequency (-2.0 to 2.0).
            **kwargs: Additional parameters forwarded to the API.

        Returns:
            The API response as a dict with OpenAI-compatible structure.
        """
        body = self._build_body(
            messages,
            tools=tools,
            output_format=output_format,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **kwargs,
        )
        response = self._sync_client.post("", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def acreate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolType] | None = None,
        output_format: OutputFormatType | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send an asynchronous chat completion request.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Tool definitions (see ``create()``).
            output_format: Structured output format (see ``create()``).
            tool_choice: Tool selection strategy.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling probability mass.
            stop: Stop sequence(s).
            n: Number of completions.
            presence_penalty: Penalty for repeated tokens.
            frequency_penalty: Penalty based on token frequency.
            **kwargs: Additional parameters forwarded to the API.

        Returns:
            The API response as a dict with OpenAI-compatible structure.
        """
        body = self._build_body(
            messages,
            tools=tools,
            output_format=output_format,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **kwargs,
        )
        response = await self._async_client.post("", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolType] | None = None,
        output_format: OutputFormatType | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream a synchronous chat completion, yielding parsed SSE chunks.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Tool definitions (see ``create()``).
            output_format: Structured output format (see ``create()``).
            tool_choice: Tool selection strategy.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling probability mass.
            stop: Stop sequence(s).
            n: Number of completions.
            presence_penalty: Penalty for repeated tokens.
            frequency_penalty: Penalty based on token frequency.
            **kwargs: Additional parameters forwarded to the API.

        Yields:
            Parsed JSON dicts for each SSE chunk from the response stream.
        """
        body = self._build_body(
            messages,
            tools=tools,
            output_format=output_format,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **kwargs,
        )
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
        *,
        tools: list[ToolType] | None = None,
        output_format: OutputFormatType | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream an asynchronous chat completion, yielding parsed SSE chunks.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Tool definitions (see ``create()``).
            output_format: Structured output format (see ``create()``).
            tool_choice: Tool selection strategy.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling probability mass.
            stop: Stop sequence(s).
            n: Number of completions.
            presence_penalty: Penalty for repeated tokens.
            frequency_penalty: Penalty based on token frequency.
            **kwargs: Additional parameters forwarded to the API.

        Yields:
            Parsed JSON dicts for each SSE chunk from the response stream.
        """
        body = self._build_body(
            messages,
            tools=tools,
            output_format=output_format,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **kwargs,
        )
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

    def _build_body(
        self,
        input: str | list[str],
        *,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # The normalized embeddings endpoint resolves the model from routing headers;
        # the body only needs "input" (matching UiPathEmbeddings in the langchain client).
        # The API requires input to always be a list.
        body: dict[str, Any] = {"input": [input] if isinstance(input, str) else input}
        if encoding_format is not None:
            body["encoding_format"] = encoding_format
        if dimensions is not None:
            body["dimensions"] = dimensions
        body.update(kwargs)
        return body

    def create(
        self,
        input: str | list[str],
        *,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate embeddings synchronously.

        Args:
            input: A single string or list of strings to embed.
            encoding_format: The format of the returned embeddings
                (e.g., ``"float"``, ``"base64"``).
            dimensions: The number of dimensions for the output embeddings
                (only supported by some models).
            **kwargs: Additional parameters forwarded to the API.

        Returns:
            The API response dict. Extract vectors via
            ``response["data"][i]["embedding"]``.
        """
        body = self._build_body(
            input, encoding_format=encoding_format, dimensions=dimensions, **kwargs
        )
        response = self._sync_client.post("", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def acreate(
        self,
        input: str | list[str],
        *,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate embeddings asynchronously.

        Args:
            input: A single string or list of strings to embed.
            encoding_format: The format of the returned embeddings.
            dimensions: The number of dimensions for the output embeddings.
            **kwargs: Additional parameters forwarded to the API.

        Returns:
            The API response dict. Extract vectors via
            ``response["data"][i]["embedding"]``.
        """
        body = self._build_body(
            input, encoding_format=encoding_format, dimensions=dimensions, **kwargs
        )
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
        >>> # Structured output with Pydantic
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     result: int
        >>> response = client.completions.create(
        ...     messages=[{"role": "user", "content": "What is 2+2?"}],
        ...     output_format=Answer,
        ... )
        >>>
        >>> embed_client = UiPathNormalizedClient(model_name="text-embedding-3-large")
        >>> result = embed_client.embeddings.create(input="Hello world")
        >>> print(result["data"][0]["embedding"])
    """

    completions: NormalizedCompletions
    embeddings: NormalizedEmbeddings

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
