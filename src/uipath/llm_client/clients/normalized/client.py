"""UiPath Normalized Client.

A provider-agnostic LLM client that uses UiPath's normalized API to provide
a consistent interface across all supported providers (OpenAI, Google, Anthropic, etc.).

No optional dependencies required - works with the base uipath-llm-client package.

Example:
    >>> from uipath.llm_client.clients.normalized import UiPathNormalizedClient
    >>>
    >>> client = UiPathNormalizedClient(model_name="gpt-4o-2024-11-20")
    >>>
    >>> # Chat completion
    >>> response = client.completions.create(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... )
    >>> print(response.choices[0].message.content)
    >>>
    >>> # Streaming
    >>> for chunk in client.completions.stream(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... ):
    ...     print(chunk.choices[0].delta.content, end="")
    >>>
    >>> # Async
    >>> response = await client.completions.acreate(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... )
    >>>
    >>> # Structured output
    >>> from pydantic import BaseModel
    >>> class Answer(BaseModel):
    ...     text: str
    ...     confidence: float
    >>>
    >>> response = client.completions.create(
    ...     messages=[{"role": "user", "content": "What is 2+2?"}],
    ...     output_format=Answer,
    ... )
    >>> print(response.choices[0].message.parsed)  # Answer(text='4', confidence=1.0)
    >>>
    >>> # Embeddings
    >>> response = client.embeddings.create(input=["Hello world"])
    >>> print(len(response.data[0].embedding))
"""

import logging
from collections.abc import Mapping, Sequence
from functools import cached_property

from uipath.llm_client.clients.normalized.completions import Completions
from uipath.llm_client.clients.normalized.embeddings import Embeddings
from uipath.llm_client.clients.utils import build_httpx_async_client, build_httpx_client
from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import UiPathBaseSettings, get_default_client_settings
from uipath.llm_client.settings.base import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.utils.retry import RetryConfig


class UiPathNormalizedClient:
    """Provider-agnostic LLM client using UiPath's normalized API.

    Routes requests through UiPath's LLM Gateway using the normalized API,
    which provides a consistent interface across all supported LLM providers.
    No vendor-specific SDK dependencies are required.

    Namespaces:
        - ``completions``: ``create``, ``acreate``, ``stream``, ``astream``
        - ``embeddings``: ``create``, ``acreate``

    Args:
        model_name: The model name (e.g., "gpt-4o-2024-11-20", "gemini-2.5-flash").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        captured_headers: Response header prefixes to capture (case-insensitive).
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.

    Example:
        >>> client = UiPathNormalizedClient(model_name="gpt-4o-2024-11-20")
        >>> response = client.completions.create(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
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
        self._model_name = model_name
        self._byo_connection_id = byo_connection_id
        self._client_settings = client_settings or get_default_client_settings()
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers = default_headers
        self._captured_headers = captured_headers
        self._retry_config = retry_config
        self._logger = logger

        self._completions_api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
            freeze_base_url=True,
        )
        self._embeddings_api_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.NORMALIZED,
            freeze_base_url=True,
        )

    # ------------------------------------------------------------------
    # HTTP clients (lazily created)
    # ------------------------------------------------------------------

    def _build_sync(self, api_config: UiPathAPIConfig) -> UiPathHttpxClient:
        return build_httpx_client(
            model_name=self._model_name,
            byo_connection_id=self._byo_connection_id,
            client_settings=self._client_settings,
            timeout=self._timeout,
            max_retries=self._max_retries,
            default_headers=self._default_headers,
            captured_headers=self._captured_headers,
            retry_config=self._retry_config,
            logger=self._logger,
            api_config=api_config,
        )

    def _build_async(self, api_config: UiPathAPIConfig) -> UiPathHttpxAsyncClient:
        return build_httpx_async_client(
            model_name=self._model_name,
            byo_connection_id=self._byo_connection_id,
            client_settings=self._client_settings,
            timeout=self._timeout,
            max_retries=self._max_retries,
            default_headers=self._default_headers,
            captured_headers=self._captured_headers,
            retry_config=self._retry_config,
            logger=self._logger,
            api_config=api_config,
        )

    @cached_property
    def _sync_client(self) -> UiPathHttpxClient:
        return self._build_sync(self._completions_api_config)

    @cached_property
    def _async_client(self) -> UiPathHttpxAsyncClient:
        return self._build_async(self._completions_api_config)

    @cached_property
    def _embedding_sync_client(self) -> UiPathHttpxClient:
        return self._build_sync(self._embeddings_api_config)

    @cached_property
    def _embedding_async_client(self) -> UiPathHttpxAsyncClient:
        return self._build_async(self._embeddings_api_config)

    # ------------------------------------------------------------------
    # Public namespaces
    # ------------------------------------------------------------------

    @cached_property
    def completions(self) -> Completions:
        """Chat completions namespace (``create``, ``acreate``, ``stream``, ``astream``)."""
        return Completions(self)

    @cached_property
    def embeddings(self) -> Embeddings:
        """Embeddings namespace (``create``, ``acreate``)."""
        return Embeddings(self)
