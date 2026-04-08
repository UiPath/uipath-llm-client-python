"""UiPath LiteLLM Client — provider-agnostic LLM access via LiteLLM.

Routes completion and embedding requests through UiPath's LLM Gateway
while using LiteLLM for provider-specific request/response formatting.

Example:
    >>> from uipath.llm_client.clients.litellm import UiPathLiteLLM
    >>>
    >>> client = UiPathLiteLLM(model_name="gpt-4o-2024-11-20")
    >>> response = client.completion(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... )
    >>> print(response.choices[0].message.content)
"""

import logging
import os
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, Literal, Union

from httpx import Request
from pydantic import BaseModel

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import UiPathBaseSettings, get_default_client_settings
from uipath.llm_client.settings.base import UiPathAPIConfig
from uipath.llm_client.settings.constants import (
    API_FLAVOR_TO_VENDOR_TYPE,
    ApiFlavor,
    ApiType,
    RoutingMode,
    VendorType,
)
from uipath.llm_client.utils.retry import RetryConfig

# Route OpenAI chat completions through base_llm_http_handler (accepts HTTPHandler)
# instead of the OpenAI SDK path. This allows us to inject our UiPath httpx client
# uniformly for all providers including OpenAI.
os.environ.setdefault("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")

try:
    import litellm
    from litellm import CustomStreamWrapper, EmbeddingResponse, ModelResponse
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
    from litellm.types.llms.anthropic import AnthropicThinkingParam
    from litellm.types.llms.openai import (
        ChatCompletionAudioParam,
        ChatCompletionModality,
        ChatCompletionPredictionContentParam,
        OpenAIWebSearchOptions,
    )
except ImportError as e:
    raise ImportError(
        "The 'litellm' extra is required to use UiPathLiteLLM. "
        "Install it with: uv add uipath-llm-client[litellm]"
    ) from e

# Skip Google OAuth for Vertex AI — the UiPath httpx client handles auth.
VertexBase._ensure_access_token = lambda self, *a, **kw: ("unused", "unused")  # type: ignore[assignment]

# Dummy AWS credentials injected for Bedrock so litellm takes the session-token
# auth path (no STS/boto3 HTTP calls). The UiPath httpx client handles auth.
_BEDROCK_PLACEHOLDER_KWARGS: dict[str, str] = {
    "aws_access_key_id": "PLACEHOLDER",
    "aws_secret_access_key": "PLACEHOLDER",
    "aws_session_token": "PLACEHOLDER",
    "aws_region_name": "us-east-1",
}

# ---------------------------------------------------------------------------
# VendorType / ApiFlavor → litellm custom_llm_provider
# ---------------------------------------------------------------------------

_VENDOR_TO_LITELLM: dict[str, str] = {
    "openai": "openai",
    "vertexai": "gemini",
    "awsbedrock": "bedrock",
    "azure": "azure",
    "anthropic": "anthropic",
}

# Providers where litellm's embedding() path expects an OpenAI SDK client.
# For these, we use "hosted_vllm" which routes through base_llm_http_handler
# and accepts HTTPHandler (same OpenAI-compatible format).
_EMBEDDING_PROVIDER_OVERRIDE: dict[str, str] = {
    "openai": "hosted_vllm",
    "azure": "hosted_vllm",
}

_FLAVOR_TO_LITELLM: dict[str, str] = {
    "chat-completions": "openai",
    "responses": "openai",
    "generate-content": "gemini",
    "converse": "bedrock",
    "invoke": "bedrock",
    "anthropic-claude": "vertex_ai",
}

_ANTHROPIC_FAMILY = "anthropicclaude"


def _drop_nones(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


class UiPathLiteLLM:
    """Provider-agnostic LLM client using LiteLLM for request formatting
    and UiPath's LLM Gateway for routing.

    LiteLLM handles translating the unified interface to provider-specific
    request formats (OpenAI, Anthropic, Vertex AI, Bedrock, etc.), while
    the UiPath httpx client handles authentication, URL routing, and retries.

    The client always discovers the model from the backend. Optional
    ``vendor_type`` and ``api_flavor`` parameters allow filtering or
    overriding the discovered values (same pattern as the LangChain factory).

    Args:
        model_name: The model name (e.g., "gpt-4o-2024-11-20").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        vendor_type: Filter/override vendor type from discovery.
        api_flavor: Override API flavor from discovery (e.g., ApiFlavor.RESPONSES,
            ApiFlavor.CONVERSE, ApiFlavor.INVOKE).
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        captured_headers: Response header prefixes to capture (case-insensitive).
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.

    Example:
        >>> client = UiPathLiteLLM(model_name="gpt-4o-2024-11-20")
        >>> response = client.completion(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        vendor_type: VendorType | str | None = None,
        api_flavor: ApiFlavor | str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        captured_headers: Sequence[str] = ("x-uipath-",),
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self._model_name = model_name
        self._client_settings = client_settings or get_default_client_settings()
        self._byo_connection_id = byo_connection_id
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers = default_headers
        self._captured_headers = captured_headers
        self._retry_config = retry_config
        self._logger = logger

        self._api_config, self._model_family = self._discover_and_build_api_config(
            vendor_type=vendor_type,
            api_flavor=api_flavor,
        )
        self._custom_llm_provider = self._resolve_llm_provider()
        self._embedding_llm_provider = _EMBEDDING_PROVIDER_OVERRIDE.get(
            self._custom_llm_provider, self._custom_llm_provider
        )
        self._litellm_model = self._resolve_litellm_model()

        # Extra kwargs injected into litellm calls to bypass provider auth.
        # Bedrock: dummy AWS creds → session-token path (no STS calls).
        self._extra_litellm_kwargs: dict[str, Any] = (
            _BEDROCK_PLACEHOLDER_KWARGS if self._custom_llm_provider == "bedrock" else {}
        )

    # ------------------------------------------------------------------
    # Discovery & provider resolution
    # ------------------------------------------------------------------

    def _discover_and_build_api_config(
        self,
        *,
        vendor_type: VendorType | str | None = None,
        api_flavor: ApiFlavor | str | None = None,
    ) -> tuple[UiPathAPIConfig, str | None]:
        """Discover model info from the backend and build api_config.

        User-supplied ``vendor_type`` filters models during discovery.
        User-supplied ``api_flavor`` overrides the discovered value.
        """
        available_models = self._client_settings.get_available_models()
        matching = [
            m for m in available_models if m["modelName"].lower() == self._model_name.lower()
        ]

        if vendor_type is not None:
            matching = [
                m for m in matching if m.get("vendor", "").lower() == str(vendor_type).lower()
            ]

        if not matching:
            raise ValueError(
                f"Model '{self._model_name}' not found. "
                f"Available: {[m['modelName'] for m in available_models]}"
            )
        model_info = matching[0]

        model_family: str | None = None
        raw_family = model_info.get("modelFamily", None)
        if raw_family is not None:
            model_family = raw_family.lower()

        discovered_vendor = model_info.get("vendor", None)
        discovered_flavor = model_info.get("apiFlavor", None)

        if discovered_vendor is None and discovered_flavor is not None:
            discovered_vendor = API_FLAVOR_TO_VENDOR_TYPE.get(discovered_flavor, None)
        if discovered_vendor is None:
            raise ValueError(f"Cannot determine vendor for model '{self._model_name}'")

        resolved_vendor = str(vendor_type or discovered_vendor).lower()
        resolved_flavor = str(api_flavor) if api_flavor is not None else discovered_flavor

        # OpenAI defaults to chat-completions when no flavor is discovered
        if resolved_flavor is None and resolved_vendor in ("openai", "azure"):
            resolved_flavor = ApiFlavor.CHAT_COMPLETIONS

        # Claude on Bedrock defaults to invoke
        if (
            resolved_flavor is None
            and resolved_vendor == "awsbedrock"
            and model_family == _ANTHROPIC_FAMILY
        ):
            resolved_flavor = ApiFlavor.INVOKE

        # Claude on Vertex defaults to anthropic-claude
        if (
            resolved_flavor is None
            and resolved_vendor == "vertexai"
            and model_family == _ANTHROPIC_FAMILY
        ):
            resolved_flavor = ApiFlavor.ANTHROPIC_CLAUDE

        api_config = UiPathAPIConfig(
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type=resolved_vendor,
            api_flavor=resolved_flavor,
            freeze_base_url=True,
        )
        return api_config, model_family

    def _resolve_llm_provider(self) -> str:
        """Map api_config + model_family to the litellm ``custom_llm_provider`` string.

        The model_family disambiguates cases where the same vendor hosts
        models from different providers (e.g. Claude on Vertex AI or Bedrock).
        """
        is_claude = self._model_family == _ANTHROPIC_FAMILY
        vendor = str(self._api_config.vendor_type or "openai")

        # Claude on Vertex AI → vertex_ai (uses VertexAIAnthropicConfig)
        if vendor == "vertexai" and is_claude:
            return "vertex_ai"

        # Claude on Bedrock → bedrock (litellm handles invoke/converse routing)
        if vendor == "awsbedrock" and is_claude:
            return "bedrock"

        # For everything else, check flavor first, then fall back to vendor
        if self._api_config.api_flavor:
            provider = _FLAVOR_TO_LITELLM.get(str(self._api_config.api_flavor))
            if provider:
                return provider
        return _VENDOR_TO_LITELLM.get(vendor, vendor)

    def _resolve_litellm_model(self) -> str:
        """Build the model name litellm expects, with route prefixes where needed."""
        model = self._model_name
        flavor = str(self._api_config.api_flavor) if self._api_config.api_flavor else None

        # OpenAI Responses API: prepend responses/ so litellm bridges to /v1/responses
        if flavor == "responses":
            model = f"responses/{model}"

        # Bedrock route prefixes (invoke/, converse/)
        if self._custom_llm_provider == "bedrock" and flavor in ("invoke", "converse"):
            model = f"{flavor}/{model}"

        return model

    # ------------------------------------------------------------------
    # LiteLLM HTTP handlers (lazily created)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_gemini_sse_hook(httpx_client: UiPathHttpxClient | UiPathHttpxAsyncClient) -> None:
        """Append an event hook that adds ``alt=sse`` for Gemini streaming requests.

        The UiPath LLM Gateway needs this query parameter to return SSE-formatted
        chunks instead of raw Gemini JSON arrays.
        """

        def _fix_url(request: Request) -> None:
            if request.headers.get("X-UiPath-Streaming-Enabled") == "true":
                request.url = request.url.copy_add_param("alt", "sse")

        async def _fix_url_async(request: Request) -> None:
            _fix_url(request)

        if isinstance(httpx_client, UiPathHttpxAsyncClient):
            httpx_client.event_hooks["request"].append(_fix_url_async)
        else:
            httpx_client.event_hooks["request"].append(_fix_url)

    def _build_client(
        self, api_type: ApiType, *, async_: bool = False
    ) -> HTTPHandler | AsyncHTTPHandler:
        """Build an HTTPHandler wrapping a UiPath httpx client for litellm."""
        api_config = UiPathAPIConfig(
            api_type=api_type,
            routing_mode=self._api_config.routing_mode,
            vendor_type=self._api_config.vendor_type,
            api_flavor=self._api_config.api_flavor,
            api_version=self._api_config.api_version,
            freeze_base_url=True,
        )
        httpx_kwargs: dict[str, Any] = dict(
            model_name=self._model_name,
            byo_connection_id=self._byo_connection_id,
            client_settings=self._client_settings,
            api_config=api_config,
            timeout=self._timeout,
            max_retries=self._max_retries,
            headers=self._default_headers,
            captured_headers=self._captured_headers,
            retry_config=self._retry_config,
            logger=self._logger,
        )
        is_gemini = self._custom_llm_provider == "gemini"
        if async_:
            handler = AsyncHTTPHandler()
            httpx_async = UiPathHttpxAsyncClient(**httpx_kwargs)  # type: ignore[arg-type]
            if is_gemini:
                self._add_gemini_sse_hook(httpx_async)
            handler.client = httpx_async
            return handler
        httpx_sync = UiPathHttpxClient(**httpx_kwargs)  # type: ignore[arg-type]
        if is_gemini:
            self._add_gemini_sse_hook(httpx_sync)
        return HTTPHandler(client=httpx_sync)  # type: ignore[arg-type]

    @cached_property
    def _completion_client(self) -> HTTPHandler:
        return self._build_client(ApiType.COMPLETIONS)  # type: ignore[return-value]

    @cached_property
    def _completion_async_client(self) -> AsyncHTTPHandler:
        return self._build_client(ApiType.COMPLETIONS, async_=True)  # type: ignore[return-value]

    @cached_property
    def _embedding_client(self) -> HTTPHandler:
        return self._build_client(ApiType.EMBEDDINGS)  # type: ignore[return-value]

    @cached_property
    def _embedding_async_client(self) -> AsyncHTTPHandler:
        return self._build_client(ApiType.EMBEDDINGS, async_=True)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def completion(
        self,
        *,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        stop: str | list[str] | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        modalities: list[ChatCompletionModality] | None = None,
        prediction: ChatCompletionPredictionContentParam | None = None,
        audio: ChatCompletionAudioParam | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, Any] | None = None,
        user: str | None = None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"] | None
        ) = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        seed: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        thinking: AnthropicThinkingParam | None = None,
        web_search_options: OpenAIWebSearchOptions | None = None,
        extra_headers: dict[str, str] | None = None,
        enable_json_schema_validation: bool | None = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        optional = _drop_nones(
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            modalities=modalities,
            prediction=prediction,
            audio=audio,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            thinking=thinking,
            web_search_options=web_search_options,
            extra_headers=extra_headers,
            enable_json_schema_validation=enable_json_schema_validation,
        )
        return litellm.completion(
            model=self._litellm_model,
            messages=messages,
            custom_llm_provider=self._custom_llm_provider,
            api_key="PLACEHOLDER",
            api_base=str(self._completion_client.client.base_url),
            client=self._completion_client,
            num_retries=0,
            max_retries=0,
            **self._extra_litellm_kwargs,
            **optional,
            **kwargs,
        )

    async def acompletion(
        self,
        *,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        stop: str | list[str] | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        modalities: list[ChatCompletionModality] | None = None,
        prediction: ChatCompletionPredictionContentParam | None = None,
        audio: ChatCompletionAudioParam | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, Any] | None = None,
        user: str | None = None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"] | None
        ) = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        seed: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        thinking: AnthropicThinkingParam | None = None,
        web_search_options: OpenAIWebSearchOptions | None = None,
        extra_headers: dict[str, str] | None = None,
        enable_json_schema_validation: bool | None = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        optional = _drop_nones(
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            modalities=modalities,
            prediction=prediction,
            audio=audio,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            thinking=thinking,
            web_search_options=web_search_options,
            extra_headers=extra_headers,
            enable_json_schema_validation=enable_json_schema_validation,
        )
        return await litellm.acompletion(
            model=self._litellm_model,
            messages=messages,
            custom_llm_provider=self._custom_llm_provider,
            api_key="PLACEHOLDER",
            api_base=str(self._completion_async_client.client.base_url),
            client=self._completion_async_client,
            num_retries=0,
            max_retries=0,
            **self._extra_litellm_kwargs,
            **optional,
            **kwargs,
        )

    def embedding(
        self,
        *,
        input: list[str] | str,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        return litellm.embedding(  # type: ignore[return-value]
            model=self._litellm_model,
            input=input,
            custom_llm_provider=self._embedding_llm_provider,
            api_key="PLACEHOLDER",
            api_base=str(self._embedding_client.client.base_url),
            client=self._embedding_client,
            max_retries=0,
            num_retries=0,
            dimensions=dimensions,
            encoding_format=encoding_format,
            timeout=int(timeout) if timeout is not None else 300,
            extra_headers=extra_headers,
            **kwargs,
        )

    async def aembedding(
        self,
        *,
        input: list[str] | str,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        return await litellm.aembedding(
            model=self._litellm_model,
            input=input,
            custom_llm_provider=self._embedding_llm_provider,
            api_key="PLACEHOLDER",
            api_base=str(self._embedding_async_client.client.base_url),
            client=self._embedding_async_client,
            max_retries=0,
            num_retries=0,
            dimensions=dimensions,
            encoding_format=encoding_format,
            timeout=int(timeout) if timeout is not None else 300,
            extra_headers=extra_headers,
            **kwargs,
        )
