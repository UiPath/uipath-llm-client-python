"""UiPath LiteLLM client wrapper for routing through UiPath LLM Gateway.

This module provides a LiteLLM wrapper that routes requests through the UiPath
LLM Gateway while preserving LiteLLM's multi-provider interface.

The caller provides the api_config (api_type, routing_mode, vendor_type, api_flavor)
externally — LiteLLM acts purely as a client, with no automatic provider detection
from model names.

Example:
    >>> from uipath.llm_client.clients.litellm import UiPathLiteLLM
    >>> from uipath.llm_client.settings.base import UiPathAPIConfig
    >>> from uipath.llm_client.settings.constants import ApiType, RoutingMode, VendorType
    >>>
    >>> client = UiPathLiteLLM(
    ...     model_name="gpt-4o",
    ...     api_config=UiPathAPIConfig(
    ...         api_type=ApiType.COMPLETIONS,
    ...         routing_mode=RoutingMode.PASSTHROUGH,
    ...         vendor_type=VendorType.OPENAI,
    ...         freeze_base_url=True,
    ...     ),
    ... )
    >>> response = client.completion(messages=[{"role": "user", "content": "Hello!"}])
"""

import logging
from typing import Any, Literal

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.utils.retry import RetryConfig

try:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat import (
        ChatCompletionAudioParam,
        ChatCompletionPredictionContentParam,
    )
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required to use UiPathLiteLLM. "
        "Install it with: uv add uipath-llm-client[openai]"
    ) from e

try:
    import litellm as _litellm
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.types.llms.anthropic import AnthropicThinkingParam
    from litellm.types.llms.openai import OpenAIWebSearchOptions
    from litellm.types.utils import EmbeddingResponse, ModelResponse
except ImportError as e:
    raise ImportError(
        "The 'litellm' extra is required to use UiPathLiteLLM. "
        "Install it with: uv add uipath-llm-client[litellm]"
    ) from e

from pydantic import BaseModel


class UiPathLiteLLM:
    """LiteLLM client routed through UiPath LLM Gateway.

    Wraps litellm.completion/acompletion/embedding/aembedding to route requests
    through UiPath's LLM Gateway by injecting a UiPath-configured OpenAI client
    as the HTTP transport.

    The api_config must be provided by the caller — this client does not
    auto-detect providers from model names.

    Args:
        model_name: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022").
        api_config: API configuration (api_type, routing_mode, vendor_type, etc.).
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments (timeout, max_retries, default_headers).
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_config: UiPathAPIConfig,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        timeout = kwargs.pop("timeout", None)
        max_retries = kwargs.pop("max_retries", None)
        default_headers = kwargs.pop("default_headers", {})

        base_url = client_settings.build_base_url(model_name=model_name, api_config=api_config)
        headers = {
            **default_headers,
            **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
        }
        auth = client_settings.build_auth_pipeline()

        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=timeout,
            max_retries=max_retries,
            retry_config=retry_config,
            base_url=base_url,
            headers=headers,
            logger=logger,
            auth=auth,
        )
        httpx_async_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=timeout,
            max_retries=max_retries,
            retry_config=retry_config,
            base_url=base_url,
            headers=headers,
            logger=logger,
            auth=auth,
        )

        self._model_name = model_name
        self._openai_client = OpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=httpx_client,
            base_url=str(httpx_client.base_url).rstrip("/"),
        )
        self._async_openai_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=httpx_async_client,
            base_url=str(httpx_async_client.base_url).rstrip("/"),
        )

    @property
    def openai_client(self) -> OpenAI:
        """The underlying OpenAI client backed by UiPath's HTTP transport."""
        return self._openai_client

    @property
    def async_openai_client(self) -> AsyncOpenAI:
        """The underlying async OpenAI client backed by UiPath's HTTP transport."""
        return self._async_openai_client

    def completion(
        self,
        messages: list | None = None,
        *,
        model: str | None = None,
        timeout: float | str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stream: bool | None = None,
        stream_options: dict | None = None,
        stop: str | list[str] | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        modalities: list[Literal["text", "audio"]] | None = None,
        prediction: ChatCompletionPredictionContentParam | None = None,
        audio: ChatCompletionAudioParam | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict | None = None,
        user: str | None = None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"] | None
        ) = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
        response_format: dict | type[BaseModel] | None = None,
        seed: int | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        web_search_options: OpenAIWebSearchOptions | None = None,
        extra_headers: dict | None = None,
        safety_identifier: str | None = None,
        service_tier: str | None = None,
        functions: list | None = None,
        function_call: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        model_list: list | None = None,
        thinking: AnthropicThinkingParam | None = None,
        enable_json_schema_validation: bool | None = None,
        **kwargs: Any,
    ) -> ModelResponse | CustomStreamWrapper:
        """Call litellm.completion with UiPath routing.

        All parameters mirror litellm.completion(). The model defaults to the
        model_name provided at init. The UiPath OpenAI client is injected
        automatically for HTTP transport.
        """
        kwargs["client"] = self._openai_client
        return _litellm.completion(
            model=model or self._model_name,
            messages=messages or [],
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
            verbosity=verbosity,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            web_search_options=web_search_options,
            extra_headers=extra_headers,
            safety_identifier=safety_identifier,
            service_tier=service_tier,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            thinking=thinking,
            enable_json_schema_validation=enable_json_schema_validation,
            **kwargs,
        )

    async def acompletion(
        self,
        messages: list | None = None,
        *,
        model: str | None = None,
        timeout: float | int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stream: bool | None = None,
        stream_options: dict | None = None,
        stop: str | list[str] | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        modalities: list[Literal["text", "audio"]] | None = None,
        prediction: ChatCompletionPredictionContentParam | None = None,
        audio: ChatCompletionAudioParam | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict | None = None,
        user: str | None = None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"] | None
        ) = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
        response_format: dict | type[BaseModel] | None = None,
        seed: int | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        web_search_options: OpenAIWebSearchOptions | None = None,
        extra_headers: dict | None = None,
        safety_identifier: str | None = None,
        service_tier: str | None = None,
        functions: list | None = None,
        function_call: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        model_list: list | None = None,
        thinking: AnthropicThinkingParam | None = None,
        enable_json_schema_validation: bool | None = None,
        **kwargs: Any,
    ) -> ModelResponse | CustomStreamWrapper:
        """Call litellm.acompletion with UiPath routing.

        All parameters mirror litellm.acompletion(). The model defaults to the
        model_name provided at init. The UiPath async OpenAI client is injected
        automatically for HTTP transport.
        """
        kwargs["client"] = self._async_openai_client
        return await _litellm.acompletion(
            model=model or self._model_name,
            messages=messages or [],
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
            verbosity=verbosity,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            web_search_options=web_search_options,
            extra_headers=extra_headers,
            safety_identifier=safety_identifier,
            service_tier=service_tier,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            thinking=thinking,
            enable_json_schema_validation=enable_json_schema_validation,
            **kwargs,
        )

    def embedding(
        self,
        input: list | str = [],  # noqa: A002
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        timeout: int = 600,
        api_base: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        api_type: str | None = None,
        caching: bool = False,
        user: str | None = None,
        custom_llm_provider: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Call litellm.embedding with UiPath routing.

        All parameters mirror litellm.embedding(). The model defaults to the
        model_name provided at init. The UiPath OpenAI client is injected
        automatically for HTTP transport.
        """
        kwargs["client"] = self._openai_client
        return _litellm.embedding(  # type: ignore[no-any-return]
            model=model or self._model_name,
            input=input,
            dimensions=dimensions,
            encoding_format=encoding_format,
            timeout=timeout,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            api_type=api_type,
            caching=caching,
            user=user,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

    async def aembedding(
        self,
        input: list | str = [],  # noqa: A002
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        timeout: int = 600,
        api_base: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        api_type: str | None = None,
        caching: bool = False,
        user: str | None = None,
        custom_llm_provider: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Call litellm.aembedding with UiPath routing.

        All parameters mirror litellm.aembedding(). The model defaults to the
        model_name provided at init. The UiPath async OpenAI client is injected
        automatically for HTTP transport.
        """
        kwargs["client"] = self._async_openai_client
        return await _litellm.aembedding(
            model=model or self._model_name,
            input=input,
            dimensions=dimensions,
            encoding_format=encoding_format,
            timeout=timeout,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            api_type=api_type,
            caching=caching,
            user=user,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
