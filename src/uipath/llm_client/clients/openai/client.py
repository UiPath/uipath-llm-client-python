import logging
from collections.abc import Mapping, Sequence

from uipath.llm_client.clients.openai.utils import OpenAIRequestHandler
from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import UiPathBaseSettings, get_default_client_settings
from uipath.llm_client.utils.retry import RetryConfig

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required to use UiPathOpenAIClient. "
        "Install it with: uv add uipath-llm-client[openai]"
    ) from e


class UiPathOpenAI(OpenAI):
    """OpenAI client routed through UiPath LLM Gateway.

    Wraps the standard OpenAI client to route requests through UiPath's
    LLM Gateway while preserving the full OpenAI SDK interface.

    Args:
        model_name: The OpenAI model name (e.g., "gpt-4o-2024-11-20").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        captured_headers: Response header prefixes to capture (case-insensitive).
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        _strict_response_validation: Enable strict response validation.
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
        _strict_response_validation: bool = False,
    ):
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            client_settings=client_settings,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            event_hooks={
                "request": [
                    OpenAIRequestHandler(
                        model_name, client_settings, byo_connection_id
                    ).fix_url_and_headers
                ]
            },
        )
        super().__init__(
            api_key="PLACEHOLDER",
            max_retries=0,
            _strict_response_validation=_strict_response_validation,
            http_client=httpx_client,
            base_url=str(httpx_client.base_url).rstrip("/"),
        )


class UiPathAsyncOpenAI(AsyncOpenAI):
    """Async OpenAI client routed through UiPath LLM Gateway.

    Wraps the standard AsyncOpenAI client to route requests through UiPath's
    LLM Gateway while preserving the full OpenAI SDK interface.

    Args:
        model_name: The OpenAI model name (e.g., "gpt-4o-2024-11-20").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        captured_headers: Response header prefixes to capture (case-insensitive).
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        _strict_response_validation: Enable strict response validation.
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
        _strict_response_validation: bool = False,
    ):
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            client_settings=client_settings,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            event_hooks={
                "request": [
                    OpenAIRequestHandler(
                        model_name, client_settings, byo_connection_id
                    ).fix_url_and_headers_async
                ]
            },
        )
        super().__init__(
            api_key="PLACEHOLDER",
            max_retries=0,
            _strict_response_validation=_strict_response_validation,
            http_client=httpx_client,
            base_url=str(httpx_client.base_url).rstrip("/"),
        )


class UiPathAzureOpenAI(AzureOpenAI):
    """Azure OpenAI client routed through UiPath LLM Gateway.

    Wraps the AzureOpenAI client to route requests through UiPath's
    LLM Gateway while preserving the full Azure OpenAI SDK interface.

    Args:
        model_name: The model name (e.g., "gpt-4o-2024-11-20").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        captured_headers: Response header prefixes to capture (case-insensitive).
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        _strict_response_validation: Enable strict response validation.
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
        _strict_response_validation: bool = False,
    ):
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            client_settings=client_settings,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            event_hooks={
                "request": [
                    OpenAIRequestHandler(
                        model_name, client_settings, byo_connection_id
                    ).fix_url_and_headers
                ]
            },
        )
        super().__init__(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            max_retries=0,
            _strict_response_validation=_strict_response_validation,
            http_client=httpx_client,
        )


class UiPathAsyncAzureOpenAI(AsyncAzureOpenAI):
    """Async Azure OpenAI client routed through UiPath LLM Gateway.

    Wraps the AsyncAzureOpenAI client to route requests through UiPath's
    LLM Gateway while preserving the full Azure OpenAI SDK interface.

    Args:
        model_name: The model name (e.g., "gpt-4o-2024-11-20").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        captured_headers: Response header prefixes to capture (case-insensitive).
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        _strict_response_validation: Enable strict response validation.
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
        _strict_response_validation: bool = False,
    ):
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            client_settings=client_settings,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            event_hooks={
                "request": [
                    OpenAIRequestHandler(
                        model_name, client_settings, byo_connection_id
                    ).fix_url_and_headers_async
                ]
            },
        )
        super().__init__(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            max_retries=0,
            _strict_response_validation=_strict_response_validation,
            http_client=httpx_client,
        )
