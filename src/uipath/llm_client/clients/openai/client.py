import logging
from collections.abc import Mapping, Sequence

from uipath.llm_client.clients.openai.utils import OpenAIRequestHandler
from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import get_default_client_settings
from uipath.llm_client.settings.base import UiPathBaseSettings
from uipath.llm_client.utils.retry import RetryConfig

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required to use UiPathOpenAIClient. "
        "Install it with: uv add uipath-llm-client[openai]"
    ) from e


class UiPathOpenAI(OpenAI):
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
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
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
            http_client=httpx_client,
            base_url=str(httpx_client.base_url).rstrip("/"),
        )


class UiPathAsyncOpenAI(AsyncOpenAI):
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
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
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
            http_client=httpx_client,
            base_url=str(httpx_client.base_url).rstrip("/"),
        )


class UiPathAzureOpenAI(AzureOpenAI):
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
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
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
            http_client=httpx_client,
        )


class UiPathAsyncAzureOpenAI(AsyncAzureOpenAI):
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
        client_settings = client_settings or get_default_client_settings()
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            timeout=timeout,
            max_retries=max_retries,
            headers=default_headers,
            captured_headers=captured_headers,
            retry_config=retry_config,
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
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
            http_client=httpx_client,
        )
