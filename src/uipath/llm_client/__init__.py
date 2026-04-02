"""
UiPath LLM Client

A Python client for interacting with UiPath's LLM services. This package provides
the core HTTP client with authentication, retry logic, and request handling.

For framework-specific integrations, see:
    - uipath_langchain_client: LangChain-compatible models
    - uipath_llamaindex_client: LlamaIndex-compatible models

Quick Start:
    >>> from uipath.llm_client import UiPathHttpxClient
    >>> from uipath.llm_client.settings import get_default_client_settings, UiPathAPIConfig
    >>> from uipath.llm_client.settings.constants import ApiType, RoutingMode
    >>>
    >>> settings = get_default_client_settings()
    >>> api_config = UiPathAPIConfig(
    ...     api_type=ApiType.COMPLETIONS,
    ...     routing_mode=RoutingMode.PASSTHROUGH,
    ...     vendor_type="openai",
    ... )
    >>> client = UiPathHttpxClient(
    ...     model_name="gpt-4o-2024-11-20",
    ...     api_config=api_config,
    ...     base_url=settings.build_base_url(model_name="gpt-4o-2024-11-20", api_config=api_config),
    ...     auth=settings.build_auth_pipeline(),
    ... )
"""

from uipath.llm_client.__version__ import __version__
from uipath.llm_client.httpx_client import (
    UiPathHttpxAsyncClient,
    UiPathHttpxClient,
)
from uipath.llm_client.settings import (
    LLMGatewaySettings,
    PlatformSettings,
    get_default_client_settings,
)
from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathAuthenticationError,
    UiPathBadRequestError,
    UiPathConflictError,
    UiPathGatewayTimeoutError,
    UiPathInternalServerError,
    UiPathNotFoundError,
    UiPathPermissionDeniedError,
    UiPathRateLimitError,
    UiPathRequestTooLargeError,
    UiPathServiceUnavailableError,
    UiPathTooManyRequestsError,
    UiPathUnprocessableEntityError,
)
from uipath.llm_client.utils.retry import RetryConfig

__all__ = [
    "__version__",
    # Settings
    "get_default_client_settings",
    "PlatformSettings",
    "LLMGatewaySettings",
    # HTTPX clients
    "UiPathHttpxClient",
    "UiPathHttpxAsyncClient",
    # Retry
    "RetryConfig",
    # Exceptions
    "UiPathAPIError",
    "UiPathAuthenticationError",
    "UiPathBadRequestError",
    "UiPathConflictError",
    "UiPathGatewayTimeoutError",
    "UiPathInternalServerError",
    "UiPathNotFoundError",
    "UiPathPermissionDeniedError",
    "UiPathRateLimitError",
    "UiPathRequestTooLargeError",
    "UiPathServiceUnavailableError",
    "UiPathTooManyRequestsError",
    "UiPathUnprocessableEntityError",
]
