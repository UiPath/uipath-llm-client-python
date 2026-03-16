"""
UiPath LLM Client

A Python client for interacting with UiPath's LLM services. This package provides
the core HTTP client with authentication, retry logic, and request handling.

For framework-specific integrations, see:
    - uipath_langchain_client: LangChain-compatible models
    - uipath_llamaindex_client: LlamaIndex-compatible models

Quick Start:
    >>> from uipath.llm_client import UiPathBaseLLMClient, UiPathAPIConfig
    >>> from uipath.llm_client.settings import get_default_client_settings
    >>>
    >>> settings = get_default_client_settings()
    >>> client = UiPathBaseLLMClient(
    ...     model="gpt-4o-2024-11-20",
    ...     api_config=UiPathAPIConfig(
    ...         api_type=ApiType.COMPLETIONS,
    ...         routing_mode=RoutingMode.PASSTHROUGH,
    ...         vendor_type="openai",
    ...     ),
    ...     settings=settings,
    ... )
    >>> response = client.uipath_request(request_body={...})
"""

from uipath.llm_client.__version__ import __version__
from uipath.llm_client.httpx_client import (
    UiPathHttpxAsyncClient,
    UiPathHttpxClient,
)
from uipath.llm_client.settings import (
    AgentHubSettings,
    LLMGatewaySettings,
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
    "AgentHubSettings",
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
