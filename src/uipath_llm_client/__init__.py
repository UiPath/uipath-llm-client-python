"""
UiPath LLM Client

A Python client for interacting with UiPath's LLM services. This package provides
the core HTTP client with authentication, retry logic, and request handling.

For framework-specific integrations, see:
    - uipath_langchain_client: LangChain-compatible models
    - uipath_llamaindex_client: LlamaIndex-compatible models

Quick Start:
    >>> from uipath_llm_client import UiPathBaseLLMClient, UiPathAPIConfig
    >>> from uipath_llm_client.settings import get_default_client_settings
    >>>
    >>> settings = get_default_client_settings()
    >>> client = UiPathBaseLLMClient(
    ...     model="gpt-4o-2024-11-20",
    ...     api_config=UiPathAPIConfig(
    ...         api_type="completions",
    ...         client_type="passthrough",
    ...         vendor_type="openai",
    ...     ),
    ...     client_settings=settings,
    ... )
    >>> response = client.uipath_request(request_body={...})
"""

from uipath_llm_client.__version__ import __version__
from uipath_llm_client.httpx_client import (
    UiPathHttpxAsyncClient,
    UiPathHttpxClient,
)
from uipath_llm_client.settings import (
    AgentHubSettings,
    LLMGatewaySettings,
    get_default_client_settings,
)
from uipath_llm_client.utils.exceptions import (
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
from uipath_llm_client.utils.retry import RetryConfig

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
