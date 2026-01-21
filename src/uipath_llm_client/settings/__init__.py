"""
UiPath LLM Client Settings Module

This module provides configuration settings for connecting to UiPath's LLM services.
It supports two backends:

1. AgentHub (default): Uses UiPath's AgentHub infrastructure with automatic
   CLI-based authentication. Best for development and interactive use.

2. LLMGateway: Uses UiPath's LLM Gateway with S2S (server-to-server)
   authentication. Best for production deployments.

The backend is selected via:
- The `backend` parameter in `get_default_client_settings()`
- The `UIPATH_LLM_BACKEND` environment variable
- Defaults to "agenthub" if neither is specified

Example:
    >>> from uipath_llm_client.settings import get_default_client_settings
    >>>
    >>> # Use default (AgentHub)
    >>> settings = get_default_client_settings()
    >>>
    >>> # Explicitly use LLMGateway
    >>> settings = get_default_client_settings(backend="llmgateway")
"""

import os
from typing import Literal

from uipath_llm_client.settings.agenthub import AgentHubSettings
from uipath_llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
from uipath_llm_client.settings.llmgateway import LLMGatewaySettings

# Environment variable to determine which backend to use
UIPATH_LLM_BACKEND_ENV = "UIPATH_LLM_BACKEND"

# Type alias for valid backend values
BackendType = Literal["agenthub", "llmgateway"]


def get_default_client_settings(
    backend: BackendType | None = None,
) -> UiPathBaseSettings:
    """Factory function to create the appropriate client settings based on configuration.

    The backend is determined in the following order:
    1. Explicit `backend` parameter if provided
    2. UIPATH_LLM_BACKEND environment variable if set
    3. Default to "agenthub"

    Args:
        backend: Explicitly specify the backend to use ("agenthub" or "llmgateway")

    Returns:
        UiPathBaseSettings: The appropriate settings instance for the selected backend

    Raises:
        ValueError: If an invalid backend type is specified

    Examples:
        >>> settings = get_default_client_settings()  # Uses env var or defaults to agenthub
        >>> settings = get_default_client_settings("llmgateway")  # Explicitly use llmgateway
    """
    if backend is None:
        backend = os.getenv(UIPATH_LLM_BACKEND_ENV, "agenthub").lower()  # type: ignore[assignment]

    match backend:
        case "agenthub":
            return AgentHubSettings()
        case "llmgateway":
            return LLMGatewaySettings()
        case _:
            raise ValueError(f"Invalid backend type: {backend}. Must be 'agenthub' or 'llmgateway'")


__all__ = [
    # Factory
    "get_default_client_settings",
    # Base classes
    "UiPathAPIConfig",
    "UiPathBaseSettings",
    # Backend-specific settings
    "AgentHubSettings",
    "LLMGatewaySettings",
    # Constants
    "UIPATH_LLM_BACKEND_ENV",
    "BackendType",
]
