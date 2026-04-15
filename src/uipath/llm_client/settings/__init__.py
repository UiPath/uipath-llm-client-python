"""
UiPath LLM Client Settings Module

This module provides configuration settings for connecting to UiPath's LLM services.
It supports two backends:

1. Platform (default): Uses UiPath's Platform infrastructure (AgentHub/Orchestrator)
   with automatic CLI-based authentication. The EndpointManager transparently selects
   between AgentHub and Orchestrator based on service availability.

2. LLMGateway: Uses UiPath's LLM Gateway with S2S (server-to-server)
   authentication. Best for production deployments.

The backend is selected via:
- The `backend` parameter in `get_default_client_settings()`
- The `UIPATH_LLM_SERVICE` environment variable
- Defaults to "agenthub" if neither is specified

Example:
    >>> from uipath.llm_client.settings import get_default_client_settings
    >>>
    >>> # Use default (Platform - AgentHub/Orchestrator)
    >>> settings = get_default_client_settings()
    >>>
    >>> # Explicitly use LLMGateway
    >>> settings = get_default_client_settings(backend="llmgateway")
"""

import os
from typing import Literal

from uipath.llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
from uipath.llm_client.settings.constants import (
    ApiFlavor,
    ApiType,
    ByomApiFlavor,
    RoutingMode,
    VendorType,
)
from uipath.llm_client.settings.llmgateway import LLMGatewaySettings
from uipath.llm_client.settings.platform import PlatformSettings

# Environment variable to determine which backend to use
UIPATH_LLM_SERVICE_ENV = "UIPATH_LLM_SERVICE"

# Type alias for valid backend values
BackendType = Literal["agenthub", "orchestrator", "llmgateway"]


def get_default_client_settings(
    backend: BackendType | None = None,
) -> UiPathBaseSettings:
    """Factory function to create the appropriate client settings based on configuration.

    The backend is determined in the following order:
    1. Explicit `backend` parameter if provided
    2. UIPATH_LLM_SERVICE environment variable if set
    3. Default to "agenthub"

    Args:
        backend: Explicitly specify the backend to use ("agenthub", "orchestrator" or "llmgateway").
            Both "agenthub" and "orchestrator" use UiPath Platform with AgentHub/Orchestrator.

    Returns:
        UiPathBaseSettings: The appropriate settings instance for the selected backend

    Raises:
        ValueError: If an invalid backend type is specified

    Examples:
        >>> settings = get_default_client_settings()  # Uses env var or defaults to platform
        >>> settings = get_default_client_settings("llmgateway")  # Explicitly use llmgateway
    """
    if backend is None:
        backend = os.getenv(UIPATH_LLM_SERVICE_ENV, "agenthub").lower()  # type: ignore[assignment]

    match backend:
        case "orchestrator" | "agenthub":
            return PlatformSettings()
        case "llmgateway":
            return LLMGatewaySettings()
        case _:
            raise ValueError(
                f"Invalid backend type: {backend}. Must be 'orchestrator', 'agenthub', or 'llmgateway'"
            )


__all__ = [
    # Factory
    "get_default_client_settings",
    # Base classes
    "UiPathAPIConfig",
    "UiPathBaseSettings",
    # Backend-specific settings
    "PlatformSettings",
    "LLMGatewaySettings",
    # Constants
    "UIPATH_LLM_SERVICE_ENV",
    "BackendType",
    # Enums
    "ApiType",
    "RoutingMode",
    "VendorType",
    "ApiFlavor",
    "ByomApiFlavor",
]
