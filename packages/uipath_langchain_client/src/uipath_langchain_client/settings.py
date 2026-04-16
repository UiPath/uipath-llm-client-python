"""
Settings re-exports for UiPath LangChain Client.

This module re-exports the settings classes from uipath.llm_client for convenience,
allowing users to configure authentication without importing from the base package.

Example:
    >>> from uipath_langchain_client.settings import get_default_client_settings
    >>>
    >>> # Auto-detect backend from environment (defaults to AgentHub)
    >>> settings = get_default_client_settings()
    >>>
    >>> # Or explicitly use LLMGateway
    >>> from uipath_langchain_client.settings import LLMGatewaySettings
    >>> settings = LLMGatewaySettings()
"""

from uipath.llm_client.settings import (
    LLMGatewaySettings,
    PlatformSettings,
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import (
    API_FLAVOR_TO_VENDOR_TYPE,
    BYOM_TO_ROUTING_FLAVOR,
    ApiFlavor,
    ApiType,
    ByomApiFlavor,
    RoutingMode,
    VendorType,
)

__all__ = [
    "get_default_client_settings",
    "LLMGatewaySettings",
    "PlatformSettings",
    "UiPathAPIConfig",
    "UiPathBaseSettings",
    "ApiType",
    "RoutingMode",
    "ApiFlavor",
    "ByomApiFlavor",
    "VendorType",
    "API_FLAVOR_TO_VENDOR_TYPE",
    "BYOM_TO_ROUTING_FLAVOR",
]
