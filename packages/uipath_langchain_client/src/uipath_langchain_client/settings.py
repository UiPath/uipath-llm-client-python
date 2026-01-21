"""
Settings re-exports for UiPath LangChain Client.

This module re-exports the settings classes from uipath_llm_client for convenience,
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

from uipath_llm_client.settings import (
    AgentHubSettings,
    LLMGatewaySettings,
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)

__all__ = [
    "get_default_client_settings",
    "LLMGatewaySettings",
    "AgentHubSettings",
    "UiPathAPIConfig",
    "UiPathBaseSettings",
]
