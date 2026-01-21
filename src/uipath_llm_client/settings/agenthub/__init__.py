"""
AgentHub Settings Module.

This module provides configuration and authentication for UiPath AgentHub.
"""

from typing import override

from httpx import Auth

from uipath_llm_client.settings.agenthub.auth import AgentHubAuth
from uipath_llm_client.settings.agenthub.settings import AgentHubBaseSettings


class AgentHubSettings(AgentHubBaseSettings):
    """Configuration settings for UiPath AgentHub client requests.

    These settings control routing, authentication, and tracking for requests to AgentHub.

    Attributes:
        environment: The UiPath environment ("cloud", "staging", "alpha").
        access_token: Access token for authentication.
        base_url: Base URL of the AgentHub API.
        tenant_id: Tenant ID for request routing.
        organization_id: Organization ID for request routing.
        client_id: Client ID for OAuth authentication.
        client_secret: Client secret for OAuth authentication.
        client_scope: OAuth scope for authentication.
        agenthub_config: AgentHub configuration for tracing.
        process_key: Process key for tracing.
        job_key: Job key for tracing.
    """

    @override
    def build_auth_pipeline(self) -> Auth:
        """Build an httpx Auth pipeline for AgentHub authentication.

        Returns:
            An AgentHubAuth instance that handles Bearer token authentication
            with automatic refresh on 401 responses.
        """
        return AgentHubAuth(settings=self)


__all__ = [
    "AgentHubSettings",
    "AgentHubAuth",
]
