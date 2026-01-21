"""
LLM Gateway Settings Module.

This module provides configuration and authentication for UiPath LLM Gateway.
"""

from typing import override

from httpx import Auth

from uipath_llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth
from uipath_llm_client.settings.llmgateway.settings import LLMGatewayBaseSettings


class LLMGatewaySettings(LLMGatewayBaseSettings):
    """Configuration settings for LLM Gateway client requests.

    These settings control routing, authentication, and tracking for requests to LLM Gateway.

    Attributes:
        base_url: Base URL of the LLM Gateway (required)
        org_id: Organization ID for request routing (required)
        tenant_id: Tenant ID for request routing (required)
        requesting_product: Product name making the request (for tracking) (required)
        requesting_feature: Feature name making the request (for tracking) (required)
        user_id: User ID for tracking and billing (optional)
        action_id: Action ID for tracking (optional)
        additional_headers: Additional custom headers to include in requests (optional)
    """

    @override
    def build_auth_pipeline(self) -> Auth:
        """Build an httpx Auth pipeline for LLM Gateway authentication.

        Returns:
            An LLMGatewayS2SAuth instance that handles Bearer token authentication
            with automatic refresh on 401 responses.
        """
        return LLMGatewayS2SAuth(settings=self)


__all__ = [
    "LLMGatewaySettings",
    "LLMGatewayS2SAuth",
]
