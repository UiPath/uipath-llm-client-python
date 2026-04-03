"""
Base Settings Module for UiPath LLM Client

This module defines the abstract base classes and data models for UiPath API settings.
Concrete implementations are provided in the `platform` and `llmgateway` submodules.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Self

from httpx import Auth
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType


class UiPathAPIConfig(BaseModel):
    """Configuration for UiPath API request routing.

    This model defines how requests are routed to the appropriate API endpoint.

    Attributes:
        api_type: The type of API call (e.g., ApiType.COMPLETIONS, ApiType.EMBEDDINGS).
        routing_mode: API routing mode - RoutingMode.PASSTHROUGH for vendor-specific APIs or
            RoutingMode.NORMALIZED for UiPath's provider-agnostic API.
        vendor_type: The LLM vendor (e.g., VendorType.OPENAI, VendorType.VERTEXAI).
            Required when routing_mode is PASSTHROUGH.
        api_flavor: Vendor-specific API flavor (e.g., ApiFlavor.CHAT_COMPLETIONS).
        api_version: Vendor-specific API version (e.g., "2025-03-01-preview").
        freeze_base_url: If True, prevents httpx from modifying the base URL.
            Used when the URL must remain exactly as configured.

    Example:
        >>> # For OpenAI passthrough
        >>> settings = UiPathAPIConfig(
        ...     api_type=ApiType.COMPLETIONS,
        ...     routing_mode=RoutingMode.PASSTHROUGH,
        ...     vendor_type=VendorType.OPENAI,
        ... )
        >>>
        >>> # For normalized API
        >>> settings = UiPathAPIConfig(
        ...     api_type=ApiType.COMPLETIONS,
        ...     routing_mode=RoutingMode.NORMALIZED,
        ... )
    """

    api_type: ApiType | str | None = None
    api_flavor: ApiFlavor | str | None = None
    api_version: str | None = None
    vendor_type: VendorType | str | None = None
    routing_mode: RoutingMode | str | None = None
    freeze_base_url: bool = False

    @model_validator(mode="after")
    def validate_api_config(self) -> Self:
        """Validate that vendor_type is provided for passthrough mode."""
        if self.routing_mode == RoutingMode.PASSTHROUGH:
            if self.vendor_type is None:
                raise ValueError("vendor_type required when routing_mode='passthrough'")
        return self


class UiPathBaseSettings(BaseSettings, ABC):
    """Abstract base class for UiPath client settings.

    This class defines the interface that all backend-specific settings must implement.
    Subclasses (PlatformSettings, LLMGatewaySettings) provide
    concrete implementations for their respective backends.

    The settings are loaded from environment variables using pydantic-settings,
    with validation aliases allowing flexible naming conventions.
    """

    model_config = SettingsConfigDict(
        validate_by_alias=True,
        populate_by_name=True,
        extra="allow",
    )

    @abstractmethod
    def build_base_url(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> str:
        """Build the base URL for API requests.

        Args:
            model_name: The name of the model being accessed.
            api_config: API routing configuration.

        Returns:
            The fully-qualified base URL for the API endpoint.
        """
        ...

    @abstractmethod
    def build_auth_headers(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> Mapping[str, str]:
        """Build authentication and routing headers for API requests.

        Args:
            model_name: The name of the model being accessed.
            api_config: API routing configuration.

        Returns:
            A mapping of header names to values.
        """
        ...

    @abstractmethod
    def build_auth_pipeline(
        self,
    ) -> Auth:
        """Build an httpx Auth pipeline for request authentication.

        Subclasses must implement this method to provide backend-specific
        authentication handling.

        Returns:
            An httpx.Auth instance that handles authentication flow,
            including automatic token refresh on 401 responses.
        """
        ...

    @abstractmethod
    def get_available_models(
        self,
    ) -> list[dict[str, Any]]:
        """Get the list of available models from the backend.

        Subclasses must implement this method to query the backend's
        model discovery endpoint.

        Returns:
            A list of dictionaries containing model information.
        """
        ...

    @abstractmethod
    def validate_byo_model(self, model_info: dict[str, Any]) -> None:
        """Validate that the model is a BYOM model."""
        ...
