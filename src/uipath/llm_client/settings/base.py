"""
Base Settings Module for UiPath LLM Client

This module defines the abstract base classes and data models for UiPath API settings.
Concrete implementations are provided in the `platform` and `llmgateway` submodules.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Self

from httpx import Auth
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from uipath.llm_client.settings.constants import (
    ApiFlavor,
    ApiType,
    ByomApiFlavor,
    RoutingMode,
    VendorType,
)


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

    _discovery_cache: ClassVar[dict[tuple[str, ...], list[dict[str, Any]]]] = {}

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
    def _discovery_cache_key(self) -> tuple[str, ...]:
        """Return a tuple that uniquely identifies the discovery endpoint.

        Used as the cache key for ``get_available_models``. Subclasses should
        include all settings properties that affect which models are returned
        (e.g. base URL, org/tenant IDs, requesting product).
        """
        ...

    @abstractmethod
    def _fetch_available_models(self) -> list[dict[str, Any]]:
        """Fetch the list of available models from the backend.

        Subclasses must implement this method to query the backend's
        model discovery endpoint.

        Returns:
            A list of dictionaries containing model information.
        """
        ...

    def get_available_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """Get the list of available models, with caching.

        Returns cached results if available. Pass ``refresh=True`` to
        bypass the cache and fetch fresh results.

        Args:
            refresh: If True, skip the cache and fetch from the backend.

        Returns:
            A list of dictionaries containing model information.
        """
        key = self._discovery_cache_key()

        if not refresh:
            cached = self._discovery_cache.get(key)
            if cached is not None:
                return cached

        models = self._fetch_available_models()
        self._discovery_cache[key] = models
        return models

    def validate_byo_model(self, model_info: dict[str, Any]) -> None:
        """Validate that the model is a BYOM model."""
        return

    def get_model_info(
        self,
        model_name: str,
        *,
        byo_connection_id: str | None = None,
        vendor_type: VendorType | str | None = None,
    ) -> dict[str, Any]:
        """Look up a model by name from the available models list.

        Filters the cached available models by name, and optionally by
        vendor type and BYO connection ID. When multiple matches exist and
        no ``byo_connection_id`` is given, UiPath-owned models are preferred.

        Args:
            model_name: The model name to search for (case-insensitive).
            byo_connection_id: Filter by BYO integration connection ID.
            vendor_type: Filter by vendor (e.g. ``VendorType.OPENAI``).

        Returns:
            The first matching model info dictionary.

        Raises:
            ValueError: If no matching model is found.
        """
        available_models = self.get_available_models()

        matching_models = [
            m for m in available_models if m["modelName"].lower() == model_name.lower()
        ]

        if vendor_type is not None:
            matching_models = [
                m
                for m in matching_models
                if m.get("vendor", "").lower() == str(vendor_type).lower()
            ]

        if byo_connection_id:
            matching_models = [
                m
                for m in matching_models
                if (byom_details := m.get("byomDetails"))
                and byom_details.get("integrationServiceConnectionId", "").lower()
                == byo_connection_id.lower()
            ]

        if not byo_connection_id and len(matching_models) > 1:
            matching_models = [
                m
                for m in matching_models
                if (
                    (m.get("modelSubscriptionType", "") == "UiPathOwned")
                    or (m.get("byomDetails") is None)
                )
            ]

        # When multiple OpenAI entries remain (both chat-completions and responses
        # flavors discovered), prefer the Responses API.
        if len(matching_models) > 1:
            vendor = str(matching_models[0].get("vendor", "")).lower()
            if vendor == VendorType.OPENAI:
                responses_matches = [
                    m
                    for m in matching_models
                    if m.get("apiFlavor") in (ApiFlavor.RESPONSES, ByomApiFlavor.OPENAI_RESPONSES)
                ]
                if responses_matches:
                    matching_models = responses_matches

        if not matching_models:
            raise ValueError(
                f"Model {model_name} not found. "
                f"Available models are: {[m['modelName'] for m in available_models]}"
            )

        model_info = matching_models[0]
        is_uipath_owned = model_info.get("modelSubscriptionType") == "UiPathOwned"
        if not is_uipath_owned:
            self.validate_byo_model(model_info)
        return model_info
