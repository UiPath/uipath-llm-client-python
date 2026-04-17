"""Base settings for UiPath Platform (AgentHub/Orchestrator) client."""

from collections.abc import Mapping
from typing import Any, Self
from urllib.parse import quote

from pydantic import Field, SecretStr, model_validator
from typing_extensions import override
from uipath.platform import UiPath
from uipath.platform.common import EndpointManager
from uipath.platform.common._config import UiPathConfig
from uipath.platform.common.constants import (
    ENV_BASE_URL,
    ENV_FOLDER_KEY,
    ENV_JOB_KEY,
    ENV_ORGANIZATION_ID,
    ENV_PROCESS_KEY,
    ENV_TENANT_ID,
    ENV_UIPATH_ACCESS_TOKEN,
    ENV_UIPATH_TRACE_ID,
    HEADER_AGENTHUB_CONFIG,
    HEADER_FOLDER_KEY,
    HEADER_INTERNAL_ACCOUNT_ID,
    HEADER_INTERNAL_TENANT_ID,
    HEADER_JOB_KEY,
    HEADER_LICENSING_CONTEXT,
    HEADER_PROCESS_KEY,
    HEADER_TRACE_ID,
)

from uipath.llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.settings.platform.utils import is_token_expired, parse_access_token


class PlatformBaseSettings(UiPathBaseSettings):
    """Configuration settings for UiPath Platform (AgentHub/Orchestrator) client requests.

    These settings handle routing, authentication, and tracking for requests to UiPath's
    LLM services. The EndpointManager transparently selects between AgentHub and Orchestrator
    endpoints based on service availability and the UIPATH_LLM_SERVICE environment variable.

    Attributes:
        access_token: Access token for authentication.
        base_url: Base URL of the UiPath Platform API.
        tenant_id: Tenant ID for request routing.
        organization_id: Organization ID for request routing.
        client_id: Client ID for OAuth authentication.
        client_secret: Client secret for OAuth authentication.
        client_scope: OAuth scope for authentication.
        agenthub_config: AgentHub configuration for tracing.
        process_key: Process key for tracing.
        job_key: Job key for tracing.
    """

    # Authentication fields - retrieved from uipath auth as well
    access_token: SecretStr = Field(default=..., validation_alias=ENV_UIPATH_ACCESS_TOKEN)
    base_url: str = Field(default=..., validation_alias=ENV_BASE_URL)
    tenant_id: str = Field(default=..., validation_alias=ENV_TENANT_ID)
    organization_id: str = Field(default=..., validation_alias=ENV_ORGANIZATION_ID)

    # Credentials used for refreshing the access token
    client_id: str | None = Field(default=None)
    refresh_token: SecretStr | None = Field(default=None, validation_alias="UIPATH_REFRESH_TOKEN")

    # AgentHub configuration (used for discovery)
    agenthub_config: str | None = Field(
        default="agentsruntime", validation_alias="UIPATH_AGENTHUB_CONFIG"
    )

    # Tracing configuration
    process_key: str | None = Field(default=None, validation_alias=ENV_PROCESS_KEY)
    folder_key: str | None = Field(default=None, validation_alias=ENV_FOLDER_KEY)
    job_key: str | None = Field(default=None, validation_alias=ENV_JOB_KEY)
    trace_id: str | None = Field(default=None, validation_alias=ENV_UIPATH_TRACE_ID)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate access token expiry and extract client_id."""
        access_token = self.access_token.get_secret_value()
        if is_token_expired(access_token):
            raise ValueError(
                "Access token is expired. Try running `uipath auth` to refresh the token."
            )

        parsed_token_data = parse_access_token(access_token)
        self.client_id = parsed_token_data.get("client_id")
        return self

    @staticmethod
    def _format_endpoint(endpoint: str, **kwargs: str | None) -> str:
        """Format an endpoint template, stripping query params with None values."""
        # Remove query parameters whose values are None
        if "?" in endpoint:
            base, query = endpoint.split("?", 1)
            params = [
                p
                for p in query.split("&")
                if not any(f"{{{k}}}" in p for k, v in kwargs.items() if v is None)
            ]
            endpoint = f"{base}?{'&'.join(params)}" if params else base
        return endpoint.format(**{k: v for k, v in kwargs.items() if v is not None})

    @override
    def build_base_url(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> str:
        """Build the base URL for API requests."""
        if model_name is None:
            raise ValueError("model_name is required for PlatformBaseSettings.build_base_url")
        if api_config is None:
            raise ValueError("api_config is required for PlatformBaseSettings.build_base_url")
        if (
            api_config.routing_mode == RoutingMode.NORMALIZED
            and api_config.api_type == ApiType.COMPLETIONS
        ):
            url = f"{self.base_url}/{EndpointManager.get_normalized_endpoint()}"
        elif (
            api_config.routing_mode == RoutingMode.NORMALIZED
            and api_config.api_type == ApiType.EMBEDDINGS
        ):
            raise ValueError(
                "Normalized embeddings are not supported on UiPath Platform (AgentHub/Orchestrator). "
                "Use passthrough routing mode for embeddings instead."
            )
        elif (
            api_config.routing_mode == RoutingMode.PASSTHROUGH
            and api_config.api_type == ApiType.COMPLETIONS
        ):
            endpoint = EndpointManager.get_vendor_endpoint()
            url = f"{self.base_url}/{self._format_endpoint(endpoint, model=model_name, vendor=api_config.vendor_type, api_version=api_config.api_version)}"
        elif (
            api_config.routing_mode == RoutingMode.PASSTHROUGH
            and api_config.api_type == ApiType.EMBEDDINGS
        ):
            if api_config.vendor_type is not None and api_config.vendor_type != "openai":
                raise ValueError(
                    f"Platform embeddings endpoint only supports OpenAI-compatible models, "
                    f"got vendor_type='{api_config.vendor_type}'."
                )
            endpoint = EndpointManager.get_embeddings_endpoint()
            url = f"{self.base_url}/{self._format_endpoint(endpoint, model=model_name, api_version=api_config.api_version)}"
        else:
            raise ValueError(f"Invalid API configuration: {api_config}")
        return url

    @override
    def build_auth_headers(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> Mapping[str, str]:
        """Build authentication and routing headers for API requests.

        Mirrors the platform-wide header convention (see
        ``uipath.platform.common.constants``): routing headers come from the
        configured org/tenant, tracing headers come from pydantic fields
        (which pull from env vars), and licensing context is read dynamically
        from ``UiPathConfig`` at call time so updates are picked up without
        rebuilding settings.
        """
        headers: dict[str, str] = {
            HEADER_INTERNAL_ACCOUNT_ID: self.organization_id,
            HEADER_INTERNAL_TENANT_ID: self.tenant_id,
        }
        if self.agenthub_config:
            headers[HEADER_AGENTHUB_CONFIG] = self.agenthub_config
        if self.process_key:
            headers[HEADER_PROCESS_KEY] = quote(self.process_key, safe="")
        if self.folder_key:
            headers[HEADER_FOLDER_KEY] = self.folder_key
        if self.job_key:
            headers[HEADER_JOB_KEY] = self.job_key
        if self.trace_id:
            headers[HEADER_TRACE_ID] = self.trace_id
        if licensing_context := UiPathConfig.licensing_context:
            headers[HEADER_LICENSING_CONTEXT] = licensing_context
        return headers

    @override
    def _discovery_cache_key(self) -> tuple[str, ...]:
        return (self.base_url, self.organization_id, self.tenant_id)

    @override
    def _fetch_available_models(self) -> list[dict[str, Any]]:
        models = UiPath().agenthub.get_available_llm_models(
            headers=dict(self.build_auth_headers()),
        )
        return [model.model_dump(by_alias=True) for model in models]
