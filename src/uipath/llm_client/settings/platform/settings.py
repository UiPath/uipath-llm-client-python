"""Base settings for UiPath Platform (AgentHub/Orchestrator) client."""

from collections.abc import Mapping
from typing import Any, Self

from pydantic import Field, SecretStr, model_validator
from typing_extensions import override
from uipath.platform.common import EndpointManager

from uipath.llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
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
    access_token: SecretStr | None = Field(default=None, validation_alias="UIPATH_ACCESS_TOKEN")
    base_url: str | None = Field(default=None, validation_alias="UIPATH_URL")
    tenant_id: str | None = Field(default=None, validation_alias="UIPATH_TENANT_ID")
    organization_id: str | None = Field(default=None, validation_alias="UIPATH_ORGANIZATION_ID")

    # Credentials used for refreshing the access token
    client_id: str | None = Field(default=None)
    refresh_token: SecretStr | None = Field(default=None, validation_alias="UIPATH_REFRESH_TOKEN")

    # AgentHub configuration (used for discovery)
    agenthub_config: str | None = Field(
        default="agentsruntime", validation_alias="UIPATH_AGENTHUB_CONFIG"
    )

    # Tracing configuration
    process_key: str | None = Field(default=None, validation_alias="UIPATH_PROCESS_KEY")
    job_key: str | None = Field(default=None, validation_alias="UIPATH_JOB_KEY")

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment and trigger authentication."""
        if (
            self.access_token is None
            or self.base_url is None
            or self.tenant_id is None
            or self.organization_id is None
        ):
            raise ValueError(
                "Base URL, access token, tenant ID, and organization ID are required. Try running `uipath auth` to authenticate."
            )

        access_token = self.access_token.get_secret_value()
        if is_token_expired(access_token):
            raise ValueError(
                "Access token is expired. Try running `uipath auth` to refresh the token."
            )

        parsed_token_data = parse_access_token(access_token)
        self.client_id = parsed_token_data.get("client_id", None)
        return self

    @override
    def build_base_url(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> str:
        """Build the base URL for API requests."""
        assert model_name is not None
        assert api_config is not None
        if api_config.routing_mode == "normalized" and api_config.api_type == "completions":
            url = f"{self.base_url}/{EndpointManager.get_normalized_endpoint()}"
        elif api_config.routing_mode == "normalized" and api_config.api_type == "embeddings":
            raise ValueError(
                "Normalized embeddings are not supported on UiPath Platform (AgentHub/Orchestrator). "
                "Use passthrough routing mode for embeddings instead."
            )
        elif api_config.routing_mode == "passthrough" and api_config.api_type == "embeddings":
            assert api_config.api_version is not None
            url = f"{self.base_url}/{EndpointManager.get_embeddings_endpoint().format(model=model_name, api_version=api_config.api_version)}"
        else:
            assert api_config.vendor_type is not None
            url = f"{self.base_url}/{EndpointManager.get_vendor_endpoint().format(model=model_name, vendor=api_config.vendor_type)}"
        return url

    @override
    def build_auth_headers(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> Mapping[str, str]:
        """Build authentication and routing headers for API requests."""
        headers: dict[str, str] = {}
        if self.agenthub_config:
            headers["X-UiPath-AgentHub-Config"] = self.agenthub_config
        if self.process_key:
            headers["X-UiPath-ProcessKey"] = self.process_key
        if self.job_key:
            headers["X-UiPath-JobKey"] = self.job_key
        return headers

    @override
    def get_available_models(self) -> list[dict[str, Any]]:
        from uipath.platform import UiPath

        models = UiPath().agenthub.get_available_llm_models(
            headers=dict(self.build_auth_headers()),
        )
        return [model.model_dump(by_alias=True) for model in models]

    @override
    def validate_byo_model(self, model_info: dict[str, Any]) -> None:
        return
