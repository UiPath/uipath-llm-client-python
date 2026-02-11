"""Base settings for UiPath AgentHub client."""

import os
from collections.abc import Mapping
from typing import Any, Self

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from typing_extensions import override
from uipath._cli._auth._auth_service import AuthService
from uipath.utils import EndpointManager

from uipath_llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings


class AgentHubBaseSettings(UiPathBaseSettings):
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

    # Environment configuration: alpha, staging, cloud
    environment: str | None = Field(default=None, validation_alias="UIPATH_ENVIRONMENT")

    # Authentication fields - retrieved from uipath auth as well
    access_token: SecretStr | None = Field(default=None, validation_alias="UIPATH_ACCESS_TOKEN")
    base_url: str | None = Field(default=None, validation_alias="UIPATH_URL")
    tenant_id: str | None = Field(default=None, validation_alias="UIPATH_TENANT_ID")
    organization_id: str | None = Field(default=None, validation_alias="UIPATH_ORGANIZATION_ID")

    # OAuth credentials
    client_id: SecretStr | None = Field(default=None, validation_alias="UIPATH_CLIENT_ID")
    client_secret: SecretStr | None = Field(default=None, validation_alias="UIPATH_CLIENT_SECRET")
    client_scope: str | None = Field(default=None, validation_alias="UIPATH_CLIENT_SCOPE")

    # AgentHub configuration (used for discovery)
    agenthub_config: str = Field(default="agentsruntime", validation_alias="UIPATH_AGENTHUB_CONFIG")

    # Tracing configuration
    process_key: str | None = Field(default=None, validation_alias="UIPATH_PROCESS_KEY")
    job_key: str | None = Field(default=None, validation_alias="UIPATH_JOB_KEY")

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment and trigger authentication."""
        self.authenticate()
        return self

    def check_credentials(self) -> bool:
        """Check if all required credentials are present."""
        return (
            self.access_token is not None
            and self.base_url is not None
            and self.tenant_id is not None
            and self.organization_id is not None
        )

    def authenticate(self) -> None:
        """Authenticate with UiPath using the configured credentials."""
        if not self.check_credentials():
            auth_service = AuthService(
                environment=self.environment,
                force=True,
                client_id=self.client_id.get_secret_value() if self.client_id is not None else None,
                client_secret=self.client_secret.get_secret_value()
                if self.client_secret is not None
                else None,
                base_url=self.base_url,
                tenant=self.tenant_id,
                scope=self.client_scope,
            )
            auth_service.authenticate()
            load_dotenv(override=True)
            self.access_token = SecretStr(os.getenv("UIPATH_ACCESS_TOKEN", ""))
            self.base_url = os.getenv("UIPATH_URL")
            self.tenant_id = os.getenv("UIPATH_TENANT_ID")
            self.organization_id = os.getenv("UIPATH_ORGANIZATION_ID")
            if not self.check_credentials():
                raise ValueError("Could not authenticate with UiPath")

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
        if api_config.client_type == "normalized" and api_config.api_type == "completions":
            url = f"{self.base_url}/{EndpointManager.get_normalized_endpoint()}"
        elif api_config.client_type == "passthrough" and api_config.api_type == "embeddings":
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
