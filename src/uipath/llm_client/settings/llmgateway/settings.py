import logging
from collections.abc import Mapping
from typing import Any, Self

from httpx import Client
from pydantic import Field, SecretStr, model_validator
from typing_extensions import override

from uipath.llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
from uipath.llm_client.settings.llmgateway.utils import LLMGatewayEndpoints
from uipath.llm_client.utils.exceptions import UiPathAPIError


class LLMGatewayBaseSettings(UiPathBaseSettings):
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

    # Required to work, but if it's not set will be retrieved using S2S authentication
    access_token: SecretStr | None = Field(default=None, validation_alias="LLMGW_ACCESS_TOKEN")

    # Required for S2S authentication if access_token is not provided
    client_id: SecretStr | None = Field(default=None, validation_alias="LLMGW_CLIENT_ID")
    client_secret: SecretStr | None = Field(default=None, validation_alias="LLMGW_CLIENT_SECRET")

    # Required fields - no default means validation fails if missing
    base_url: str = Field(default=..., validation_alias="LLMGW_URL")
    org_id: str = Field(default=..., validation_alias="LLMGW_SEMANTIC_ORG_ID")
    tenant_id: str = Field(default=..., validation_alias="LLMGW_SEMANTIC_TENANT_ID")
    requesting_product: str = Field(default=..., validation_alias="LLMGW_REQUESTING_PRODUCT")
    requesting_feature: str = Field(default=..., validation_alias="LLMGW_REQUESTING_FEATURE")

    # Optional fields - situational usecase
    user_id: str | None = Field(
        default=None, validation_alias="LLMGW_SEMANTIC_USER_ID"
    )  # used to apply governance rules
    action_id: str | None = Field(
        default=None, validation_alias="LLMGW_ACTION_ID"
    )  # used to track the action
    operation_code: str | None = Field(
        default=None, validation_alias="LLMGW_OPERATION_CODE"
    )  # used to correctly identify byo models

    # additional headers for the request (e.g. X-UiPath-LlmGateway-Telemetry-SessionId)
    additional_headers: Mapping[str, str] = Field(
        default_factory=dict, validation_alias="LLMGW_ADDITIONAL_HEADERS"
    )

    @model_validator(mode="after")
    def validate_auth_settings(self) -> Self:
        """Validate that either access_token or S2S credentials are provided."""
        if self.access_token is None and (self.client_id is None or self.client_secret is None):
            raise ValueError(
                "Either access_token or both client_id and client_secret must be provided"
            )
        return self

    @override
    def build_base_url(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> str:
        base_url = f"{self.base_url}/{self.org_id}/{self.tenant_id}"
        if api_config is not None and api_config.client_type == "normalized":
            url = f"{base_url}/{LLMGatewayEndpoints.NORMALIZED_ENDPOINT.value.format(api_type='chat/completions' if api_config.api_type == 'completions' else 'embeddings')}"
        else:
            if api_config is None:
                raise ValueError("api_config is required for passthrough client_type")
            url = f"{base_url}/{LLMGatewayEndpoints.PASSTHROUGH_ENDPOINT.value.format(vendor=api_config.vendor_type, model=model_name, api_type=api_config.api_type)}"
        return url

    @override
    def build_auth_headers(
        self,
        *,
        model_name: str | None = None,
        api_config: UiPathAPIConfig | None = None,
    ) -> Mapping[str, str]:
        headers = {
            "X-UiPath-LlmGateway-RequestingProduct": self.requesting_product,
            "X-UiPath-LlmGateway-RequestingFeature": self.requesting_feature,
            "X-UiPath-Internal-AccountId": self.org_id,
            "X-UiPath-Internal-TenantId": self.tenant_id,
        }
        if self.user_id:
            headers["X-UiPath-LlmGateway-UserId"] = self.user_id
        if self.action_id:
            headers["X-UiPath-LlmGateway-ActionId"] = self.action_id
        if self.operation_code:
            headers["X-UiPath-LlmGateway-OperationCode"] = self.operation_code
        if self.additional_headers:
            headers.update(self.additional_headers)
        return headers

    @override
    def get_available_models(self) -> list[dict[str, Any]]:
        discovery_url = f"{self.base_url}/{self.org_id}/{self.tenant_id}/{LLMGatewayEndpoints.DISCOVERY_ENDPOINT.value}"
        with Client(auth=self.build_auth_pipeline(), headers=self.build_auth_headers()) as client:
            response = client.get(discovery_url)
            if response.is_error:
                raise UiPathAPIError.from_response(response)
            return response.json()

    @override
    def validate_byo_model(self, model_info: dict[str, Any]) -> None:
        byom_details = model_info.get("byomDetails", {})
        operation_codes = byom_details.get("operationCodes", [])
        if self.operation_code and self.operation_code not in operation_codes:
            raise ValueError(
                f"The operation code {self.operation_code} is not allowed for the model {model_info['modelName']}"
            )
        if not self.operation_code and len(operation_codes) > 0:
            if len(operation_codes) > 1:
                logging.warning(
                    "Multiple operation codes are allowed for the model %s, but no operation code was provided, picking the first one available: %s",
                    model_info["modelName"],
                    operation_codes[0],
                )
            self.operation_code = operation_codes[0]
