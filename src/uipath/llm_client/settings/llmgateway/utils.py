from enum import StrEnum


class LLMGatewayEndpoints(StrEnum):
    """API endpoint paths for the LLM Gateway.

    Normalized endpoints provide a consistent API across all providers.
    Passthrough endpoints expose vendor-specific APIs with formats for vendor/model.
    """

    IDENTITY_ENDPOINT = "identity_/connect/token"
    DISCOVERY_ENDPOINT = "llmgateway_/api/discovery"
    NORMALIZED_ENDPOOINT = "llmgateway_/api/{api_type}"
    PASSTHROUGH_ENDPOOINT = "llmgateway_/api/raw/vendor/{vendor}/model/{model}/{api_type}"
