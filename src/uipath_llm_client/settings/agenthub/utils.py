"""Utility classes for AgentHub settings."""

from enum import StrEnum


class AgentHubEndpoints(StrEnum):
    """API endpoint paths for UiPath AgentHub.

    Normalized endpoints provide a consistent API across all providers.
    Passthrough endpoints expose vendor-specific APIs with formats for vendor/model.
    """

    NORMALIZED_ENDPOINT = "agenthub_/llm/api/chat/{api_type}"
    PASSTHROUGH_ENDPOINT = "agenthub_/llm/raw/vendor/{vendor}/model/{model}/{api_type}"
