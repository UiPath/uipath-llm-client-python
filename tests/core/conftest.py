"""Core test fixtures shared across all core test modules."""

from unittest.mock import patch

import pytest

from uipath.llm_client.settings import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.settings.utils import SingletonMeta


@pytest.fixture(autouse=True)
def clear_singleton_instances():
    """Clear singleton instances before each test to ensure isolation."""
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


@pytest.fixture
def llmgw_env_vars():
    """Environment variables for LLMGatewaySettings."""
    return {
        "LLMGW_URL": "https://cloud.uipath.com",
        "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
        "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
        "LLMGW_REQUESTING_PRODUCT": "test-product",
        "LLMGW_REQUESTING_FEATURE": "test-feature",
        "LLMGW_ACCESS_TOKEN": "test-access-token",
    }


@pytest.fixture
def llmgw_s2s_env_vars():
    """Environment variables for LLMGatewaySettings with S2S auth."""
    return {
        "LLMGW_URL": "https://cloud.uipath.com",
        "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
        "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
        "LLMGW_REQUESTING_PRODUCT": "test-product",
        "LLMGW_REQUESTING_FEATURE": "test-feature",
        "LLMGW_CLIENT_ID": "test-client-id",
        "LLMGW_CLIENT_SECRET": "test-client-secret",
    }


@pytest.fixture
def platform_env_vars():
    """Environment variables for PlatformSettings."""
    return {
        "UIPATH_ACCESS_TOKEN": "test-access-token",
        "UIPATH_URL": "https://cloud.uipath.com/org/tenant",
        "UIPATH_TENANT_ID": "test-tenant-id",
        "UIPATH_ORGANIZATION_ID": "test-org-id",
    }


@pytest.fixture
def mock_platform_auth():
    """Patches is_token_expired and parse_access_token for PlatformSettings tests."""
    with (
        patch(
            "uipath.llm_client.settings.platform.settings.is_token_expired",
            return_value=False,
        ),
        patch(
            "uipath.llm_client.settings.platform.settings.parse_access_token",
            return_value={"client_id": "test-client-id"},
        ),
    ):
        yield


@pytest.fixture
def passthrough_api_config():
    """API config for passthrough mode."""
    return UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type="openai",
    )


@pytest.fixture
def normalized_api_config():
    """API config for normalized mode."""
    return UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.NORMALIZED,
    )


@pytest.fixture
def embeddings_api_config():
    """API config for embeddings."""
    return UiPathAPIConfig(
        api_type=ApiType.EMBEDDINGS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type="vertexai",
    )
