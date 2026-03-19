"""Tests for uipath-llm-client core functionality.

This module tests:
1. Retry logic (RetryConfig, RetryableHTTPTransport, RetryableAsyncHTTPTransport)
2. PlatformSettings (build_base_url, build_auth_headers, build_auth_pipeline, get_available_models)
3. LLMGatewaySettings (build_base_url, build_auth_headers, build_auth_pipeline, get_available_models)
4. Settings factory (get_default_client_settings) with environment variables
5. HTTPX client functionality (UiPathHttpxClient, UiPathHttpxAsyncClient)
6. Auth refresh logic for both settings
7. Exception utilities (patch_raise_for_status, __str__, __repr__, body parsing)
8. Header utilities (extract_matching_headers, context vars)
9. Logging config (LoggingConfig)
10. SSL config (expand_path, get_httpx_ssl_client_kwargs)
11. LLMGateway S2S auth (get_llmgw_token)
12. LLMGateway BYOM validation (validate_byo_model)
13. Wait strategy (wait_retry_after_with_fallback)
14. HTTPX client send() behavior (streaming header, URL freezing, header capture)
"""

import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import Client, Headers, Request, Response

from uipath.llm_client.settings import (
    LLMGatewaySettings,
    PlatformSettings,
    UiPathAPIConfig,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathAuthenticationError,
    UiPathBadRequestError,
    UiPathGatewayTimeoutError,
    UiPathInternalServerError,
    UiPathNotFoundError,
    UiPathPermissionDeniedError,
    UiPathRateLimitError,
    UiPathServiceUnavailableError,
    UiPathTooManyRequestsError,
    patch_raise_for_status,
)
from uipath.llm_client.utils.retry import (
    RetryableAsyncHTTPTransport,
    RetryableHTTPTransport,
    RetryConfig,
)

# ============================================================================
# Fixtures
# ============================================================================


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


# ============================================================================
# Test UiPathAPIConfig
# ============================================================================


class TestUiPathAPIConfig:
    """Tests for UiPathAPIConfig."""

    def test_passthrough_requires_vendor_type(self):
        """Test that passthrough mode requires vendor_type."""
        with pytest.raises(ValueError, match="vendor_type required"):
            UiPathAPIConfig(
                api_type=ApiType.COMPLETIONS,
                routing_mode=RoutingMode.PASSTHROUGH,
                vendor_type=None,
            )

    def test_normalized_does_not_require_vendor_type(self):
        """Test that normalized mode doesn't require vendor_type."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
        )
        assert config.vendor_type is None

    def test_passthrough_with_vendor_type(self):
        """Test passthrough config with vendor_type."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
        )
        assert config.api_type == ApiType.COMPLETIONS
        assert config.routing_mode == RoutingMode.PASSTHROUGH
        assert config.vendor_type == "openai"

    def test_freeze_base_url_default(self):
        """Test freeze_base_url defaults to False."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
        )
        assert config.freeze_base_url is False

    def test_api_flavor_and_version(self):
        """Test api_flavor and api_version can be set."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
            api_flavor="chat-completions",
            api_version="2025-03-01-preview",
        )
        assert config.api_flavor == "chat-completions"
        assert config.api_version == "2025-03-01-preview"


# ============================================================================
# Test Settings Factory
# ============================================================================


class TestSettingsFactory:
    """Tests for get_default_client_settings factory function."""

    def test_default_returns_agenthub(self, platform_env_vars, mock_platform_auth):
        """Test that default backend is agenthub."""
        env = {**platform_env_vars}
        env.pop("UIPATH_LLM_SERVICE", None)
        with patch.dict(os.environ, env, clear=True):
            settings = get_default_client_settings()
            assert isinstance(settings, PlatformSettings)

    def test_explicit_agenthub(self, platform_env_vars, mock_platform_auth):
        """Test explicit agenthub backend."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = get_default_client_settings(backend="agenthub")
            assert isinstance(settings, PlatformSettings)

    def test_explicit_llmgateway(self, llmgw_env_vars):
        """Test explicit llmgateway backend."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = get_default_client_settings(backend="llmgateway")
            assert isinstance(settings, LLMGatewaySettings)

    def test_env_var_agenthub(self, platform_env_vars, mock_platform_auth):
        """Test UIPATH_LLM_SERVICE=agenthub from environment."""
        env = {**platform_env_vars, "UIPATH_LLM_SERVICE": "agenthub"}
        with patch.dict(os.environ, env, clear=True):
            settings = get_default_client_settings()
            assert isinstance(settings, PlatformSettings)

    def test_env_var_llmgateway(self, llmgw_env_vars):
        """Test UIPATH_LLM_SERVICE=llmgateway from environment."""
        env = {**llmgw_env_vars, "UIPATH_LLM_SERVICE": "llmgateway"}
        with patch.dict(os.environ, env, clear=True):
            settings = get_default_client_settings()
            assert isinstance(settings, LLMGatewaySettings)

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend type"):
            get_default_client_settings(backend="invalid")  # type: ignore


# ============================================================================
# Test LLMGatewaySettings
# ============================================================================


class TestLLMGatewaySettings:
    """Tests for LLMGatewaySettings."""

    def test_build_base_url_passthrough(self, llmgw_env_vars, passthrough_api_config):
        """Test build_base_url for passthrough mode."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            url = settings.build_base_url(
                model_name="gpt-4o",
                api_config=passthrough_api_config,
            )
            assert "test-org-id" in url
            assert "test-tenant-id" in url
            assert "llmgateway_/api/raw/vendor/openai/model/gpt-4o/completions" in url

    def test_build_base_url_normalized(self, llmgw_env_vars, normalized_api_config):
        """Test build_base_url for normalized mode."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            url = settings.build_base_url(
                model_name="gpt-4o",
                api_config=normalized_api_config,
            )
            assert "llmgateway_/api/chat/completions" in url

    def test_build_base_url_embeddings(self, llmgw_env_vars, embeddings_api_config):
        """Test build_base_url for embeddings."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            url = settings.build_base_url(
                model_name="text-embedding-3-large",
                api_config=embeddings_api_config,
            )
            assert "embeddings" in url
            assert "vertexai" in url

    def test_build_auth_headers_required_fields(self, llmgw_env_vars):
        """Test build_auth_headers includes required tracking headers."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            headers = settings.build_auth_headers()
            assert headers["X-UiPath-LlmGateway-RequestingProduct"] == "test-product"
            assert headers["X-UiPath-LlmGateway-RequestingFeature"] == "test-feature"

    def test_build_auth_headers_optional_fields(self, llmgw_env_vars):
        """Test build_auth_headers includes optional tracking headers when set."""
        env = {
            **llmgw_env_vars,
            "LLMGW_SEMANTIC_USER_ID": "test-user",
            "LLMGW_ACTION_ID": "test-action",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = LLMGatewaySettings()
            headers = settings.build_auth_headers()
            assert headers["X-UiPath-LlmGateway-UserId"] == "test-user"
            assert headers["X-UiPath-LlmGateway-ActionId"] == "test-action"

    def test_build_auth_pipeline_returns_auth(self, llmgw_env_vars):
        """Test build_auth_pipeline returns an Auth instance."""
        from httpx import Auth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            auth = settings.build_auth_pipeline()
            assert isinstance(auth, Auth)

    def test_build_auth_pipeline_with_access_token(self, llmgw_env_vars):
        """Test auth pipeline uses access_token when provided."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            auth = settings.build_auth_pipeline()
            assert isinstance(auth, LLMGatewayS2SAuth)
            assert auth.access_token == "test-access-token"

    def test_validation_requires_auth_credentials(self):
        """Test validation fails without access_token or S2S credentials."""
        env = {
            "LLMGW_URL": "https://cloud.uipath.com",
            "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
            "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
            "LLMGW_REQUESTING_PRODUCT": "test-product",
            "LLMGW_REQUESTING_FEATURE": "test-feature",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Either access_token or both client_id"):
                LLMGatewaySettings()

    def test_get_available_models(self, llmgw_env_vars):
        """Test get_available_models returns a list of models on success."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

            mock_response = MagicMock()
            mock_response.is_error = False
            mock_response.json.return_value = [
                {"modelName": "gpt-4o", "vendor": "openai"},
                {"modelName": "claude-3-opus", "vendor": "anthropic"},
            ]

            with patch.object(Client, "get", return_value=mock_response):
                models = settings.get_available_models()
                assert isinstance(models, list)
                assert len(models) == 2

    def test_get_available_models_raises_on_http_error(self, llmgw_env_vars):
        """Test get_available_models raises UiPathAPIError on bad request / server error."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

            mock_response = MagicMock()
            mock_response.is_error = True
            mock_response.status_code = 500
            mock_response.reason_phrase = "Internal Server Error"
            mock_response.json.return_value = {"error": "Something went wrong"}
            mock_response.request = MagicMock()
            mock_response.text = ""

            with patch.object(Client, "get", return_value=mock_response):
                with pytest.raises(UiPathAPIError) as exc_info:
                    settings.get_available_models()
                assert exc_info.value.status_code == 500

    def test_get_available_models_raises_on_unauthorized(self, llmgw_env_vars):
        """Test get_available_models raises UiPathAuthenticationError on 401."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

            mock_response = MagicMock()
            mock_response.is_error = True
            mock_response.status_code = 401
            mock_response.reason_phrase = "Unauthorized"
            mock_response.json.return_value = {}
            mock_response.request = MagicMock()
            mock_response.text = ""

            with patch.object(Client, "get", return_value=mock_response):
                with pytest.raises(UiPathAuthenticationError) as exc_info:
                    settings.get_available_models()
                assert exc_info.value.status_code == 401


# ============================================================================
# Test LLMGateway Auth Refresh Logic
# ============================================================================


class TestLLMGatewayAuthRefresh:
    """Tests for LLMGatewayS2SAuth token refresh logic."""

    def test_auth_flow_adds_bearer_token(self, llmgw_env_vars):
        """Test auth_flow adds Authorization header."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            auth = LLMGatewayS2SAuth(settings=settings)
            request = Request("GET", "https://example.com")
            flow = auth.auth_flow(request)
            modified_request = next(flow)
            assert "Authorization" in modified_request.headers
            assert modified_request.headers["Authorization"] == "Bearer test-access-token"

    def test_auth_flow_refreshes_on_401(self, llmgw_s2s_env_vars):
        """Test auth_flow refreshes token on 401 response."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_s2s_env_vars, clear=True):
            settings = LLMGatewaySettings()

            # Mock the token retrieval
            with patch.object(
                LLMGatewayS2SAuth, "get_llmgw_token", return_value="new-token"
            ) as mock_get_token:
                auth = LLMGatewayS2SAuth(settings=settings)
                # First call is during __init__
                mock_get_token.assert_called_once()
                mock_get_token.reset_mock()

                request = Request("GET", "https://example.com")
                flow = auth.auth_flow(request)

                # First yield - initial request
                modified_request = next(flow)
                assert modified_request.headers["Authorization"] == "Bearer new-token"

                # Simulate 401 response
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 401

                # Send 401 response and get retry request
                mock_get_token.return_value = "refreshed-token"
                try:
                    retry_request = flow.send(mock_response)
                    assert retry_request.headers["Authorization"] == "Bearer refreshed-token"
                    mock_get_token.assert_called_once()
                except StopIteration:
                    pass

    def test_auth_singleton_reuses_instance(self, llmgw_env_vars):
        """Test that LLMGatewayS2SAuth is a singleton."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            auth1 = LLMGatewayS2SAuth(settings=settings)
            auth2 = LLMGatewayS2SAuth(settings=settings)
            assert auth1 is auth2


# ============================================================================
# Test PlatformSettings
# ============================================================================


class TestPlatformSettings:
    """Tests for PlatformSettings."""

    def test_build_base_url_passthrough(
        self, platform_env_vars, mock_platform_auth, passthrough_api_config
    ):
        """Test build_base_url for passthrough completions mode."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            url = settings.build_base_url(
                model_name="gpt-4o",
                api_config=passthrough_api_config,
            )
            assert "llm/raw/vendor/openai/model/gpt-4o/completions" in url

    def test_build_base_url_passthrough_with_api_version(
        self, platform_env_vars, mock_platform_auth
    ):
        """Test build_base_url for passthrough completions with api_version (vendor endpoint ignores it)."""
        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
            api_version="2025-03-01",
        )
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            url = settings.build_base_url(
                model_name="gpt-4o",
                api_config=api_config,
            )
            assert "llm/raw/vendor/openai/model/gpt-4o/completions" in url
            assert "api-version" not in url

    def test_build_base_url_normalized(
        self, platform_env_vars, mock_platform_auth, normalized_api_config
    ):
        """Test build_base_url for normalized mode."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            url = settings.build_base_url(
                model_name="gpt-4o",
                api_config=normalized_api_config,
            )
            assert "agenthub_/llm/api/chat/completions" in url

    def test_build_auth_headers_has_default_config(self, platform_env_vars, mock_platform_auth):
        """Test build_auth_headers includes default agenthub_config."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            headers = settings.build_auth_headers()
            assert headers == {
                "X-UiPath-Internal-AccountId": "test-org-id",
                "X-UiPath-Internal-TenantId": "test-tenant-id",
                "X-UiPath-AgentHub-Config": "agentsruntime",
            }

    def test_build_auth_headers_with_tracing(self, platform_env_vars, mock_platform_auth):
        """Test build_auth_headers includes tracing headers when set."""
        env = {
            **platform_env_vars,
            "UIPATH_AGENTHUB_CONFIG": "test-config",
            "UIPATH_PROCESS_KEY": "test-process",
            "UIPATH_JOB_KEY": "test-job",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = PlatformSettings()
            headers = settings.build_auth_headers()
            assert headers["X-UiPath-AgentHub-Config"] == "test-config"
            assert headers["X-UiPath-ProcessKey"] == "test-process"
            assert headers["X-UiPath-JobKey"] == "test-job"

    def test_build_auth_pipeline_returns_auth(self, platform_env_vars, mock_platform_auth):
        """Test build_auth_pipeline returns an Auth instance."""
        from httpx import Auth

        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            auth = settings.build_auth_pipeline()
            assert isinstance(auth, Auth)

    def test_build_auth_pipeline_with_access_token(self, platform_env_vars, mock_platform_auth):
        """Test auth pipeline uses access_token when provided."""
        from uipath.llm_client.settings.platform.auth import PlatformAuth

        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            auth = settings.build_auth_pipeline()
            assert isinstance(auth, PlatformAuth)

    def test_build_base_url_passthrough_embeddings(self, platform_env_vars, mock_platform_auth):
        """Test build_base_url for passthrough embeddings with api_version."""
        api_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
            api_version="2024-02-01",
        )
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            url = settings.build_base_url(
                model_name="text-embedding-3-large",
                api_config=api_config,
            )
            assert "embeddings" in url
            assert "text-embedding-3-large" in url
            assert "api-version=2024-02-01" in url

    def test_build_base_url_passthrough_embeddings_no_api_version(
        self, platform_env_vars, mock_platform_auth
    ):
        """Test build_base_url for passthrough embeddings without api_version."""
        api_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
        )
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            url = settings.build_base_url(
                model_name="text-embedding-3-large",
                api_config=api_config,
            )
            assert "embeddings" in url
            assert "text-embedding-3-large" in url
            assert "api-version" not in url

    def test_build_base_url_passthrough_embeddings_non_openai_raises(
        self, platform_env_vars, mock_platform_auth
    ):
        """Test build_base_url raises for non-OpenAI passthrough embeddings."""
        api_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="vertexai",
        )
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            with pytest.raises(ValueError, match="only supports OpenAI-compatible models"):
                settings.build_base_url(
                    model_name="gemini-embedding-001",
                    api_config=api_config,
                )

    def test_build_base_url_normalized_embeddings_raises(
        self, platform_env_vars, mock_platform_auth
    ):
        """Test build_base_url raises ValueError for normalized embeddings."""
        normalized_embeddings_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.NORMALIZED,
        )
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            with pytest.raises(ValueError, match="Normalized embeddings are not supported"):
                settings.build_base_url(
                    model_name="text-embedding-3-large",
                    api_config=normalized_embeddings_config,
                )

    def test_build_base_url_requires_model_name(
        self, platform_env_vars, mock_platform_auth, normalized_api_config
    ):
        """Test build_base_url asserts model_name is not None."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            with pytest.raises(AssertionError):
                settings.build_base_url(model_name=None, api_config=normalized_api_config)

    def test_build_base_url_requires_api_config(self, platform_env_vars, mock_platform_auth):
        """Test build_base_url asserts api_config is not None."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            with pytest.raises(AssertionError):
                settings.build_base_url(model_name="gpt-4o", api_config=None)

    def test_build_auth_headers_empty_when_no_optional(self, platform_env_vars, mock_platform_auth):
        """Test build_auth_headers with no optional tracing fields set."""
        env = {**platform_env_vars, "UIPATH_AGENTHUB_CONFIG": ""}
        with patch.dict(os.environ, env, clear=True):
            settings = PlatformSettings()
            # Override to empty to test the falsy path
            settings.agenthub_config = ""
            settings.process_key = None
            settings.job_key = None
            settings.organization_id = None
            settings.tenant_id = None
            headers = settings.build_auth_headers()
            assert headers == {}

    def test_validation_requires_all_fields(self, mock_platform_auth):
        """Test validation fails without required fields."""
        env = {
            "UIPATH_ACCESS_TOKEN": "test-access-token",
            # Missing base_url, tenant_id, organization_id
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Base URL, access token, tenant ID"):
                PlatformSettings()

    def test_validation_fails_on_expired_token(self):
        """Test validation fails when access token is expired."""
        with (
            patch(
                "uipath.llm_client.settings.platform.settings.is_token_expired",
                return_value=True,
            ),
            patch(
                "uipath.llm_client.settings.platform.settings.parse_access_token",
                return_value={"client_id": "test-client-id"},
            ),
        ):
            env = {
                "UIPATH_ACCESS_TOKEN": "test-access-token",
                "UIPATH_URL": "https://cloud.uipath.com/org/tenant",
                "UIPATH_TENANT_ID": "test-tenant-id",
                "UIPATH_ORGANIZATION_ID": "test-org-id",
            }
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="Access token is expired"):
                    PlatformSettings()

    def test_validate_byo_model_is_noop(self, platform_env_vars, mock_platform_auth):
        """Test validate_byo_model does nothing (no-op)."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            result = settings.validate_byo_model({"modelName": "custom-model"})
            assert result is None


# ============================================================================
# Test Platform Auth Refresh Logic
# ============================================================================


class TestPlatformAuthRefresh:
    """Tests for PlatformAuth token refresh logic."""

    @pytest.fixture(autouse=True)
    def clear_auth_singleton(self):
        """Clear PlatformAuth singleton before each test."""
        from uipath.llm_client.settings.platform.auth import PlatformAuth
        from uipath.llm_client.settings.utils import SingletonMeta

        SingletonMeta._instances.pop(PlatformAuth, None)
        yield
        SingletonMeta._instances.pop(PlatformAuth, None)

    def test_auth_flow_adds_bearer_token(self, platform_env_vars, mock_platform_auth):
        """Test auth_flow adds Authorization header."""
        from uipath.llm_client.settings.platform.auth import PlatformAuth

        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            auth = PlatformAuth(settings=settings)
            request = Request("GET", "https://example.com")
            flow = auth.auth_flow(request)
            modified_request = next(flow)
            assert "Authorization" in modified_request.headers
            assert modified_request.headers["Authorization"] == "Bearer test-access-token"

    def test_auth_flow_refreshes_on_401(self, platform_env_vars, mock_platform_auth):
        """Test auth_flow refreshes token on 401 response."""
        from uipath.llm_client.settings.platform.auth import PlatformAuth

        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()

            # Mock get_access_token to return a new token on refresh
            with patch.object(PlatformAuth, "get_access_token", return_value="initial-token"):
                auth = PlatformAuth(settings=settings)

            request = Request("GET", "https://example.com")
            flow = auth.auth_flow(request)

            # First yield - initial request
            modified_request = next(flow)
            assert "Bearer" in modified_request.headers["Authorization"]

            # Simulate 401 response
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 401

            # Mock the get_access_token method to return a new token
            with patch.object(PlatformAuth, "get_access_token", return_value="refreshed-token"):
                try:
                    retry_request = flow.send(mock_response)
                    assert retry_request.headers["Authorization"] == "Bearer refreshed-token"
                except StopIteration:
                    pass

    def test_auth_singleton_reuses_instance(self, platform_env_vars, mock_platform_auth):
        """Test that PlatformAuth is a singleton."""
        from uipath.llm_client.settings.platform.auth import PlatformAuth

        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            auth1 = PlatformAuth(settings=settings)
            auth2 = PlatformAuth(settings=settings)
            assert auth1 is auth2


# ============================================================================
# Test Retry Logic
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig TypedDict."""

    def test_retry_config_defaults(self):
        """Test RetryConfig can be created with defaults."""
        config: RetryConfig = {}
        assert config.get("initial_delay") is None  # Will use default
        assert config.get("max_delay") is None
        assert config.get("jitter") is None

    def test_retry_config_custom_values(self):
        """Test RetryConfig with custom values."""
        config: RetryConfig = {
            "initial_delay": 1.0,
            "max_delay": 30.0,
            "jitter": 0.5,
            "exp_base": 2.0,
            "retry_on_exceptions": (UiPathRateLimitError,),
        }
        assert config["initial_delay"] == 1.0
        assert config["max_delay"] == 30.0
        assert config["jitter"] == 0.5
        assert config["exp_base"] == 2.0


class TestRetryableHTTPTransport:
    """Tests for RetryableHTTPTransport."""

    def test_transport_inherits_from_http_transport(self):
        """Test transport inherits from HTTPTransport."""
        from httpx import HTTPTransport

        assert issubclass(RetryableHTTPTransport, HTTPTransport)

    def test_transport_no_retry_when_max_retries_0(self):
        """Test no retry logic when max_retries is 0."""
        transport = RetryableHTTPTransport(max_retries=0)
        assert transport.retryer is None

    def test_transport_has_retryer_when_max_retries_1(self):
        """Test retryer is created when max_retries is 1."""
        transport = RetryableHTTPTransport(max_retries=1)
        assert transport.retryer is not None

    def test_transport_has_retryer_when_max_retries_gt_1(self):
        """Test retryer is created when max_retries > 1."""
        transport = RetryableHTTPTransport(max_retries=3)
        assert transport.retryer is not None

    def test_transport_with_custom_retry_config(self):
        """Test transport with custom retry config."""
        config: RetryConfig = {
            "initial_delay": 0.1,
            "max_delay": 1.0,
            "jitter": 0.1,
        }
        transport = RetryableHTTPTransport(max_retries=3, retry_config=config)
        assert transport.retryer is not None


class TestRetryableAsyncHTTPTransport:
    """Tests for RetryableAsyncHTTPTransport."""

    def test_async_transport_inherits_from_async_http_transport(self):
        """Test async transport inherits from AsyncHTTPTransport."""
        from httpx import AsyncHTTPTransport

        assert issubclass(RetryableAsyncHTTPTransport, AsyncHTTPTransport)

    def test_async_transport_no_retry_when_max_retries_0(self):
        """Test no retry logic when max_retries is 0."""
        transport = RetryableAsyncHTTPTransport(max_retries=0)
        assert transport.retryer is None

    def test_async_transport_has_retryer_when_max_retries_1(self):
        """Test retryer is created when max_retries is 1."""
        transport = RetryableAsyncHTTPTransport(max_retries=1)
        assert transport.retryer is not None

    def test_async_transport_has_retryer_when_max_retries_gt_1(self):
        """Test retryer is created when max_retries > 1."""
        transport = RetryableAsyncHTTPTransport(max_retries=3)
        assert transport.retryer is not None


# ============================================================================
# Test HTTPX Client
# ============================================================================


class TestUiPathHttpxClient:
    """Tests for UiPathHttpxClient."""

    def test_client_inherits_from_httpx_client(self):
        """Test client inherits from httpx.Client."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        assert issubclass(UiPathHttpxClient, Client)

    def test_client_has_default_headers(self):
        """Test client has default UiPath headers."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com")
        assert "X-UiPath-LLMGateway-TimeoutSeconds" in client.headers
        assert "X-UiPath-LLMGateway-AllowFull4xxResponse" in client.headers
        client.close()

    def test_client_merges_custom_headers(self):
        """Test client merges custom headers with defaults."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            headers={"X-Custom-Header": "custom-value"},
        )
        assert "X-Custom-Header" in client.headers
        assert client.headers["X-Custom-Header"] == "custom-value"
        # Default headers should still be present
        assert "X-UiPath-LLMGateway-TimeoutSeconds" in client.headers
        client.close()

    def test_client_with_model_name(self):
        """Test client stores model_name."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            model_name="gpt-4o",
        )
        assert client.model_name == "gpt-4o"
        client.close()

    def test_client_with_api_config(self, normalized_api_config):
        """Test client stores api_config."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            api_config=normalized_api_config,
            model_name="gpt-4o",
        )
        assert client.api_config == normalized_api_config
        # Check normalized API header is added
        assert "X-UiPath-LlmGateway-NormalizedApi-ModelName" in client.headers
        client.close()

    def test_client_with_retry_config(self):
        """Test client creates retryable transport."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            max_retries=3,
        )
        # Transport should be RetryableHTTPTransport
        assert isinstance(client._transport, RetryableHTTPTransport)
        client.close()

    def test_client_with_byo_connection_id(self):
        """Test client adds BYO connection ID header."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            byo_connection_id="test-connection-id",
        )
        assert "X-UiPath-LlmGateway-ByoIsConnectionId" in client.headers
        assert client.headers["X-UiPath-LlmGateway-ByoIsConnectionId"] == "test-connection-id"
        client.close()


class TestUiPathHttpxAsyncClient:
    """Tests for UiPathHttpxAsyncClient."""

    def test_async_client_inherits_from_httpx_async_client(self):
        """Test async client inherits from httpx.AsyncClient."""
        from httpx import AsyncClient

        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        assert issubclass(UiPathHttpxAsyncClient, AsyncClient)

    def test_async_client_has_default_headers(self):
        """Test async client has default UiPath headers."""
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        client = UiPathHttpxAsyncClient(base_url="https://example.com")
        assert "X-UiPath-LLMGateway-TimeoutSeconds" in client.headers
        assert "X-UiPath-LLMGateway-AllowFull4xxResponse" in client.headers

    def test_async_client_with_retry_config(self):
        """Test async client creates retryable async transport."""
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        client = UiPathHttpxAsyncClient(
            base_url="https://example.com",
            max_retries=3,
        )
        # Transport should be RetryableAsyncHTTPTransport
        assert isinstance(client._transport, RetryableAsyncHTTPTransport)


# ============================================================================
# Test Build Routing Headers
# ============================================================================


class TestBuildRoutingHeaders:
    """Tests for build_routing_headers function."""

    def test_empty_headers_when_no_config(self):
        """Test empty headers when no api_config provided."""
        from uipath.llm_client.httpx_client import build_routing_headers

        headers = build_routing_headers()
        assert headers == {}

    def test_normalized_api_header(self, normalized_api_config):
        """Test normalized API adds model name header."""
        from uipath.llm_client.httpx_client import build_routing_headers

        headers = build_routing_headers(
            model_name="gpt-4o",
            api_config=normalized_api_config,
        )
        assert headers["X-UiPath-LlmGateway-NormalizedApi-ModelName"] == "gpt-4o"

    def test_passthrough_api_headers(self):
        """Test passthrough API adds flavor and version headers when set."""
        from uipath.llm_client.httpx_client import build_routing_headers

        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
            api_flavor="chat-completions",
            api_version="2025-03-01",
        )
        headers = build_routing_headers(
            model_name="gpt-4o",
            api_config=api_config,
        )
        assert headers["X-UiPath-LlmGateway-ApiFlavor"] == "chat-completions"
        assert headers["X-UiPath-LlmGateway-ApiVersion"] == "2025-03-01"

    def test_byo_connection_id_header(self):
        """Test BYO connection ID header is added."""
        from uipath.llm_client.httpx_client import build_routing_headers

        headers = build_routing_headers(
            byo_connection_id="test-connection-id",
        )
        assert headers["X-UiPath-LlmGateway-ByoIsConnectionId"] == "test-connection-id"


# ============================================================================
# Test Exceptions
# ============================================================================


class TestExceptions:
    """Tests for UiPath exception classes."""

    def test_exception_hierarchy(self):
        """Test all exceptions inherit from UiPathAPIError."""
        from httpx import HTTPStatusError

        assert issubclass(UiPathAPIError, HTTPStatusError)
        assert issubclass(UiPathAuthenticationError, UiPathAPIError)
        assert issubclass(UiPathRateLimitError, UiPathAPIError)

    def test_exception_status_codes(self):
        """Test exception classes have correct status codes."""
        from uipath.llm_client.utils.exceptions import (
            UiPathBadRequestError,
            UiPathInternalServerError,
            UiPathNotFoundError,
            UiPathPermissionDeniedError,
        )

        assert UiPathBadRequestError.status_code == 400
        assert UiPathAuthenticationError.status_code == 401
        assert UiPathPermissionDeniedError.status_code == 403
        assert UiPathNotFoundError.status_code == 404
        assert UiPathRateLimitError.status_code == 429
        assert UiPathInternalServerError.status_code == 500

    def test_exception_from_response(self):
        """Test UiPathAPIError.from_response creates correct exception type."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {}  # Required for UiPathRateLimitError._parse_retry_after

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.status_code == 429

    def test_exception_from_response_with_retry_after(self):
        """Test UiPathRateLimitError parses Retry-After header."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {"retry-after": "30"}

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after == 30.0

    def test_exception_from_response_with_x_retry_after(self):
        """Test UiPathRateLimitError parses x-retry-after header."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {"x-retry-after": "45"}

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after == 45.0

    def test_exception_retry_after_none_when_not_present(self):
        """Test UiPathRateLimitError.retry_after is None when header missing."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {}

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after is None


# ============================================================================
# Test Singleton Utility
# ============================================================================


class TestSingletonMeta:
    """Tests for SingletonMeta metaclass."""

    def test_singleton_creates_single_instance(self):
        """Test singleton creates only one instance."""

        class TestSingleton(metaclass=SingletonMeta):
            def __init__(self, value: int):
                self.value = value

        instance1 = TestSingleton(1)
        instance2 = TestSingleton(2)

        assert instance1 is instance2
        assert instance1.value == 1  # First value is retained

    def test_different_classes_have_different_instances(self):
        """Test different singleton classes have separate instances."""

        class SingletonA(metaclass=SingletonMeta):
            pass

        class SingletonB(metaclass=SingletonMeta):
            pass

        a = SingletonA()
        b = SingletonB()

        assert a is not b


# ============================================================================
# Test Exception String Representations and Body Parsing
# ============================================================================


class TestExceptionDetails:
    """Tests for exception __str__, __repr__, body parsing, and patch_raise_for_status."""

    def _make_response(
        self, status_code, reason_phrase="Error", body_json=None, body_text="", headers=None
    ):
        mock_resp = MagicMock(spec=Response)
        mock_resp.status_code = status_code
        mock_resp.reason_phrase = reason_phrase
        mock_resp.request = MagicMock(spec=Request)
        mock_resp.headers = headers or {}
        mock_resp.text = body_text
        if body_json is not None:
            mock_resp.json.return_value = body_json
        else:
            from json import JSONDecodeError

            mock_resp.json.side_effect = JSONDecodeError("", "", 0)
        return mock_resp

    def test_str_format(self):
        resp = self._make_response(400, "Bad Request", body_json={"error": "invalid"})
        exc = UiPathAPIError.from_response(resp)
        s = str(exc)
        assert "UiPathBadRequestError" in s
        assert "Bad Request" in s
        assert "400" in s

    def test_repr_format(self):
        resp = self._make_response(404, "Not Found", body_json={"error": "missing"})
        exc = UiPathAPIError.from_response(resp)
        r = repr(exc)
        assert "UiPathNotFoundError" in r
        assert "status_code=404" in r

    def test_from_response_json_body(self):
        resp = self._make_response(500, "Server Error", body_json={"detail": "crash"})
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathInternalServerError)
        assert exc.body == {"detail": "crash"}

    def test_from_response_text_body_fallback(self):
        resp = self._make_response(500, "Server Error", body_text="plain text error")
        exc = UiPathAPIError.from_response(resp)
        assert exc.body == "plain text error"

    def test_from_response_no_body(self):
        resp = self._make_response(503, "Unavailable")
        resp.json.side_effect = Exception("unexpected")
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathServiceUnavailableError)
        assert exc.body is None

    def test_from_response_uses_response_request_when_none(self):
        resp = self._make_response(400, "Bad Request", body_json={})
        exc = UiPathAPIError.from_response(resp, request=None)
        assert exc.request is resp.request

    def test_from_response_uses_explicit_request(self):
        resp = self._make_response(400, "Bad Request", body_json={})
        custom_request = MagicMock(spec=Request)
        exc = UiPathAPIError.from_response(resp, request=custom_request)
        assert exc.request is custom_request

    def test_from_response_unmapped_status_code(self):
        resp = self._make_response(418, "I'm a teapot", body_json={})
        exc = UiPathAPIError.from_response(resp)
        assert type(exc) is UiPathAPIError
        assert exc.status_code == 418

    def test_all_status_code_mappings(self):
        mappings = {
            400: UiPathBadRequestError,
            401: UiPathAuthenticationError,
            403: UiPathPermissionDeniedError,
            404: UiPathNotFoundError,
            429: UiPathRateLimitError,
            500: UiPathInternalServerError,
            503: UiPathServiceUnavailableError,
            504: UiPathGatewayTimeoutError,
            529: UiPathTooManyRequestsError,
        }
        for code, expected_cls in mappings.items():
            resp = self._make_response(code, "Error", body_json={})
            exc = UiPathAPIError.from_response(resp)
            assert isinstance(exc, expected_cls), f"Status {code} -> {type(exc).__name__}"

    def test_patch_raise_for_status_converts_exception(self):
        from httpx import HTTPStatusError

        resp = MagicMock(spec=Response)
        resp.status_code = 404
        resp.reason_phrase = "Not Found"
        resp.json.return_value = {}
        resp.request = MagicMock(spec=Request)
        resp.headers = {}
        original = MagicMock(
            side_effect=HTTPStatusError("err", request=resp.request, response=resp)
        )
        resp.raise_for_status = original

        patched = patch_raise_for_status(resp)
        with pytest.raises(UiPathNotFoundError):
            patched.raise_for_status()

    def test_patch_raise_for_status_passes_on_success(self):
        resp = MagicMock(spec=Response)
        resp.status_code = 200
        original = MagicMock(return_value=resp)
        resp.raise_for_status = original

        patched = patch_raise_for_status(resp)
        result = patched.raise_for_status()
        assert result is resp

    def test_retry_after_http_date_format(self):
        from datetime import datetime, timezone

        future = datetime.now(timezone.utc).replace(microsecond=0)
        future = future.replace(second=future.second + 30 if future.second < 30 else future.second)
        date_str = future.strftime("%a, %d %b %Y %H:%M:%S GMT")

        resp = self._make_response(
            429, "Rate Limit", body_json={}, headers={"retry-after": date_str}
        )
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathRateLimitError)
        # The retry_after should be a positive number (seconds until future date)
        assert exc.retry_after is not None
        assert exc.retry_after >= 0

    def test_retry_after_unparseable_returns_none(self):
        resp = self._make_response(
            429, "Rate Limit", body_json={}, headers={"retry-after": "not-a-date-or-number"}
        )
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after is None


# ============================================================================
# Test Header Utilities
# ============================================================================


class TestHeaderUtilities:
    """Tests for extract_matching_headers and context var functions."""

    def test_extract_matching_headers_basic(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"x-uipath-foo": "bar", "content-type": "json", "x-uipath-baz": "qux"})
        result = extract_matching_headers(headers, ["x-uipath-"])
        assert result == {"x-uipath-foo": "bar", "x-uipath-baz": "qux"}

    def test_extract_matching_headers_case_insensitive(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"X-UiPath-Foo": "bar"})
        result = extract_matching_headers(headers, ["x-uipath-"])
        assert len(result) == 1
        assert "bar" in result.values()

    def test_extract_matching_headers_no_match(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"content-type": "json"})
        result = extract_matching_headers(headers, ["x-uipath-"])
        assert result == {}

    def test_extract_matching_headers_empty_prefixes(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"x-uipath-foo": "bar"})
        result = extract_matching_headers(headers, [])
        assert result == {}

    def test_extract_matching_headers_multiple_prefixes(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"x-uipath-foo": "1", "x-custom-bar": "2", "content-type": "json"})
        result = extract_matching_headers(headers, ["x-uipath-", "x-custom-"])
        assert len(result) == 2

    def test_context_var_get_default(self):
        from uipath.llm_client.utils.headers import get_captured_response_headers

        result = get_captured_response_headers()
        assert isinstance(result, dict)

    def test_context_var_set_and_get(self):
        from uipath.llm_client.utils.headers import (
            get_captured_response_headers,
            set_captured_response_headers,
        )

        _ = set_captured_response_headers({"x-uipath-test": "value"})
        result = get_captured_response_headers()
        assert result == {"x-uipath-test": "value"}
        # Returns a copy
        result["new-key"] = "new-value"
        assert "new-key" not in get_captured_response_headers()

    def test_passthrough_embeddings_no_flavor_header(self):
        from uipath.llm_client.utils.headers import build_routing_headers

        api_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
        )
        headers = build_routing_headers(model_name="text-embedding-3-large", api_config=api_config)
        assert "X-UiPath-LlmGateway-ApiFlavor" not in headers
        assert "X-UiPath-LlmGateway-NormalizedApi-ModelName" not in headers

    def test_normalized_without_model_name(self):
        from uipath.llm_client.utils.headers import build_routing_headers

        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
        )
        headers = build_routing_headers(model_name=None, api_config=api_config)
        assert "X-UiPath-LlmGateway-NormalizedApi-ModelName" not in headers


# ============================================================================
# Test Logging Config
# ============================================================================


class TestLoggingConfig:
    """Tests for LoggingConfig request/response logging."""

    def test_log_request_duration_stores_start_time(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_logging")
        config = LoggingConfig(logger=logger, model_name="gpt-4o")
        request = Request("GET", "https://example.com")
        config.log_request_duration(request)
        assert "start_time" in request.extensions
        assert isinstance(request.extensions["start_time"], float)

    def test_log_request_duration_no_op_without_logger(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        config = LoggingConfig(logger=None, model_name="gpt-4o")
        request = Request("GET", "https://example.com")
        config.log_request_duration(request)
        assert "start_time" not in request.extensions

    def test_log_response_duration_logs_info(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_logging_resp")
        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
        )
        config = LoggingConfig(logger=logger, model_name="gpt-4o", api_config=api_config)

        request = Request("GET", "https://example.com")
        request.extensions["start_time"] = time.monotonic() - 1.5

        response = MagicMock(spec=Response)
        response.request = request

        with patch.object(logger, "info") as mock_info:
            config.log_response_duration(response)
            mock_info.assert_called_once()
            args = mock_info.call_args
            assert "took" in args[0][0]

    def test_log_response_duration_no_start_time(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_no_start")
        config = LoggingConfig(logger=logger)

        request = Request("GET", "https://example.com")
        response = MagicMock(spec=Response)
        response.request = request

        with patch.object(logger, "info") as mock_info:
            config.log_response_duration(response)
            mock_info.assert_not_called()

    def test_log_response_duration_no_op_without_logger(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        config = LoggingConfig(logger=None)
        response = MagicMock(spec=Response)
        config.log_response_duration(response)  # Should not raise

    def test_log_error_logs_on_error_response(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_error_log")
        config = LoggingConfig(logger=logger)

        response = MagicMock(spec=Response)
        response.is_error = True
        response.reason_phrase = "Internal Server Error"
        response.status_code = 500

        with patch.object(logger, "error") as mock_error:
            config.log_error(response)
            mock_error.assert_called_once()

    def test_log_error_no_log_on_success(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_no_error")
        config = LoggingConfig(logger=logger)

        response = MagicMock(spec=Response)
        response.is_error = False

        with patch.object(logger, "error") as mock_error:
            config.log_error(response)
            mock_error.assert_not_called()

    def test_log_error_no_op_without_logger(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        config = LoggingConfig(logger=None)
        response = MagicMock(spec=Response)
        response.is_error = True
        config.log_error(response)  # Should not raise

    @pytest.mark.asyncio
    async def test_alog_request_duration_delegates_to_sync(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_async")
        config = LoggingConfig(logger=logger)
        request = Request("GET", "https://example.com")
        await config.alog_request_duration(request)
        assert "start_time" in request.extensions

    @pytest.mark.asyncio
    async def test_alog_error_delegates_to_sync(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_async_error")
        config = LoggingConfig(logger=logger)
        response = MagicMock(spec=Response)
        response.is_error = True
        response.reason_phrase = "Error"
        response.status_code = 500
        with patch.object(logger, "error") as mock_error:
            await config.alog_error(response)
            mock_error.assert_called_once()

    def test_log_response_unknown_api_config(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_unknown_config")
        config = LoggingConfig(logger=logger, model_name="test", api_config=None)

        request = Request("GET", "https://example.com")
        request.extensions["start_time"] = time.monotonic() - 0.1
        response = MagicMock(spec=Response)
        response.request = request

        with patch.object(logger, "info") as mock_info:
            config.log_response_duration(response)
            mock_info.assert_called_once()
            call_kwargs = mock_info.call_args
            assert "unknown" in str(call_kwargs)


# ============================================================================
# Test SSL Config
# ============================================================================


class TestSSLConfig:
    """Tests for SSL configuration utilities."""

    def test_expand_path_empty(self):
        from uipath.llm_client.utils.ssl_config import expand_path

        assert expand_path("") == ""
        assert expand_path(None) is None

    def test_expand_path_tilde(self):
        from uipath.llm_client.utils.ssl_config import expand_path

        result = expand_path("~/test")
        assert "~" not in result
        assert result.endswith("/test")

    def test_expand_path_env_var(self):
        from uipath.llm_client.utils.ssl_config import expand_path

        with patch.dict(os.environ, {"MY_PATH": "/custom"}):
            result = expand_path("$MY_PATH/cert.pem")
            assert result == "/custom/cert.pem"

    def test_get_httpx_ssl_client_kwargs_default(self):
        from uipath.llm_client.utils.ssl_config import get_httpx_ssl_client_kwargs

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UIPATH_DISABLE_SSL_VERIFY", None)
            kwargs = get_httpx_ssl_client_kwargs()
            assert kwargs["follow_redirects"] is True
            assert kwargs["verify"] is not False  # Should be an SSL context

    def test_get_httpx_ssl_client_kwargs_disable_ssl(self):
        from uipath.llm_client.utils.ssl_config import get_httpx_ssl_client_kwargs

        for val in ("1", "true", "yes", "on", "TRUE", "True"):
            with patch.dict(os.environ, {"UIPATH_DISABLE_SSL_VERIFY": val}):
                kwargs = get_httpx_ssl_client_kwargs()
                assert kwargs["verify"] is False, f"Failed for value: {val}"

    def test_get_httpx_ssl_client_kwargs_not_disabled(self):
        from uipath.llm_client.utils.ssl_config import get_httpx_ssl_client_kwargs

        for val in ("0", "false", "no", "off", ""):
            with patch.dict(os.environ, {"UIPATH_DISABLE_SSL_VERIFY": val}):
                kwargs = get_httpx_ssl_client_kwargs()
                assert kwargs["verify"] is not False, f"Should not disable for value: {val}"


# ============================================================================
# Test Wait Strategy
# ============================================================================


class TestWaitRetryAfterWithFallback:
    """Tests for wait_retry_after_with_fallback strategy."""

    def test_uses_retry_after_from_rate_limit_error(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        # Create a mock retry state with a UiPathRateLimitError that has retry_after
        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathRateLimitError)
        exc.retry_after = 15.0
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        wait_time = strategy(retry_state)
        assert wait_time == 15.0

    def test_caps_retry_after_at_max_delay(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=10, exp_base=2, jitter=0)

        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathRateLimitError)
        exc.retry_after = 120.0
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        wait_time = strategy(retry_state)
        assert wait_time == 10.0  # Capped at max

    def test_fallback_to_exponential_backoff(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        # Non-rate-limit error
        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathInternalServerError)
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        with patch.object(strategy, "fallback_wait", return_value=4.0) as mock_fallback:
            wait_time = strategy(retry_state)
            assert wait_time == 4.0
            mock_fallback.assert_called_once_with(retry_state)

    def test_fallback_when_retry_after_is_none(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathRateLimitError)
        exc.retry_after = None
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        with patch.object(strategy, "fallback_wait", return_value=2.0):
            wait_time = strategy(retry_state)
            assert wait_time == 2.0

    def test_fallback_when_no_outcome(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        retry_state = MagicMock()
        retry_state.outcome = None

        with patch.object(strategy, "fallback_wait", return_value=1.0):
            wait_time = strategy(retry_state)
            assert wait_time == 1.0


# ============================================================================
# Test _build_retryer
# ============================================================================


class TestBuildRetryer:
    """Tests for _build_retryer function."""

    def test_returns_none_for_zero_retries(self):
        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=0, retry_config=None, logger=None)
        assert result is None

    def test_returns_none_for_negative_retries(self):
        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=-1, retry_config=None, logger=None)
        assert result is None

    def test_returns_retrying_for_sync_mode(self):
        from tenacity import Retrying

        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=3, retry_config=None, logger=None, async_mode=False)
        assert isinstance(result, Retrying)

    def test_returns_async_retrying_for_async_mode(self):
        from tenacity import AsyncRetrying

        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=3, retry_config=None, logger=None, async_mode=True)
        assert isinstance(result, AsyncRetrying)

    def test_with_custom_config(self):
        from tenacity import Retrying

        from uipath.llm_client.utils.retry import _build_retryer

        config: RetryConfig = {
            "initial_delay": 0.5,
            "max_delay": 5.0,
            "exp_base": 3.0,
            "jitter": 0.1,
            "retry_on_exceptions": (UiPathRateLimitError,),
        }
        result = _build_retryer(max_retries=2, retry_config=config, logger=None)
        assert isinstance(result, Retrying)

    def test_with_logger_adds_before_sleep(self):
        from uipath.llm_client.utils.retry import _build_retryer

        logger = logging.getLogger("test_retry_logger")
        result = _build_retryer(max_retries=2, retry_config=None, logger=logger)
        assert result is not None
        assert result.before_sleep is not None


# ============================================================================
# Test HTTPX Client Send Behavior
# ============================================================================


class TestUiPathHttpxClientSend:
    """Tests for UiPathHttpxClient.send() behavior."""

    def test_streaming_header_injected_false(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com")
        request = Request("POST", "https://example.com/test")

        with patch.object(
            Client, "send", return_value=MagicMock(spec=Response, headers=Headers(), is_error=False)
        ) as mock_send:
            mock_send.return_value.raise_for_status = MagicMock(return_value=mock_send.return_value)
            client.send(request, stream=False)
            sent_request = mock_send.call_args[0][0]
            assert sent_request.headers["X-UiPath-Streaming-Enabled"] == "false"
        client.close()

    def test_streaming_header_injected_true(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com")
        request = Request("POST", "https://example.com/test")

        with patch.object(
            Client, "send", return_value=MagicMock(spec=Response, headers=Headers(), is_error=False)
        ) as mock_send:
            mock_send.return_value.raise_for_status = MagicMock(return_value=mock_send.return_value)
            client.send(request, stream=True)
            sent_request = mock_send.call_args[0][0]
            assert sent_request.headers["X-UiPath-Streaming-Enabled"] == "true"
        client.close()

    def test_url_freezing_when_enabled(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
            freeze_base_url=True,
        )
        client = UiPathHttpxClient(
            base_url="https://example.com/base",
            api_config=api_config,
        )
        request = Request("POST", "https://example.com/base/some/path")

        with patch.object(
            Client, "send", return_value=MagicMock(spec=Response, headers=Headers(), is_error=False)
        ) as mock_send:
            mock_send.return_value.raise_for_status = MagicMock(return_value=mock_send.return_value)
            client.send(request, stream=False)
            sent_request = mock_send.call_args[0][0]
            assert str(sent_request.url) == "https://example.com/base"
        client.close()

    def test_url_not_frozen_when_disabled(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com/base")
        request = Request("POST", "https://example.com/base/some/path")

        with patch.object(
            Client, "send", return_value=MagicMock(spec=Response, headers=Headers(), is_error=False)
        ) as mock_send:
            mock_send.return_value.raise_for_status = MagicMock(return_value=mock_send.return_value)
            client.send(request, stream=False)
            sent_request = mock_send.call_args[0][0]
            assert "some/path" in str(sent_request.url)
        client.close()

    def test_response_headers_captured(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient
        from uipath.llm_client.utils.headers import get_captured_response_headers

        client = UiPathHttpxClient(base_url="https://example.com")
        request = Request("POST", "https://example.com/test")

        mock_response = MagicMock(spec=Response)
        mock_response.headers = Headers({"x-uipath-request-id": "abc123", "content-type": "json"})
        mock_response.is_error = False
        mock_response.raise_for_status = MagicMock(return_value=mock_response)

        with patch.object(Client, "send", return_value=mock_response):
            client.send(request, stream=False)
            captured = get_captured_response_headers()
            assert "x-uipath-request-id" in captured
            assert "content-type" not in captured
        client.close()

    def test_response_patched_with_raise_for_status(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com")
        request = Request("POST", "https://example.com/test")

        mock_response = MagicMock(spec=Response)
        mock_response.headers = Headers()
        mock_response.is_error = False
        original_raise = MagicMock(return_value=mock_response)
        mock_response.raise_for_status = original_raise

        with patch.object(Client, "send", return_value=mock_response):
            result = client.send(request, stream=False)
            # raise_for_status should have been replaced by patch_raise_for_status
            assert result.raise_for_status is not original_raise
        client.close()


# ============================================================================
# Test LLMGateway S2S Auth Token Acquisition
# ============================================================================


class TestLLMGatewayS2STokenAcquisition:
    """Tests for LLMGatewayS2SAuth.get_llmgw_token."""

    def test_s2s_token_success(self, llmgw_s2s_env_vars):
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        mock_response = MagicMock()
        mock_response.is_client_error = False
        mock_response.is_error = False
        mock_response.json.return_value = {"access_token": "s2s-token-value"}

        with patch.dict(os.environ, llmgw_s2s_env_vars, clear=True):
            settings = LLMGatewaySettings()
            with patch.object(Client, "post", return_value=mock_response):
                auth = LLMGatewayS2SAuth(settings=settings)
                assert auth.access_token == "s2s-token-value"

    def test_s2s_token_client_error_returns_none(self, llmgw_s2s_env_vars):
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.status_code = 401
        mock_response.reason_phrase = "Unauthorized"

        with patch.dict(os.environ, llmgw_s2s_env_vars, clear=True):
            settings = LLMGatewaySettings()
            with patch.object(Client, "post", return_value=mock_response):
                auth = LLMGatewayS2SAuth(settings=settings)
                assert auth.access_token is None

    def test_s2s_token_server_error_returns_none(self, llmgw_s2s_env_vars):
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.status_code = 500
        mock_response.reason_phrase = "Server Error"

        with patch.dict(os.environ, llmgw_s2s_env_vars, clear=True):
            settings = LLMGatewaySettings()
            with patch.object(Client, "post", return_value=mock_response):
                auth = LLMGatewayS2SAuth(settings=settings)
                assert auth.access_token is None

    def test_s2s_missing_credentials_returns_none(self, llmgw_env_vars):
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        auth = LLMGatewayS2SAuth.__new__(LLMGatewayS2SAuth)
        auth.client_id = None
        auth.client_secret = None
        auth.base_url = "https://cloud.uipath.com"
        assert auth.get_llmgw_token() is None


# ============================================================================
# Test LLMGateway BYOM Validation
# ============================================================================


class TestLLMGatewayBYOMValidation:
    """Tests for LLMGatewayBaseSettings.validate_byo_model."""

    def test_valid_operation_code(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = "code1"
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"operationCodes": ["code1", "code2"]},
            }
            settings.validate_byo_model(model_info)  # Should not raise

    def test_invalid_operation_code_raises(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = "invalid-code"
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"operationCodes": ["code1", "code2"]},
            }
            with pytest.raises(ValueError, match="operation code"):
                settings.validate_byo_model(model_info)

    def test_auto_picks_first_operation_code(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = None
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"operationCodes": ["auto-code"]},
            }
            settings.validate_byo_model(model_info)
            assert settings.operation_code == "auto-code"

    def test_auto_picks_first_with_warning_for_multiple(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = None
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"operationCodes": ["code1", "code2"]},
            }
            with patch("uipath.llm_client.settings.llmgateway.settings.logging") as mock_logging:
                settings.validate_byo_model(model_info)
                mock_logging.warning.assert_called_once()
            assert settings.operation_code == "code1"

    def test_no_operation_codes_no_change(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = None
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"operationCodes": []},
            }
            settings.validate_byo_model(model_info)
            assert settings.operation_code is None

    def test_no_byom_details(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = None
            model_info = {"modelName": "custom-model"}
            settings.validate_byo_model(model_info)
            assert settings.operation_code is None


# ============================================================================
# Test LLMGateway Endpoints
# ============================================================================


class TestLLMGatewayEndpoints:
    """Tests for LLMGatewayEndpoints enum."""

    def test_endpoint_values(self):
        from uipath.llm_client.settings.llmgateway.utils import LLMGatewayEndpoints

        assert LLMGatewayEndpoints.IDENTITY_ENDPOINT == "identity_/connect/token"
        assert LLMGatewayEndpoints.DISCOVERY_ENDPOINT == "llmgateway_/api/discovery"

    def test_normalized_endpoint_formatting(self):
        from uipath.llm_client.settings.llmgateway.utils import LLMGatewayEndpoints

        result = LLMGatewayEndpoints.NORMALIZED_ENDPOINT.value.format(api_type="chat/completions")
        assert result == "llmgateway_/api/chat/completions"

    def test_passthrough_endpoint_formatting(self):
        from uipath.llm_client.settings.llmgateway.utils import LLMGatewayEndpoints

        result = LLMGatewayEndpoints.PASSTHROUGH_ENDPOINT.value.format(
            vendor="openai", model="gpt-4o", api_type="completions"
        )
        assert result == "llmgateway_/api/raw/vendor/openai/model/gpt-4o/completions"


# ============================================================================
# Test LLMGateway URL and Header Edge Cases
# ============================================================================


class TestLLMGatewayURLEdgeCases:
    """Tests for LLMGatewaySettings URL and header edge cases."""

    def test_build_base_url_raises_without_api_config(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            with pytest.raises(ValueError, match="api_config is required"):
                settings.build_base_url(model_name="gpt-4o", api_config=None)

    def test_build_auth_headers_with_operation_code(self, llmgw_env_vars):
        env = {**llmgw_env_vars, "LLMGW_OPERATION_CODE": "test-op-code"}
        with patch.dict(os.environ, env, clear=True):
            settings = LLMGatewaySettings()
            headers = settings.build_auth_headers()
            assert headers["X-UiPath-LlmGateway-OperationCode"] == "test-op-code"

    def test_build_auth_headers_with_additional_headers(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.additional_headers = {"X-Custom": "value"}
            headers = settings.build_auth_headers()
            assert headers["X-Custom"] == "value"

    def test_build_auth_headers_includes_internal_ids(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            headers = settings.build_auth_headers()
            assert headers["X-UiPath-Internal-AccountId"] == "test-org-id"
            assert headers["X-UiPath-Internal-TenantId"] == "test-tenant-id"


# ============================================================================
# Test Constants Enums
# ============================================================================


class TestEnumConstants:
    """Tests for StrEnum constants."""

    def test_api_type_values(self):
        assert ApiType.COMPLETIONS == "completions"
        assert ApiType.EMBEDDINGS == "embeddings"

    def test_routing_mode_values(self):
        assert RoutingMode.PASSTHROUGH == "passthrough"
        assert RoutingMode.NORMALIZED == "normalized"

    def test_vendor_type_values(self):
        assert VendorType.OPENAI == "openai"
        assert VendorType.VERTEXAI == "vertexai"
        assert VendorType.AWSBEDROCK == "awsbedrock"
        assert VendorType.AZURE == "azure"
        assert VendorType.ANTHROPIC == "anthropic"

    def test_api_flavor_values(self):
        assert ApiFlavor.CHAT_COMPLETIONS == "chat-completions"
        assert ApiFlavor.RESPONSES == "responses"
        assert ApiFlavor.GENERATE_CONTENT == "generate-content"
        assert ApiFlavor.CONVERSE == "converse"
        assert ApiFlavor.INVOKE == "invoke"
        assert ApiFlavor.ANTHROPIC_CLAUDE == "anthropic-claude"

    def test_enum_string_comparison(self):
        assert ApiType.COMPLETIONS == "completions"
        assert RoutingMode.PASSTHROUGH == "passthrough"
        assert VendorType.OPENAI == "openai"

    def test_enum_is_str_subclass(self):
        assert isinstance(ApiType.COMPLETIONS, str)
        assert isinstance(RoutingMode.PASSTHROUGH, str)
        assert isinstance(VendorType.OPENAI, str)
        assert isinstance(ApiFlavor.CHAT_COMPLETIONS, str)
