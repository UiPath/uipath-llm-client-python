"""Tests for uipath-llm-client core functionality.

This module tests:
1. Retry logic (RetryConfig, RetryableHTTPTransport, RetryableAsyncHTTPTransport)
2. AgentHubSettings (build_base_url, build_auth_headers, build_auth_pipeline, get_available_models)
3. LLMGatewaySettings (build_base_url, build_auth_headers, build_auth_pipeline, get_available_models)
4. Settings factory (get_default_client_settings) with environment variables
5. HTTPX client functionality (UiPathHttpxClient, UiPathHttpxAsyncClient)
6. Auth refresh logic for both settings
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from httpx import Client, Request, Response

from uipath_llm_client.settings import (
    AgentHubSettings,
    LLMGatewaySettings,
    UiPathAPIConfig,
    get_default_client_settings,
)
from uipath_llm_client.settings.utils import SingletonMeta
from uipath_llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathAuthenticationError,
    UiPathRateLimitError,
)
from uipath_llm_client.utils.retry import (
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
def agenthub_env_vars():
    """Environment variables for AgentHubSettings."""
    return {
        "UIPATH_ACCESS_TOKEN": "test-access-token",
        "UIPATH_URL": "https://cloud.uipath.com/org/tenant",
        "UIPATH_TENANT_ID": "test-tenant-id",
        "UIPATH_ORGANIZATION_ID": "test-org-id",
    }


@pytest.fixture
def passthrough_api_config():
    """API config for passthrough mode."""
    return UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="openai",
    )


@pytest.fixture
def normalized_api_config():
    """API config for normalized mode."""
    return UiPathAPIConfig(
        api_type="completions",
        client_type="normalized",
    )


@pytest.fixture
def embeddings_api_config():
    """API config for embeddings."""
    return UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
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
                api_type="completions",
                client_type="passthrough",
                vendor_type=None,
            )

    def test_normalized_does_not_require_vendor_type(self):
        """Test that normalized mode doesn't require vendor_type."""
        config = UiPathAPIConfig(
            api_type="completions",
            client_type="normalized",
        )
        assert config.vendor_type is None

    def test_passthrough_with_vendor_type(self):
        """Test passthrough config with vendor_type."""
        config = UiPathAPIConfig(
            api_type="completions",
            client_type="passthrough",
            vendor_type="openai",
        )
        assert config.api_type == "completions"
        assert config.client_type == "passthrough"
        assert config.vendor_type == "openai"

    def test_freeze_base_url_default(self):
        """Test freeze_base_url defaults to False."""
        config = UiPathAPIConfig(
            api_type="completions",
            client_type="normalized",
        )
        assert config.freeze_base_url is False

    def test_api_flavor_and_version(self):
        """Test api_flavor and api_version can be set."""
        config = UiPathAPIConfig(
            api_type="completions",
            client_type="passthrough",
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

    def test_default_returns_agenthub(self, agenthub_env_vars):
        """Test that default backend is agenthub."""
        with patch.dict(os.environ, agenthub_env_vars, clear=False):
            # Remove UIPATH_LLM_BACKEND if set
            env = {**agenthub_env_vars}
            env.pop("UIPATH_LLM_BACKEND", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                    settings = get_default_client_settings()
                    assert isinstance(settings, AgentHubSettings)

    def test_explicit_agenthub(self, agenthub_env_vars):
        """Test explicit agenthub backend."""
        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = get_default_client_settings(backend="agenthub")
                assert isinstance(settings, AgentHubSettings)

    def test_explicit_llmgateway(self, llmgw_env_vars):
        """Test explicit llmgateway backend."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = get_default_client_settings(backend="llmgateway")
            assert isinstance(settings, LLMGatewaySettings)

    def test_env_var_agenthub(self, agenthub_env_vars):
        """Test UIPATH_LLM_BACKEND=agenthub from environment."""
        env = {**agenthub_env_vars, "UIPATH_LLM_BACKEND": "agenthub"}
        with patch.dict(os.environ, env, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = get_default_client_settings()
                assert isinstance(settings, AgentHubSettings)

    def test_env_var_llmgateway(self, llmgw_env_vars):
        """Test UIPATH_LLM_BACKEND=llmgateway from environment."""
        env = {**llmgw_env_vars, "UIPATH_LLM_BACKEND": "llmgateway"}
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
        from uipath_llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

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
        """Test get_available_models returns a list of models."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

            # Mock the HTTP request since this is a unit test
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {"modelName": "gpt-4o", "vendor": "openai"},
                {"modelName": "claude-3-opus", "vendor": "anthropic"},
            ]

            with patch.object(Client, "get", return_value=mock_response):
                models = settings.get_available_models()
                assert isinstance(models, list)
                assert len(models) == 2


# ============================================================================
# Test LLMGateway Auth Refresh Logic
# ============================================================================


class TestLLMGatewayAuthRefresh:
    """Tests for LLMGatewayS2SAuth token refresh logic."""

    def test_auth_flow_adds_bearer_token(self, llmgw_env_vars):
        """Test auth_flow adds Authorization header."""
        from uipath_llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

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
        from uipath_llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_s2s_env_vars, clear=True):
            settings = LLMGatewaySettings()

            # Mock the token retrieval
            with patch.object(
                LLMGatewayS2SAuth, "get_llmgw_token_header", return_value="new-token"
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
        from uipath_llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            auth1 = LLMGatewayS2SAuth(settings=settings)
            auth2 = LLMGatewayS2SAuth(settings=settings)
            assert auth1 is auth2


# ============================================================================
# Test AgentHubSettings
# ============================================================================


class TestAgentHubSettings:
    """Tests for AgentHubSettings."""

    def test_build_base_url_passthrough(self, agenthub_env_vars, passthrough_api_config):
        """Test build_base_url for passthrough mode."""
        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                url = settings.build_base_url(
                    model_name="gpt-4o",
                    api_config=passthrough_api_config,
                )
                assert "agenthub_/llm/raw/vendor/openai/model/gpt-4o/completions" in url

    def test_build_base_url_normalized(self, agenthub_env_vars, normalized_api_config):
        """Test build_base_url for normalized mode."""
        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                url = settings.build_base_url(
                    model_name="gpt-4o",
                    api_config=normalized_api_config,
                )
                assert "agenthub_/llm/api/chat/completions" in url

    def test_build_auth_headers_empty_by_default(self, agenthub_env_vars):
        """Test build_auth_headers returns empty dict by default."""
        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                headers = settings.build_auth_headers()
                assert headers == {}

    def test_build_auth_headers_with_tracing(self, agenthub_env_vars):
        """Test build_auth_headers includes tracing headers when set."""
        env = {
            **agenthub_env_vars,
            "UIPATH_AGENTHUB_CONFIG": "test-config",
            "UIPATH_PROCESS_KEY": "test-process",
            "UIPATH_JOB_KEY": "test-job",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                headers = settings.build_auth_headers()
                assert headers["X-UiPath-AgentHub-Config"] == "test-config"
                assert headers["X-UiPath-ProcessKey"] == "test-process"
                assert headers["X-UiPath-JobKey"] == "test-job"

    def test_build_auth_pipeline_returns_auth(self, agenthub_env_vars):
        """Test build_auth_pipeline returns an Auth instance."""
        from httpx import Auth

        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                auth = settings.build_auth_pipeline()
                assert isinstance(auth, Auth)

    def test_check_credentials(self, agenthub_env_vars):
        """Test check_credentials returns True when all credentials present."""
        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                assert settings.check_credentials() is True


# ============================================================================
# Test AgentHub Auth Refresh Logic
# ============================================================================


class TestAgentHubAuthRefresh:
    """Tests for AgentHubAuth token refresh logic."""

    def test_auth_flow_adds_bearer_token(self, agenthub_env_vars):
        """Test auth_flow adds Authorization header."""
        from uipath_llm_client.settings.agenthub.auth import AgentHubAuth

        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                auth = AgentHubAuth(settings=settings)
                request = Request("GET", "https://example.com")
                flow = auth.auth_flow(request)
                modified_request = next(flow)
                assert "Authorization" in modified_request.headers
                assert modified_request.headers["Authorization"] == "Bearer test-access-token"

    def test_auth_flow_refreshes_on_401(self, agenthub_env_vars):
        """Test auth_flow refreshes token on 401 response."""
        from uipath_llm_client.settings.agenthub.auth import AgentHubAuth

        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()

                # Mock _get_access_token to return a new token on refresh
                with patch.object(
                    AgentHubAuth, "_get_access_token", return_value="initial-token"
                ):
                    auth = AgentHubAuth(settings=settings)

                request = Request("GET", "https://example.com")
                flow = auth.auth_flow(request)

                # First yield - initial request
                modified_request = next(flow)
                assert "Bearer" in modified_request.headers["Authorization"]

                # Simulate 401 response
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 401

                # Mock the _get_access_token method to return a new token
                with patch.object(
                    AgentHubAuth, "_get_access_token", return_value="refreshed-token"
                ):
                    try:
                        retry_request = flow.send(mock_response)
                        assert retry_request.headers["Authorization"] == "Bearer refreshed-token"
                    except StopIteration:
                        pass

    def test_auth_singleton_reuses_instance(self, agenthub_env_vars):
        """Test that AgentHubAuth is a singleton."""
        from uipath_llm_client.settings.agenthub.auth import AgentHubAuth

        with patch.dict(os.environ, agenthub_env_vars, clear=True):
            with patch("uipath_llm_client.settings.agenthub.settings.AuthService"):
                settings = AgentHubSettings()
                auth1 = AgentHubAuth(settings=settings)
                auth2 = AgentHubAuth(settings=settings)
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

    def test_transport_no_retry_when_max_retries_1(self):
        """Test no retry logic when max_retries is 1."""
        transport = RetryableHTTPTransport(max_retries=1)
        assert transport.retryer is None

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

    def test_async_transport_no_retry_when_max_retries_1(self):
        """Test no retry logic when max_retries is 1."""
        transport = RetryableAsyncHTTPTransport(max_retries=1)
        assert transport.retryer is None

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
        from uipath_llm_client.httpx_client import UiPathHttpxClient

        assert issubclass(UiPathHttpxClient, Client)

    def test_client_has_default_headers(self):
        """Test client has default UiPath headers."""
        from uipath_llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com")
        assert "X-UiPath-LLMGateway-TimeoutSeconds" in client.headers
        assert "X-UiPath-LLMGateway-AllowFull4xxResponse" in client.headers
        client.close()

    def test_client_merges_custom_headers(self):
        """Test client merges custom headers with defaults."""
        from uipath_llm_client.httpx_client import UiPathHttpxClient

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
        from uipath_llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            model_name="gpt-4o",
        )
        assert client.model_name == "gpt-4o"
        client.close()

    def test_client_with_api_config(self, normalized_api_config):
        """Test client stores api_config."""
        from uipath_llm_client.httpx_client import UiPathHttpxClient

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
        from uipath_llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            max_retries=3,
        )
        # Transport should be RetryableHTTPTransport
        assert isinstance(client._transport, RetryableHTTPTransport)
        client.close()

    def test_client_with_byo_connection_id(self):
        """Test client adds BYO connection ID header."""
        from uipath_llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            byo_connection_id="test-connection-id",
        )
        assert "X-UiPath-LlmGateway-ByoConnectionId" in client.headers
        assert client.headers["X-UiPath-LlmGateway-ByoConnectionId"] == "test-connection-id"
        client.close()


class TestUiPathHttpxAsyncClient:
    """Tests for UiPathHttpxAsyncClient."""

    def test_async_client_inherits_from_httpx_async_client(self):
        """Test async client inherits from httpx.AsyncClient."""
        from httpx import AsyncClient

        from uipath_llm_client.httpx_client import UiPathHttpxAsyncClient

        assert issubclass(UiPathHttpxAsyncClient, AsyncClient)

    def test_async_client_has_default_headers(self):
        """Test async client has default UiPath headers."""
        from uipath_llm_client.httpx_client import UiPathHttpxAsyncClient

        client = UiPathHttpxAsyncClient(base_url="https://example.com")
        assert "X-UiPath-LLMGateway-TimeoutSeconds" in client.headers
        assert "X-UiPath-LLMGateway-AllowFull4xxResponse" in client.headers

    def test_async_client_with_retry_config(self):
        """Test async client creates retryable async transport."""
        from uipath_llm_client.httpx_client import UiPathHttpxAsyncClient

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
        from uipath_llm_client.httpx_client import build_routing_headers

        headers = build_routing_headers()
        assert headers == {}

    def test_normalized_api_header(self, normalized_api_config):
        """Test normalized API adds model name header."""
        from uipath_llm_client.httpx_client import build_routing_headers

        headers = build_routing_headers(
            model_name="gpt-4o",
            api_config=normalized_api_config,
        )
        assert headers["X-UiPath-LlmGateway-NormalizedApi-ModelName"] == "gpt-4o"

    def test_passthrough_api_headers(self):
        """Test passthrough API adds flavor and version headers when set."""
        from uipath_llm_client.httpx_client import build_routing_headers

        api_config = UiPathAPIConfig(
            api_type="completions",
            client_type="passthrough",
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
        from uipath_llm_client.httpx_client import build_routing_headers

        headers = build_routing_headers(
            byo_connection_id="test-connection-id",
        )
        assert headers["X-UiPath-LlmGateway-ByoConnectionId"] == "test-connection-id"


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
        from uipath_llm_client.utils.exceptions import (
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

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.status_code == 429


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
