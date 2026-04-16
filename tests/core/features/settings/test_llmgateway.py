"""Tests for LLMGatewaySettings."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import Client, Request, Response

from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.utils.exceptions import UiPathAPIError, UiPathAuthenticationError


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


class TestLLMGatewayDiscoveryCache:
    """Tests for get_available_models TTL caching."""

    def test_second_call_returns_cached_result(self, llmgw_env_vars):
        """Second call within TTL should not hit the endpoint again."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

            mock_response = MagicMock()
            mock_response.is_error = False
            mock_response.json.return_value = [{"modelName": "gpt-4o", "vendor": "openai"}]

            with patch.object(Client, "get", return_value=mock_response) as mock_get:
                first = settings.get_available_models()
                second = settings.get_available_models()
                assert first == second
                mock_get.assert_called_once()

    def test_cache_expires_after_ttl(self, llmgw_env_vars):
        """After TTL expires, the endpoint should be called again."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

            mock_response = MagicMock()
            mock_response.is_error = False
            mock_response.json.return_value = [{"modelName": "gpt-4o", "vendor": "openai"}]

            with patch.object(Client, "get", return_value=mock_response) as mock_get:
                settings.get_available_models()
                assert mock_get.call_count == 1

                # Simulate TTL expiry by rewinding the cache timestamp
                settings._models_cache_timestamp = (
                    time.monotonic() - settings.DISCOVERY_CACHE_TTL_SECONDS - 1
                )

                settings.get_available_models()
                assert mock_get.call_count == 2

    def test_cache_is_per_instance(self, llmgw_env_vars):
        """Each settings instance should have its own independent cache."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings1 = LLMGatewaySettings()
            settings2 = LLMGatewaySettings()

            mock_response = MagicMock()
            mock_response.is_error = False
            mock_response.json.return_value = [{"modelName": "gpt-4o", "vendor": "openai"}]

            with patch.object(Client, "get", return_value=mock_response) as mock_get:
                settings1.get_available_models()
                settings2.get_available_models()
                assert mock_get.call_count == 2


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

    def test_auth_singleton_reuses_instance_for_same_settings(self, llmgw_env_vars):
        """Test that LLMGatewayS2SAuth reuses the same instance for identical settings."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            auth1 = LLMGatewayS2SAuth(settings=settings)
            auth2 = LLMGatewayS2SAuth(settings=settings)
            assert auth1 is auth2

    def test_auth_creates_separate_instances_for_different_settings(self, llmgw_env_vars):
        """Test that LLMGatewayS2SAuth creates separate instances for different credentials."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        env1 = {**llmgw_env_vars, "LLMGW_CLIENT_ID": "id-a", "LLMGW_CLIENT_SECRET": "secret-a"}
        env2 = {**llmgw_env_vars, "LLMGW_CLIENT_ID": "id-b", "LLMGW_CLIENT_SECRET": "secret-b"}
        with patch.dict(os.environ, env1, clear=True):
            settings1 = LLMGatewaySettings()
        with patch.dict(os.environ, env2, clear=True):
            settings2 = LLMGatewaySettings()
        auth1 = LLMGatewayS2SAuth(settings=settings1)
        auth2 = LLMGatewayS2SAuth(settings=settings2)
        assert auth1 is not auth2


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


class TestLLMGatewayBYOMValidation:
    """Tests for LLMGatewayBaseSettings.validate_byo_model."""

    def test_valid_operation_code(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = "code1"
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"availableOperationCodes": ["code1", "code2"]},
            }
            settings.validate_byo_model(model_info)  # Should not raise

    def test_invalid_operation_code_raises(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = "invalid-code"
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"availableOperationCodes": ["code1", "code2"]},
            }
            with pytest.raises(ValueError, match="operation code"):
                settings.validate_byo_model(model_info)

    def test_auto_picks_first_operation_code(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = None
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"availableOperationCodes": ["auto-code"]},
            }
            settings.validate_byo_model(model_info)
            assert settings.operation_code == "auto-code"

    def test_auto_picks_first_with_warning_for_multiple(self, llmgw_env_vars):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            settings.operation_code = None
            model_info = {
                "modelName": "custom-model",
                "byomDetails": {"availableOperationCodes": ["code1", "code2"]},
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
                "byomDetails": {"availableOperationCodes": []},
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


class TestLLMGatewaySingletonCacheKey:
    """Tests for LLMGatewayS2SAuth._singleton_cache_key including base_url."""

    def test_different_base_urls_produce_different_cache_keys(self, llmgw_env_vars):
        """Different base_urls with same credentials produce different cache keys."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        env1 = {**llmgw_env_vars, "LLMGW_URL": "https://alpha.uipath.com"}
        env2 = {**llmgw_env_vars, "LLMGW_URL": "https://beta.uipath.com"}

        with patch.dict(os.environ, env1, clear=True):
            settings1 = LLMGatewaySettings()
        with patch.dict(os.environ, env2, clear=True):
            settings2 = LLMGatewaySettings()

        key1 = LLMGatewayS2SAuth._singleton_cache_key(settings1)
        key2 = LLMGatewayS2SAuth._singleton_cache_key(settings2)
        assert key1 != key2

    def test_same_base_url_and_credentials_produce_same_cache_key(self, llmgw_env_vars):
        """Same base_url and credentials produce identical cache keys."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings1 = LLMGatewaySettings()
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings2 = LLMGatewaySettings()

        key1 = LLMGatewayS2SAuth._singleton_cache_key(settings1)
        key2 = LLMGatewayS2SAuth._singleton_cache_key(settings2)
        assert key1 == key2

    def test_cache_key_includes_base_url(self, llmgw_env_vars):
        """The cache key tuple contains the base_url as its first element."""
        from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

        key = LLMGatewayS2SAuth._singleton_cache_key(settings)
        assert key[0] == settings.base_url
