"""Tests for LLMGatewaySettings."""

import os
from unittest.mock import MagicMock, patch

import pytest
from httpx import Client, Request, Response

from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.base import UiPathBaseSettings
from uipath.llm_client.utils.exceptions import UiPathAPIError, UiPathAuthenticationError


class TestLLMGatewaySettings:
    """Tests for LLMGatewaySettings."""

    @pytest.fixture(autouse=True)
    def _clear_discovery_cache(self):
        UiPathBaseSettings._discovery_cache.clear()

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


class TestDiscoveryCache:
    """Tests for get_available_models caching behavior."""

    @pytest.fixture(autouse=True)
    def _clear_discovery_cache(self):
        UiPathBaseSettings._discovery_cache.clear()

    def test_second_call_returns_cached_result(self, llmgw_env_vars):
        """Second call should not hit the network."""
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

    def test_refresh_bypasses_cache(self, llmgw_env_vars):
        """refresh=True should fetch again even if cached."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()

            mock_response = MagicMock()
            mock_response.is_error = False
            mock_response.json.return_value = [{"modelName": "gpt-4o", "vendor": "openai"}]

            with patch.object(Client, "get", return_value=mock_response) as mock_get:
                settings.get_available_models()
                settings.get_available_models(refresh=True)

                assert mock_get.call_count == 2

    def test_different_settings_have_separate_caches(self, llmgw_env_vars):
        """Different cache keys should not share cached results."""
        env1 = {**llmgw_env_vars, "LLMGW_REQUESTING_PRODUCT": "product-a"}
        env2 = {**llmgw_env_vars, "LLMGW_REQUESTING_PRODUCT": "product-b"}

        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = [{"modelName": "gpt-4o", "vendor": "openai"}]

        with patch.object(Client, "get", return_value=mock_response) as mock_get:
            with patch.dict(os.environ, env1, clear=True):
                settings1 = LLMGatewaySettings()
                settings1.get_available_models()

            with patch.dict(os.environ, env2, clear=True):
                settings2 = LLMGatewaySettings()
                settings2.get_available_models()

            assert mock_get.call_count == 2

    def test_cache_key_includes_requesting_product(self, llmgw_env_vars):
        """LLMGateway cache key should include requesting_product."""
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
            key = settings._discovery_cache_key()
            assert settings.requesting_product in key


class TestGetModelInfo:
    """Tests for UiPathBaseSettings.get_model_info."""

    @pytest.fixture(autouse=True)
    def _clear_discovery_cache(self):
        UiPathBaseSettings._discovery_cache.clear()

    _MODELS = [
        {"modelName": "gpt-4o", "vendor": "OpenAi", "modelSubscriptionType": "UiPathOwned"},
        {
            "modelName": "gpt-4o",
            "vendor": "OpenAi",
            "modelSubscriptionType": "BYO",
            "byomDetails": {
                "integrationServiceConnectionId": "conn-123",
                "availableOperationCodes": ["op1"],
            },
        },
        {
            "modelName": "claude-3-opus",
            "vendor": "Anthropic",
            "modelSubscriptionType": "UiPathOwned",
        },
        {
            "modelName": "gemini-2.0-flash",
            "vendor": "VertexAi",
            "modelSubscriptionType": "UiPathOwned",
        },
    ]

    def _make_settings(self, llmgw_env_vars, models=None):
        with patch.dict(os.environ, llmgw_env_vars, clear=True):
            settings = LLMGatewaySettings()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = models if models is not None else self._MODELS
        # Pre-populate the cache so we don't need to mock Client.get on every call
        with patch.object(Client, "get", return_value=mock_response):
            settings.get_available_models()
        return settings

    def test_finds_model_by_name(self, llmgw_env_vars):
        settings = self._make_settings(llmgw_env_vars)
        info = settings.get_model_info("claude-3-opus")
        assert info["modelName"] == "claude-3-opus"

    def test_case_insensitive_lookup(self, llmgw_env_vars):
        settings = self._make_settings(llmgw_env_vars)
        info = settings.get_model_info("Claude-3-Opus")
        assert info["modelName"] == "claude-3-opus"

    def test_raises_on_unknown_model(self, llmgw_env_vars):
        settings = self._make_settings(llmgw_env_vars)
        with pytest.raises(ValueError, match="not found"):
            settings.get_model_info("nonexistent-model")

    def test_filters_by_vendor_type(self, llmgw_env_vars):
        models = [
            {
                "modelName": "shared-name",
                "vendor": "OpenAi",
                "modelSubscriptionType": "UiPathOwned",
            },
            {
                "modelName": "shared-name",
                "vendor": "Anthropic",
                "modelSubscriptionType": "UiPathOwned",
            },
        ]
        settings = self._make_settings(llmgw_env_vars, models=models)
        info = settings.get_model_info("shared-name", vendor_type="anthropic")
        assert info["vendor"] == "Anthropic"

    def test_filters_by_byo_connection_id(self, llmgw_env_vars):
        settings = self._make_settings(llmgw_env_vars)
        info = settings.get_model_info("gpt-4o", byo_connection_id="conn-123")
        assert info["modelSubscriptionType"] == "BYO"

    def test_prefers_uipath_owned_when_no_byo_id(self, llmgw_env_vars):
        """When multiple matches exist and no byo_connection_id, prefer UiPathOwned."""
        settings = self._make_settings(llmgw_env_vars)
        info = settings.get_model_info("gpt-4o")
        assert info["modelSubscriptionType"] == "UiPathOwned"

    def test_calls_validate_byo_model_for_non_uipath_owned(self, llmgw_env_vars):
        """get_model_info should call validate_byo_model for BYO models."""
        settings = self._make_settings(llmgw_env_vars)
        with patch.object(settings, "validate_byo_model") as mock_validate:
            settings.get_model_info("gpt-4o", byo_connection_id="conn-123")
            mock_validate.assert_called_once()

    def test_skips_validate_byo_model_for_uipath_owned(self, llmgw_env_vars):
        """get_model_info should not call validate_byo_model for UiPath-owned models."""
        settings = self._make_settings(llmgw_env_vars)
        with patch.object(settings, "validate_byo_model") as mock_validate:
            settings.get_model_info("claude-3-opus")
            mock_validate.assert_not_called()

    def test_prefers_responses_when_both_openai_flavors_available(self, llmgw_env_vars):
        """When OpenAI discovery returns both chat-completions and responses entries,
        get_model_info returns the responses one."""
        models = [
            {
                "modelName": "custom-gpt",
                "vendor": "OpenAi",
                "apiFlavor": "OpenAiChatCompletions",
                "modelSubscriptionType": "BYO",
                "byomDetails": {
                    "integrationServiceConnectionId": "conn-1",
                    "availableOperationCodes": ["op1"],
                },
            },
            {
                "modelName": "custom-gpt",
                "vendor": "OpenAi",
                "apiFlavor": "OpenAiResponses",
                "modelSubscriptionType": "BYO",
                "byomDetails": {
                    "integrationServiceConnectionId": "conn-1",
                    "availableOperationCodes": ["op1"],
                },
            },
        ]
        settings = self._make_settings(llmgw_env_vars, models=models)
        info = settings.get_model_info("custom-gpt", byo_connection_id="conn-1")
        assert info["apiFlavor"] == "OpenAiResponses"

    def test_prefers_responses_with_plain_apiflavor_strings(self, llmgw_env_vars):
        """Tie-break also recognises the routing-form apiFlavor values."""
        models = [
            {
                "modelName": "gpt-x",
                "vendor": "OpenAi",
                "apiFlavor": "chat-completions",
                "modelSubscriptionType": "UiPathOwned",
            },
            {
                "modelName": "gpt-x",
                "vendor": "OpenAi",
                "apiFlavor": "responses",
                "modelSubscriptionType": "UiPathOwned",
            },
        ]
        settings = self._make_settings(llmgw_env_vars, models=models)
        info = settings.get_model_info("gpt-x")
        assert info["apiFlavor"] == "responses"

    def test_no_responses_preference_for_non_openai(self, llmgw_env_vars):
        """The responses preference should not fire for non-OpenAI vendors."""
        models = [
            {
                "modelName": "claude-x",
                "vendor": "Anthropic",
                "apiFlavor": "anthropic-claude",
                "modelSubscriptionType": "UiPathOwned",
            },
            {
                "modelName": "claude-x",
                "vendor": "Anthropic",
                "apiFlavor": "converse",
                "modelSubscriptionType": "UiPathOwned",
            },
        ]
        settings = self._make_settings(llmgw_env_vars, models=models)
        info = settings.get_model_info("claude-x")
        # First entry wins (no preference logic for Anthropic)
        assert info["apiFlavor"] == "anthropic-claude"
