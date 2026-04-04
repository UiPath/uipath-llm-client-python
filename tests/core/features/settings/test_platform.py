"""Tests for PlatformSettings."""

import os
from unittest.mock import MagicMock, patch

import pytest
from httpx import Request, Response

from uipath.llm_client.settings import PlatformSettings, UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode


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
        """Test build_base_url raises ValueError when model_name is None."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            with pytest.raises(ValueError, match="model_name is required"):
                settings.build_base_url(model_name=None, api_config=normalized_api_config)

    def test_build_base_url_requires_api_config(self, platform_env_vars, mock_platform_auth):
        """Test build_base_url raises ValueError when api_config is None."""
        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            with pytest.raises(ValueError, match="api_config is required"):
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


class TestPlatformAuthRefresh:
    """Tests for PlatformAuth token refresh logic."""

    @pytest.fixture(autouse=True)
    def clear_auth_singleton(self):
        """Clear PlatformAuth singleton before each test."""
        from uipath.llm_client.settings.utils import SingletonMeta

        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

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

    def test_auth_singleton_reuses_instance_for_same_settings(
        self, platform_env_vars, mock_platform_auth
    ):
        """Test that PlatformAuth reuses the same instance for identical settings."""
        from uipath.llm_client.settings.platform.auth import PlatformAuth

        with patch.dict(os.environ, platform_env_vars, clear=True):
            settings = PlatformSettings()
            auth1 = PlatformAuth(settings=settings)
            auth2 = PlatformAuth(settings=settings)
            assert auth1 is auth2

    def test_auth_creates_separate_instances_for_different_settings(
        self, platform_env_vars, mock_platform_auth
    ):
        """Test that PlatformAuth creates separate instances for different credentials."""
        from uipath.llm_client.settings.platform.auth import PlatformAuth

        env1 = {**platform_env_vars, "UIPATH_ACCESS_TOKEN": "token-x"}
        env2 = {**platform_env_vars, "UIPATH_ACCESS_TOKEN": "token-y"}
        with patch.dict(os.environ, env1, clear=True):
            settings1 = PlatformSettings()
        with patch.dict(os.environ, env2, clear=True):
            settings2 = PlatformSettings()
        auth1 = PlatformAuth(settings=settings1)
        auth2 = PlatformAuth(settings=settings2)
        assert auth1 is not auth2
