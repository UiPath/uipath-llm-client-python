"""Tests for UiPathGoogle client initialization and API config."""

from unittest.mock import MagicMock, PropertyMock, patch

import httpx

from uipath.llm_client.settings import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType

# ============================================================================
# Test Google API Config
# ============================================================================


class TestBuildApiConfig:
    """Tests for the API config built inside UiPathGoogle.__init__."""

    def test_default_api_config_fields(self):
        """The api_config created in __init__ has the expected constant values."""
        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type=VendorType.VERTEXAI,
            api_flavor=ApiFlavor.GENERATE_CONTENT,
            api_version="v1beta1",
            freeze_base_url=True,
        )
        assert api_config.api_type == "completions"
        assert api_config.routing_mode == "passthrough"
        assert api_config.vendor_type == "vertexai"
        assert api_config.api_flavor == "generate-content"
        assert api_config.api_version == "v1beta1"
        assert api_config.freeze_base_url is True


# ============================================================================
# Test UiPathGoogle Initialization
# ============================================================================


def _make_mock_sync_client():
    """Create a mock sync httpx client that passes pydantic validation."""
    client = MagicMock(spec=httpx.Client)
    type(client).base_url = PropertyMock(return_value=httpx.URL("https://example.com/base"))
    client.headers = httpx.Headers({"Authorization": "Bearer tok"})
    return client


def _make_mock_async_client():
    """Create a mock async httpx client that passes pydantic validation."""
    return MagicMock(spec=httpx.AsyncClient)


class TestUiPathGoogleInit:
    """Tests for UiPathGoogle client construction with mocked dependencies."""

    @patch("uipath.llm_client.clients.google.client.build_httpx_async_client")
    @patch("uipath.llm_client.clients.google.client.build_httpx_client")
    @patch("uipath.llm_client.clients.google.client.get_default_client_settings")
    def test_placeholder_api_key(
        self,
        mock_get_settings,
        mock_build_sync,
        mock_build_async,
    ):
        """UiPathGoogle passes api_key='PLACEHOLDER' to the parent Client."""
        from google.genai.client import Client

        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_build_sync.return_value = _make_mock_sync_client()
        mock_build_async.return_value = _make_mock_async_client()

        with patch.object(Client, "__init__", return_value=None) as mock_init:
            from uipath.llm_client.clients.google.client import UiPathGoogle

            UiPathGoogle(model_name="gemini-2.5-flash")

            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["api_key"] == "PLACEHOLDER"

    @patch("uipath.llm_client.clients.google.client.build_httpx_async_client")
    @patch("uipath.llm_client.clients.google.client.build_httpx_client")
    @patch("uipath.llm_client.clients.google.client.get_default_client_settings")
    def test_httpx_clients_passed_via_http_options(
        self,
        mock_get_settings,
        mock_build_sync,
        mock_build_async,
    ):
        """UiPathGoogle passes both httpx clients into HttpOptions."""
        from google.genai.client import Client

        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        sync_client = _make_mock_sync_client()
        async_client = _make_mock_async_client()
        mock_build_sync.return_value = sync_client
        mock_build_async.return_value = async_client

        with patch.object(Client, "__init__", return_value=None) as mock_init:
            from uipath.llm_client.clients.google.client import UiPathGoogle

            UiPathGoogle(model_name="gemini-2.5-flash")

            call_kwargs = mock_init.call_args[1]
            http_options = call_kwargs["http_options"]
            assert http_options.httpx_client is sync_client
            assert http_options.httpx_async_client is async_client
            assert str(http_options.base_url) == "https://example.com/base"
            assert http_options.retry_options is None

    @patch("uipath.llm_client.clients.google.client.build_httpx_async_client")
    @patch("uipath.llm_client.clients.google.client.build_httpx_client")
    @patch("uipath.llm_client.clients.google.client.get_default_client_settings")
    def test_uses_provided_client_settings(
        self,
        mock_get_settings,
        mock_build_sync,
        mock_build_async,
    ):
        """When client_settings is provided, get_default_client_settings is not called."""
        from google.genai.client import Client

        custom_settings = MagicMock()
        mock_build_sync.return_value = _make_mock_sync_client()
        mock_build_async.return_value = _make_mock_async_client()

        with patch.object(Client, "__init__", return_value=None):
            from uipath.llm_client.clients.google.client import UiPathGoogle

            UiPathGoogle(model_name="gemini-2.5-flash", client_settings=custom_settings)

            mock_get_settings.assert_not_called()
            assert mock_build_sync.call_args[1]["client_settings"] is custom_settings
            assert mock_build_async.call_args[1]["client_settings"] is custom_settings

    @patch("uipath.llm_client.clients.google.client.build_httpx_async_client")
    @patch("uipath.llm_client.clients.google.client.build_httpx_client")
    @patch("uipath.llm_client.clients.google.client.get_default_client_settings")
    def test_api_config_forwarded_to_builders(
        self,
        mock_get_settings,
        mock_build_sync,
        mock_build_async,
    ):
        """The internally-built api_config is forwarded to both httpx client builders."""
        from google.genai.client import Client

        mock_get_settings.return_value = MagicMock()
        mock_build_sync.return_value = _make_mock_sync_client()
        mock_build_async.return_value = _make_mock_async_client()

        with patch.object(Client, "__init__", return_value=None):
            from uipath.llm_client.clients.google.client import UiPathGoogle

            UiPathGoogle(model_name="gemini-2.5-flash")

            sync_config = mock_build_sync.call_args[1]["api_config"]
            async_config = mock_build_async.call_args[1]["api_config"]

            for cfg in (sync_config, async_config):
                assert cfg.api_type == "completions"
                assert cfg.routing_mode == "passthrough"
                assert cfg.vendor_type == "vertexai"
                assert cfg.api_flavor == "generate-content"
                assert cfg.freeze_base_url is True
