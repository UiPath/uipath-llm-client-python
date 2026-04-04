"""Tests for the Anthropic client module."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from uipath.llm_client.clients.anthropic.client import _build_api_config
from uipath.llm_client.settings.constants import ApiType, RoutingMode, VendorType

MODULE = "uipath.llm_client.clients.anthropic.client"


# ============================================================================
# _build_api_config tests
# ============================================================================


class TestBuildApiConfig:
    def test_default_returns_anthropic_passthrough(self):
        config = _build_api_config()
        assert config.vendor_type == VendorType.ANTHROPIC
        assert config.routing_mode == RoutingMode.PASSTHROUGH
        assert config.api_type == ApiType.COMPLETIONS
        assert config.freeze_base_url is True

    def test_awsbedrock_vendor(self):
        config = _build_api_config(vendor_type=VendorType.AWSBEDROCK)
        assert config.vendor_type == VendorType.AWSBEDROCK

    def test_vertexai_vendor(self):
        config = _build_api_config(vendor_type=VendorType.VERTEXAI)
        assert config.vendor_type == VendorType.VERTEXAI

    def test_azure_vendor(self):
        config = _build_api_config(vendor_type=VendorType.AZURE)
        assert config.vendor_type == VendorType.AZURE


# ============================================================================
# Client initialization tests
# ============================================================================


def _make_sync_httpx_mock():
    mock = MagicMock(spec=httpx.Client)
    mock.timeout = httpx.Timeout(None)
    mock.headers = httpx.Headers()
    return mock


def _make_async_httpx_mock():
    mock = MagicMock(spec=httpx.AsyncClient)
    mock.timeout = httpx.Timeout(None)
    mock.headers = httpx.Headers()
    return mock


@pytest.fixture
def mock_settings():
    return MagicMock()


@pytest.fixture
def _patch_client_deps(mock_settings):
    with (
        patch(
            f"{MODULE}.build_httpx_client",
            side_effect=lambda **kw: _make_sync_httpx_mock(),
        ) as sync_mock,
        patch(
            f"{MODULE}.build_httpx_async_client",
            side_effect=lambda **kw: _make_async_httpx_mock(),
        ) as async_mock,
        patch(f"{MODULE}.get_default_client_settings", return_value=mock_settings),
    ):
        yield sync_mock, async_mock


@pytest.fixture
def _foundry_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "test-resource")


@pytest.mark.usefixtures("_patch_client_deps")
class TestUiPathAnthropic:
    def test_passes_api_key_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropic

        sync_mock, _ = _patch_client_deps
        client = UiPathAnthropic(model_name="claude-3-5-sonnet")
        assert client.api_key == "PLACEHOLDER"
        assert client.max_retries == 0
        sync_mock.assert_called_once()

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropic

        sync_mock, _ = _patch_client_deps
        UiPathAnthropic(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = sync_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"


@pytest.mark.usefixtures("_patch_client_deps")
class TestUiPathAsyncAnthropic:
    def test_passes_api_key_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropic

        _, async_mock = _patch_client_deps
        client = UiPathAsyncAnthropic(model_name="claude-3-5-sonnet")
        assert client.api_key == "PLACEHOLDER"
        assert client.max_retries == 0
        async_mock.assert_called_once()

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropic

        _, async_mock = _patch_client_deps
        UiPathAsyncAnthropic(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = async_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"


@pytest.mark.usefixtures("_patch_client_deps")
class TestUiPathAnthropicBedrock:
    def test_passes_aws_placeholders_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicBedrock

        sync_mock, _ = _patch_client_deps
        client = UiPathAnthropicBedrock(model_name="claude-3-5-sonnet")
        assert client.aws_access_key == "PLACEHOLDER"
        assert client.aws_secret_key == "PLACEHOLDER"
        assert client.aws_region == "PLACEHOLDER"
        assert client.max_retries == 0
        sync_mock.assert_called_once()

    def test_uses_awsbedrock_vendor(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicBedrock

        sync_mock, _ = _patch_client_deps
        UiPathAnthropicBedrock(model_name="claude-3-5-sonnet")
        api_config = sync_mock.call_args.kwargs["api_config"]
        assert api_config.vendor_type == VendorType.AWSBEDROCK

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicBedrock

        sync_mock, _ = _patch_client_deps
        UiPathAnthropicBedrock(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = sync_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"


@pytest.mark.usefixtures("_patch_client_deps")
class TestUiPathAsyncAnthropicBedrock:
    def test_passes_aws_placeholders_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicBedrock

        _, async_mock = _patch_client_deps
        client = UiPathAsyncAnthropicBedrock(model_name="claude-3-5-sonnet")
        assert client.aws_access_key == "PLACEHOLDER"
        assert client.aws_secret_key == "PLACEHOLDER"
        assert client.aws_region == "PLACEHOLDER"
        assert client.max_retries == 0
        async_mock.assert_called_once()

    def test_uses_awsbedrock_vendor(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicBedrock

        _, async_mock = _patch_client_deps
        UiPathAsyncAnthropicBedrock(model_name="claude-3-5-sonnet")
        api_config = async_mock.call_args.kwargs["api_config"]
        assert api_config.vendor_type == VendorType.AWSBEDROCK

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicBedrock

        _, async_mock = _patch_client_deps
        UiPathAsyncAnthropicBedrock(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = async_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"


@pytest.mark.usefixtures("_patch_client_deps")
class TestUiPathAnthropicVertex:
    def test_passes_vertex_placeholders_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicVertex

        sync_mock, _ = _patch_client_deps
        client = UiPathAnthropicVertex(model_name="claude-3-5-sonnet")
        assert client.region == "PLACEHOLDER"
        assert client.project_id == "PLACEHOLDER"
        assert client.max_retries == 0
        sync_mock.assert_called_once()

    def test_uses_vertexai_vendor(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicVertex

        sync_mock, _ = _patch_client_deps
        UiPathAnthropicVertex(model_name="claude-3-5-sonnet")
        api_config = sync_mock.call_args.kwargs["api_config"]
        assert api_config.vendor_type == VendorType.VERTEXAI

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicVertex

        sync_mock, _ = _patch_client_deps
        UiPathAnthropicVertex(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = sync_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"


@pytest.mark.usefixtures("_patch_client_deps")
class TestUiPathAsyncAnthropicVertex:
    def test_passes_vertex_placeholders_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicVertex

        _, async_mock = _patch_client_deps
        client = UiPathAsyncAnthropicVertex(model_name="claude-3-5-sonnet")
        assert client.region == "PLACEHOLDER"
        assert client.project_id == "PLACEHOLDER"
        assert client.max_retries == 0
        async_mock.assert_called_once()

    def test_uses_vertexai_vendor(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicVertex

        _, async_mock = _patch_client_deps
        UiPathAsyncAnthropicVertex(model_name="claude-3-5-sonnet")
        api_config = async_mock.call_args.kwargs["api_config"]
        assert api_config.vendor_type == VendorType.VERTEXAI

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicVertex

        _, async_mock = _patch_client_deps
        UiPathAsyncAnthropicVertex(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = async_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"


@pytest.mark.usefixtures("_patch_client_deps", "_foundry_env")
class TestUiPathAnthropicFoundry:
    def test_passes_api_key_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicFoundry

        sync_mock, _ = _patch_client_deps
        client = UiPathAnthropicFoundry(model_name="claude-3-5-sonnet")
        assert client.api_key == "PLACEHOLDER"
        assert client.max_retries == 0
        sync_mock.assert_called_once()

    def test_uses_azure_vendor(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicFoundry

        sync_mock, _ = _patch_client_deps
        UiPathAnthropicFoundry(model_name="claude-3-5-sonnet")
        api_config = sync_mock.call_args.kwargs["api_config"]
        assert api_config.vendor_type == VendorType.AZURE

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAnthropicFoundry

        sync_mock, _ = _patch_client_deps
        UiPathAnthropicFoundry(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = sync_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"


@pytest.mark.usefixtures("_patch_client_deps", "_foundry_env")
class TestUiPathAsyncAnthropicFoundry:
    def test_passes_api_key_and_max_retries(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicFoundry

        _, async_mock = _patch_client_deps
        client = UiPathAsyncAnthropicFoundry(model_name="claude-3-5-sonnet")
        assert client.api_key == "PLACEHOLDER"
        assert client.max_retries == 0
        async_mock.assert_called_once()

    def test_uses_azure_vendor(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicFoundry

        _, async_mock = _patch_client_deps
        UiPathAsyncAnthropicFoundry(model_name="claude-3-5-sonnet")
        api_config = async_mock.call_args.kwargs["api_config"]
        assert api_config.vendor_type == VendorType.AZURE

    def test_passes_model_name_and_byo_connection_id(self, _patch_client_deps):
        from uipath.llm_client.clients.anthropic.client import UiPathAsyncAnthropicFoundry

        _, async_mock = _patch_client_deps
        UiPathAsyncAnthropicFoundry(model_name="claude-3-5-sonnet", byo_connection_id="conn-123")
        kwargs = async_mock.call_args.kwargs
        assert kwargs["model_name"] == "claude-3-5-sonnet"
        assert kwargs["byo_connection_id"] == "conn-123"
