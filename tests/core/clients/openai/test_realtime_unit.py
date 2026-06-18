# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
"""Unit tests for realtime support on the UiPath OpenAI clients (no network).

Covers the URL helper and the ``client.realtime`` resource wiring: lazy URL
construction, websocket_base_url + api_key setup on connect, routing header
injection, default-model resolution, and token refresh.
"""

from unittest.mock import MagicMock, patch

import httpx
from openai.resources.realtime import AsyncRealtime, Realtime

from uipath.llm_client.clients.openai import UiPathAsyncOpenAI, UiPathOpenAI
from uipath.llm_client.clients.openai.realtime import (
    _UiPathAsyncRealtime,
    _UiPathRealtime,
    build_realtime_ws_base_url,
)
from uipath.llm_client.settings import UiPathBaseSettings
from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

_CLIENT_MODULE = "uipath.llm_client.clients.openai.client"

_REALTIME_PATH = (
    "https://gw.uipath.com/org/tenant/llmgateway_/api/raw"
    "/vendor/nativeopenai/model/gpt-realtime/realtime"
)
_WS_BASE = (
    "wss://gw.uipath.com/org/tenant/llmgateway_/api/raw/vendor/nativeopenai/model/gpt-realtime"
)


def _fake_s2s_auth(token: str | None) -> MagicMock:
    # spec=LLMGatewayS2SAuth so isinstance() in _prepare_connection passes.
    auth = MagicMock(spec=LLMGatewayS2SAuth)
    auth.access_token = token
    return auth


def _fake_settings(
    *,
    url: str = _REALTIME_PATH,
    headers: dict[str, str] | None = None,
    token: str | None = "tok-abc",
) -> MagicMock:
    settings = MagicMock(spec=UiPathBaseSettings)
    settings.build_base_url.return_value = url
    settings.build_auth_headers.return_value = headers or {"X-UiPath-Internal-AccountId": "org"}
    settings.build_auth_pipeline.return_value = _fake_s2s_auth(token)
    return settings


def _mock_httpx_sync_client() -> MagicMock:
    client = MagicMock(spec=httpx.Client)
    client.base_url = "https://gw.uipath.com/llm/v1"
    client.headers = httpx.Headers()
    client._transport = MagicMock()
    client._base_url = httpx.URL("https://gw.uipath.com/llm/v1")
    return client


def _mock_httpx_async_client() -> MagicMock:
    client = MagicMock(spec=httpx.AsyncClient)
    client.base_url = "https://gw.uipath.com/llm/v1"
    client.headers = httpx.Headers()
    client._transport = MagicMock()
    client._base_url = httpx.URL("https://gw.uipath.com/llm/v1")
    return client


# ============================================================================
# build_realtime_ws_base_url
# ============================================================================


class TestBuildRealtimeWsBaseUrl:
    def test_strips_realtime_suffix_and_converts_scheme(self) -> None:
        url = build_realtime_ws_base_url(
            _fake_settings(), model_name="gpt-realtime", vendor_type="nativeopenai"
        )
        assert url == _WS_BASE

    def test_collapses_double_slashes_from_trailing_base_url(self) -> None:
        settings = _fake_settings(
            url="https://gw.uipath.com//org/tenant/llmgateway_/api/raw"
            "/vendor/nativeopenai/model/gpt-realtime/realtime"
        )
        url = build_realtime_ws_base_url(settings, model_name="gpt-realtime")
        assert url == _WS_BASE
        assert "//org" not in url
        assert url.startswith("wss://")

    def test_http_maps_to_ws(self) -> None:
        settings = _fake_settings(
            url="http://localhost:7091/o/t/llmgateway_/api/raw"
            "/vendor/nativeopenai/model/gpt-realtime/realtime"
        )
        url = build_realtime_ws_base_url(settings, model_name="gpt-realtime")
        assert url.startswith("ws://")
        assert url.endswith("/model/gpt-realtime")

    def test_passes_realtime_api_config(self) -> None:
        settings = _fake_settings()
        build_realtime_ws_base_url(settings, model_name="gpt-realtime", vendor_type="nativeopenai")
        cfg = settings.build_base_url.call_args.kwargs["api_config"]
        assert cfg.api_type == "realtime"
        assert str(cfg.routing_mode) == "passthrough"
        assert cfg.vendor_type == "nativeopenai"


# ============================================================================
# Sync client .realtime wiring
# ============================================================================


@patch(f"{_CLIENT_MODULE}.UiPathHttpxClient", return_value=_mock_httpx_sync_client())
class TestSyncRealtimeProperty:
    def test_realtime_returns_wrapper(self, _mock_build: MagicMock) -> None:
        client = UiPathOpenAI(model_name="gpt-realtime", client_settings=_fake_settings())
        assert isinstance(client.realtime, _UiPathRealtime)

    def test_construction_does_not_build_realtime_url(self, _mock_build: MagicMock) -> None:
        settings = _fake_settings()
        UiPathOpenAI(model_name="gpt-realtime", client_settings=settings)
        settings.build_base_url.assert_not_called()  # built lazily on connect

    def test_connect_sets_ws_url_token_and_headers(self, _mock_build: MagicMock) -> None:
        client = UiPathOpenAI(
            model_name="gpt-realtime",
            client_settings=_fake_settings(
                headers={"X-UiPath-Internal-AccountId": "org"}, token="tok-1"
            ),
        )
        with patch.object(Realtime, "connect", return_value="MANAGER") as mock_connect:
            result = client.realtime.connect(extra_headers={"X-Extra": "e"})

        assert result == "MANAGER"
        assert client.websocket_base_url == _WS_BASE
        assert client.api_key == "tok-1"
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["model"] == "gpt-realtime"
        assert kwargs["extra_headers"]["X-UiPath-Internal-AccountId"] == "org"
        assert kwargs["extra_headers"]["X-Extra"] == "e"

    def test_explicit_model_overrides_default(self, _mock_build: MagicMock) -> None:
        client = UiPathOpenAI(model_name="gpt-realtime", client_settings=_fake_settings())
        with patch.object(Realtime, "connect", return_value="M") as mock_connect:
            client.realtime.connect(model="gpt-realtime-2")
        assert mock_connect.call_args.kwargs["model"] == "gpt-realtime-2"

    def test_connect_refreshes_token(self, _mock_build: MagicMock) -> None:
        settings = _fake_settings(token="old")
        client = UiPathOpenAI(model_name="gpt-realtime", client_settings=settings)
        with patch.object(Realtime, "connect", return_value="M"):
            client.realtime.connect()
            assert client.api_key == "old"
            settings.build_auth_pipeline.return_value = _fake_s2s_auth("refreshed")
            client.realtime.connect()
            assert client.api_key == "refreshed"


# ============================================================================
# Async client .realtime wiring
# ============================================================================


@patch(f"{_CLIENT_MODULE}.UiPathHttpxAsyncClient", return_value=_mock_httpx_async_client())
class TestAsyncRealtimeProperty:
    def test_realtime_returns_wrapper(self, _mock_build: MagicMock) -> None:
        client = UiPathAsyncOpenAI(model_name="gpt-realtime", client_settings=_fake_settings())
        assert isinstance(client.realtime, _UiPathAsyncRealtime)

    def test_connect_sets_ws_url_token_and_headers(self, _mock_build: MagicMock) -> None:
        client = UiPathAsyncOpenAI(
            model_name="gpt-realtime",
            client_settings=_fake_settings(
                headers={"X-UiPath-Internal-AccountId": "org"}, token="tok-1"
            ),
        )
        with patch.object(AsyncRealtime, "connect", return_value="MANAGER") as mock_connect:
            result = client.realtime.connect(extra_headers={"X-Extra": "e"})

        assert result == "MANAGER"
        assert client.websocket_base_url == _WS_BASE
        assert client.api_key == "tok-1"
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["model"] == "gpt-realtime"
        assert kwargs["extra_headers"]["X-UiPath-Internal-AccountId"] == "org"
        assert kwargs["extra_headers"]["X-Extra"] == "e"
