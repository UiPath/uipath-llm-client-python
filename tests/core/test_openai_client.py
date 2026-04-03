"""Tests for the OpenAI client module.

This module tests:
1. OpenAIRequestHandler.fix_url_and_headers (sync and async) routing logic
2. Client initialization for UiPathOpenAI, UiPathAsyncOpenAI, UiPathAzureOpenAI,
   UiPathAsyncAzureOpenAI
"""

from unittest.mock import MagicMock, patch

import pytest
from httpx import Request

from uipath.llm_client.clients.openai.utils import OpenAIRequestHandler
from uipath.llm_client.settings.base import UiPathAPIConfig
from uipath.llm_client.settings.constants import (
    ApiFlavor,
    ApiType,
    RoutingMode,
    VendorType,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.build_base_url.return_value = "https://gateway.uipath.com/llm/v1"
    settings.build_auth_headers.return_value = {"Authorization": "Bearer test-token"}
    settings.build_auth_pipeline.return_value = None
    return settings


@pytest.fixture
def handler(mock_settings):
    return OpenAIRequestHandler(
        model_name="gpt-4o",
        client_settings=mock_settings,
        byo_connection_id=None,
    )


@pytest.fixture
def handler_with_byo(mock_settings):
    return OpenAIRequestHandler(
        model_name="gpt-4o",
        client_settings=mock_settings,
        byo_connection_id="my-connection-id",
    )


def _make_request(path: str) -> Request:
    return Request("POST", f"https://api.openai.com{path}")


# ============================================================================
# OpenAIRequestHandler.__init__
# ============================================================================


class TestOpenAIRequestHandlerInit:
    def test_stores_model_name(self, handler):
        assert handler.model_name == "gpt-4o"

    def test_stores_byo_connection_id(self, handler_with_byo):
        assert handler_with_byo.byo_connection_id == "my-connection-id"

    def test_base_api_config_defaults(self, handler):
        cfg = handler.base_api_config
        assert cfg.routing_mode == RoutingMode.PASSTHROUGH
        assert cfg.vendor_type == VendorType.OPENAI
        assert cfg.api_version == "2025-03-01-preview"
        assert cfg.freeze_base_url is False
        assert cfg.api_type is None
        assert cfg.api_flavor is None


# ============================================================================
# fix_url_and_headers — completions endpoint
# ============================================================================


class TestFixUrlCompletions:
    def test_sets_chat_completions_flavor(self, handler):
        request = _make_request("/v1/chat/completions")
        handler.fix_url_and_headers(request)

        call_args = handler.client_settings.build_base_url.call_args
        api_config: UiPathAPIConfig = call_args.kwargs["api_config"]
        assert api_config.api_flavor == ApiFlavor.CHAT_COMPLETIONS
        assert api_config.api_type == ApiType.COMPLETIONS

    def test_preserves_base_config_fields(self, handler):
        request = _make_request("/v1/chat/completions")
        handler.fix_url_and_headers(request)

        call_args = handler.client_settings.build_base_url.call_args
        api_config: UiPathAPIConfig = call_args.kwargs["api_config"]
        assert api_config.routing_mode == RoutingMode.PASSTHROUGH
        assert api_config.vendor_type == VendorType.OPENAI
        assert api_config.api_version == "2025-03-01-preview"

    def test_rewrites_url(self, handler):
        request = _make_request("/v1/chat/completions")
        handler.fix_url_and_headers(request)

        assert str(request.url) == "https://gateway.uipath.com/llm/v1"

    def test_injects_routing_headers(self, handler):
        request = _make_request("/v1/chat/completions")
        handler.fix_url_and_headers(request)

        assert "X-UiPath-LlmGateway-ApiFlavor" in request.headers
        assert request.headers["X-UiPath-LlmGateway-ApiFlavor"] == ApiFlavor.CHAT_COMPLETIONS

    def test_injects_api_version_header(self, handler):
        request = _make_request("/v1/chat/completions")
        handler.fix_url_and_headers(request)

        assert request.headers.get("X-UiPath-LlmGateway-ApiVersion") == "2025-03-01-preview"


# ============================================================================
# fix_url_and_headers — responses endpoint
# ============================================================================


class TestFixUrlResponses:
    def test_sets_responses_flavor(self, handler):
        request = _make_request("/v1/responses")
        handler.fix_url_and_headers(request)

        call_args = handler.client_settings.build_base_url.call_args
        api_config: UiPathAPIConfig = call_args.kwargs["api_config"]
        assert api_config.api_flavor == ApiFlavor.RESPONSES
        assert api_config.api_type == ApiType.COMPLETIONS

    def test_rewrites_url(self, handler):
        request = _make_request("/v1/responses")
        handler.fix_url_and_headers(request)

        assert str(request.url) == "https://gateway.uipath.com/llm/v1"

    def test_injects_responses_flavor_header(self, handler):
        request = _make_request("/v1/responses")
        handler.fix_url_and_headers(request)

        assert request.headers["X-UiPath-LlmGateway-ApiFlavor"] == ApiFlavor.RESPONSES


# ============================================================================
# fix_url_and_headers — embeddings endpoint
# ============================================================================


class TestFixUrlEmbeddings:
    def test_sets_embeddings_api_type(self, handler):
        request = _make_request("/v1/embeddings")
        handler.fix_url_and_headers(request)

        call_args = handler.client_settings.build_base_url.call_args
        api_config: UiPathAPIConfig = call_args.kwargs["api_config"]
        assert api_config.api_type == ApiType.EMBEDDINGS

    def test_no_api_flavor_for_embeddings(self, handler):
        request = _make_request("/v1/embeddings")
        handler.fix_url_and_headers(request)

        call_args = handler.client_settings.build_base_url.call_args
        api_config: UiPathAPIConfig = call_args.kwargs["api_config"]
        assert api_config.api_flavor is None

    def test_rewrites_url(self, handler):
        request = _make_request("/v1/embeddings")
        handler.fix_url_and_headers(request)

        assert str(request.url) == "https://gateway.uipath.com/llm/v1"

    def test_no_flavor_header_for_embeddings(self, handler):
        request = _make_request("/v1/embeddings")
        handler.fix_url_and_headers(request)

        assert "X-UiPath-LlmGateway-ApiFlavor" not in request.headers


# ============================================================================
# fix_url_and_headers — unrecognized endpoint
# ============================================================================


class TestFixUrlUnrecognized:
    def test_does_not_rewrite_url(self, handler):
        request = _make_request("/v1/models")
        original_url = str(request.url)
        handler.fix_url_and_headers(request)

        assert str(request.url) == original_url

    def test_does_not_call_build_base_url(self, handler):
        request = _make_request("/v1/models")
        handler.fix_url_and_headers(request)

        handler.client_settings.build_base_url.assert_not_called()

    def test_logs_debug_message(self, handler, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger="uipath.llm_client.clients.openai.utils"):
            request = _make_request("/v1/models")
            handler.fix_url_and_headers(request)

        assert any("Unrecognized API endpoint" in msg for msg in caplog.messages)


# ============================================================================
# fix_url_and_headers — BYO connection ID
# ============================================================================


class TestFixUrlByoConnectionId:
    def test_byo_header_injected_on_completions(self, handler_with_byo):
        request = _make_request("/v1/chat/completions")
        handler_with_byo.fix_url_and_headers(request)

        assert request.headers["X-UiPath-LlmGateway-ByoIsConnectionId"] == "my-connection-id"

    def test_byo_header_injected_on_embeddings(self, handler_with_byo):
        request = _make_request("/v1/embeddings")
        handler_with_byo.fix_url_and_headers(request)

        assert request.headers["X-UiPath-LlmGateway-ByoIsConnectionId"] == "my-connection-id"

    def test_no_byo_header_when_none(self, handler):
        request = _make_request("/v1/chat/completions")
        handler.fix_url_and_headers(request)

        assert "X-UiPath-LlmGateway-ByoIsConnectionId" not in request.headers


# ============================================================================
# fix_url_and_headers — does not mutate base_api_config
# ============================================================================


class TestFixUrlDoesNotMutateBase:
    def test_base_config_unchanged_after_completions(self, handler):
        request = _make_request("/v1/chat/completions")
        handler.fix_url_and_headers(request)

        assert handler.base_api_config.api_type is None
        assert handler.base_api_config.api_flavor is None

    def test_base_config_unchanged_after_embeddings(self, handler):
        request = _make_request("/v1/embeddings")
        handler.fix_url_and_headers(request)

        assert handler.base_api_config.api_type is None
        assert handler.base_api_config.api_flavor is None


# ============================================================================
# fix_url_and_headers_async
# ============================================================================


class TestFixUrlAndHeadersAsync:
    @pytest.mark.asyncio
    async def test_async_delegates_to_sync(self, handler):
        request = _make_request("/v1/chat/completions")
        await handler.fix_url_and_headers_async(request)

        assert str(request.url) == "https://gateway.uipath.com/llm/v1"
        call_args = handler.client_settings.build_base_url.call_args
        api_config: UiPathAPIConfig = call_args.kwargs["api_config"]
        assert api_config.api_flavor == ApiFlavor.CHAT_COMPLETIONS

    @pytest.mark.asyncio
    async def test_async_responses_endpoint(self, handler):
        request = _make_request("/v1/responses")
        await handler.fix_url_and_headers_async(request)

        call_args = handler.client_settings.build_base_url.call_args
        api_config: UiPathAPIConfig = call_args.kwargs["api_config"]
        assert api_config.api_flavor == ApiFlavor.RESPONSES


# ============================================================================
# Client initialization
# ============================================================================

_CLIENT_MODULE = "uipath.llm_client.clients.openai.client"


def _mock_httpx_sync_client():
    import httpx

    client = MagicMock(spec=httpx.Client)
    client.base_url = "https://gateway.uipath.com/llm/v1"
    client.headers = httpx.Headers()
    client._transport = MagicMock()
    client._base_url = httpx.URL("https://gateway.uipath.com/llm/v1")
    return client


def _mock_httpx_async_client():
    import httpx

    client = MagicMock(spec=httpx.AsyncClient)
    client.base_url = "https://gateway.uipath.com/llm/v1"
    client.headers = httpx.Headers()
    client._transport = MagicMock()
    client._base_url = httpx.URL("https://gateway.uipath.com/llm/v1")
    return client


class TestUiPathOpenAIInit:
    @patch(f"{_CLIENT_MODULE}.build_httpx_client", return_value=_mock_httpx_sync_client())
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_uses_default_settings_when_none(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathOpenAI

        UiPathOpenAI(model_name="gpt-4o")

        mock_get_settings.assert_called_once()

    @patch(f"{_CLIENT_MODULE}.build_httpx_client", return_value=_mock_httpx_sync_client())
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_passes_event_hooks_with_fix_url(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathOpenAI

        UiPathOpenAI(model_name="gpt-4o")

        call_kwargs = mock_build.call_args.kwargs
        assert "event_hooks" in call_kwargs
        hooks = call_kwargs["event_hooks"]
        assert "request" in hooks
        assert len(hooks["request"]) == 1
        # The hook should be a bound method of OpenAIRequestHandler
        hook_fn = hooks["request"][0]
        assert hook_fn.__func__.__name__ == "fix_url_and_headers"

    @patch(f"{_CLIENT_MODULE}.build_httpx_client", return_value=_mock_httpx_sync_client())
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_sets_max_retries_zero(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathOpenAI

        client = UiPathOpenAI(model_name="gpt-4o")

        assert client.max_retries == 0


class TestUiPathAsyncOpenAIInit:
    @patch(
        f"{_CLIENT_MODULE}.build_httpx_async_client",
        return_value=_mock_httpx_async_client(),
    )
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_passes_async_event_hooks(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathAsyncOpenAI

        UiPathAsyncOpenAI(model_name="gpt-4o")

        call_kwargs = mock_build.call_args.kwargs
        hooks = call_kwargs["event_hooks"]
        hook_fn = hooks["request"][0]
        assert hook_fn.__func__.__name__ == "fix_url_and_headers_async"

    @patch(
        f"{_CLIENT_MODULE}.build_httpx_async_client",
        return_value=_mock_httpx_async_client(),
    )
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_sets_max_retries_zero(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathAsyncOpenAI

        client = UiPathAsyncOpenAI(model_name="gpt-4o")

        assert client.max_retries == 0


class TestUiPathAzureOpenAIInit:
    @patch(f"{_CLIENT_MODULE}.build_httpx_client", return_value=_mock_httpx_sync_client())
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_passes_placeholder_values(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathAzureOpenAI

        client = UiPathAzureOpenAI(model_name="gpt-4o")

        # AzureOpenAI stores these internally; we verify the client was created
        # successfully with PLACEHOLDER values (no real Azure config needed)
        assert client.max_retries == 0

    @patch(f"{_CLIENT_MODULE}.build_httpx_client", return_value=_mock_httpx_sync_client())
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_uses_sync_event_hook(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathAzureOpenAI

        UiPathAzureOpenAI(model_name="gpt-4o")

        call_kwargs = mock_build.call_args.kwargs
        hooks = call_kwargs["event_hooks"]
        hook_fn = hooks["request"][0]
        assert hook_fn.__func__.__name__ == "fix_url_and_headers"


class TestUiPathAsyncAzureOpenAIInit:
    @patch(
        f"{_CLIENT_MODULE}.build_httpx_async_client",
        return_value=_mock_httpx_async_client(),
    )
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_sets_max_retries_zero(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathAsyncAzureOpenAI

        client = UiPathAsyncAzureOpenAI(model_name="gpt-4o")

        assert client.max_retries == 0

    @patch(
        f"{_CLIENT_MODULE}.build_httpx_async_client",
        return_value=_mock_httpx_async_client(),
    )
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_uses_async_event_hook(self, mock_get_settings, mock_build):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        from uipath.llm_client.clients.openai.client import UiPathAsyncAzureOpenAI

        UiPathAsyncAzureOpenAI(model_name="gpt-4o")

        call_kwargs = mock_build.call_args.kwargs
        hooks = call_kwargs["event_hooks"]
        hook_fn = hooks["request"][0]
        assert hook_fn.__func__.__name__ == "fix_url_and_headers_async"
