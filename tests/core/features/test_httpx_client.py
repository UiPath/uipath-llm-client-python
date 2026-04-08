"""Tests for HTTPX client functionality."""

from unittest.mock import MagicMock, patch

from httpx import Client, Headers, Request, Response

from uipath.llm_client.settings import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.utils.retry import (
    RetryableAsyncHTTPTransport,
    RetryableHTTPTransport,
)


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
        assert client.headers["X-UiPath-LLMGateway-AllowFull4xxResponse"] == "false"
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
        assert isinstance(client, UiPathHttpxClient)
        client.close()

    def test_client_with_api_config(self, normalized_api_config):
        """Test client adds routing headers from api_config."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(
            base_url="https://example.com",
            api_config=normalized_api_config,
            model_name="gpt-4o",
        )
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
        assert client.headers["X-UiPath-LLMGateway-AllowFull4xxResponse"] == "false"

    def test_async_client_with_retry_config(self):
        """Test async client creates retryable async transport."""
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        client = UiPathHttpxAsyncClient(
            base_url="https://example.com",
            max_retries=3,
        )
        # Transport should be RetryableAsyncHTTPTransport
        assert isinstance(client._transport, RetryableAsyncHTTPTransport)


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
