"""Tests for HTTPX client functionality."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import Auth, Client, Headers, MockTransport, Request, Response

from tests.lazy_stream import LazyByteStream
from uipath.llm_client.settings import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.utils.dollar_cost import INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER
from uipath.llm_client.utils.retry import (
    RetryableAsyncHTTPTransport,
    RetryableHTTPTransport,
)


class RecordingHeaderAuth(Auth):
    def __init__(self):
        self.signed_user_agents = []

    def auth_flow(self, request):
        self.signed_user_agents = request.headers.get_list("user-agent")
        request.headers["x-signed-user-agents"] = "|".join(self.signed_user_agents)
        yield request


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

    def test_client_default_max_retries_is_three(self):
        """Caller passing no ``max_retries`` should get the 3-retry default."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com")
        assert isinstance(client._transport, RetryableHTTPTransport)
        assert client._transport.retryer is not None
        client.close()

    def test_client_explicit_zero_disables_retries(self):
        """Passing ``max_retries=0`` must still disable retries."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        client = UiPathHttpxClient(base_url="https://example.com", max_retries=0)
        assert isinstance(client._transport, RetryableHTTPTransport)
        assert client._transport.retryer is None
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

    def test_async_client_default_max_retries_is_three(self):
        """Async caller passing no ``max_retries`` should get the 3-retry default."""
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        client = UiPathHttpxAsyncClient(base_url="https://example.com")
        assert isinstance(client._transport, RetryableAsyncHTTPTransport)
        assert client._transport.retryer is not None

    def test_async_client_explicit_zero_disables_retries(self):
        """Async client: passing ``max_retries=0`` must still disable retries."""
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        client = UiPathHttpxAsyncClient(base_url="https://example.com", max_retries=0)
        assert isinstance(client._transport, RetryableAsyncHTTPTransport)
        assert client._transport.retryer is None


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

    def test_case_variant_headers_are_merged_before_send(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        sent_requests = []

        def handler(request: Request) -> Response:
            sent_requests.append(request)
            return Response(200, request=request)

        client = UiPathHttpxClient(
            base_url="https://example.com",
            transport=MockTransport(handler),
        )
        request = Request(
            "POST",
            "https://example.com/test",
            headers=Headers(
                [
                    (b"User-Agent", b"provider-sdk"),
                    (b"user-agent", b"custom-httpx-client"),
                ]
            ),
        )

        client.send(request)

        sent_request = sent_requests[0]
        assert sent_request.headers.get_list("user-agent") == ["custom-httpx-client"]
        client.close()

    def test_case_variant_headers_are_merged_before_auth(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient

        auth = RecordingHeaderAuth()

        def handler(request: Request) -> Response:
            assert request.headers.get_list("user-agent") == auth.signed_user_agents
            assert request.headers["x-signed-user-agents"] == "custom-httpx-client"
            return Response(200, request=request)

        client = UiPathHttpxClient(
            auth=auth,
            base_url="https://example.com",
            transport=MockTransport(handler),
        )
        request = Request(
            "POST",
            "https://example.com/test",
            headers=Headers(
                [
                    (b"User-Agent", b"provider-sdk"),
                    (b"user-agent", b"custom-httpx-client"),
                ]
            ),
        )

        client.send(request)

        assert auth.signed_user_agents == ["custom-httpx-client"]
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

    def _opted_in_request(self) -> Request:
        return Request(
            "POST",
            "https://example.com/test",
            headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"},
        )

    def _json_response(self, body: dict[str, Any]) -> MagicMock:
        mock_response = MagicMock(spec=Response)
        mock_response.headers = Headers({"content-type": "application/json"})
        mock_response.json.return_value = body
        mock_response.is_error = False
        mock_response.raise_for_status = MagicMock(return_value=mock_response)
        return mock_response

    def test_dollar_cost_captured_when_opted_in(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient
        from uipath.llm_client.utils.dollar_cost import get_captured_dollar_cost

        client = UiPathHttpxClient(base_url="https://example.com")
        mock_response = self._json_response({"associated_dollar_cost": 0.002145})

        with patch.object(Client, "send", return_value=mock_response):
            client.send(self._opted_in_request(), stream=False)
            assert get_captured_dollar_cost() == 0.002145
        client.close()

    def test_dollar_cost_body_not_parsed_when_not_opted_in(self):
        """Without the opt-in header the field cannot exist, so the body is not re-parsed."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient
        from uipath.llm_client.utils.dollar_cost import get_captured_dollar_cost

        client = UiPathHttpxClient(base_url="https://example.com")
        mock_response = self._json_response({"associated_dollar_cost": 0.002145})
        mock_response.json.side_effect = AssertionError("body must not be parsed without opt-in")

        with patch.object(Client, "send", return_value=mock_response):
            client.send(Request("POST", "https://example.com/test"), stream=False)
            assert get_captured_dollar_cost() is None
        client.close()

    def test_dollar_cost_json_body_not_read_for_streaming_response(self):
        """A streamed body is not buffered yet; a non-SSE stream reads as None."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient
        from uipath.llm_client.utils.dollar_cost import (
            get_captured_dollar_cost,
            set_captured_dollar_cost,
        )

        client = UiPathHttpxClient(base_url="https://example.com")
        mock_response = self._json_response({})
        mock_response.json.side_effect = AssertionError("must not be read on a streamed response")

        set_captured_dollar_cost(0.5)  # stale value from an earlier request
        with patch.object(Client, "send", return_value=mock_response):
            client.send(self._opted_in_request(), stream=True)
            assert get_captured_dollar_cost() is None
        client.close()

    def test_dollar_cost_captured_from_trailing_sse_frame(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient
        from uipath.llm_client.utils.dollar_cost import get_captured_dollar_cost

        events = [
            b'data: {"id": "1"}\n\n',
            b"data: [DONE]\n\n",
            b'data: {"associated_dollar_cost": 0.002145}\n\n',
        ]

        def handler(request: Request) -> Response:
            return Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=LazyByteStream(events),
            )

        client = UiPathHttpxClient(base_url="https://example.com", transport=MockTransport(handler))
        with client.stream(
            "POST", "/test", headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"}
        ) as response:
            assert b"".join(response.iter_bytes()) == b"".join(events)
        assert get_captured_dollar_cost() == 0.002145
        client.close()

    def test_streaming_body_untouched_when_not_opted_in(self):
        from uipath.llm_client.httpx_client import UiPathHttpxClient
        from uipath.llm_client.utils.dollar_cost import (
            DollarCostSyncStream,
            get_captured_dollar_cost,
        )

        events = [b"data: [DONE]\n\n", b'data: {"associated_dollar_cost": 0.002145}\n\n']

        def handler(request: Request) -> Response:
            return Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=LazyByteStream(events),
            )

        client = UiPathHttpxClient(base_url="https://example.com", transport=MockTransport(handler))
        with client.stream("POST", "/test") as response:
            assert not isinstance(response.stream, DollarCostSyncStream)
            response.read()
        assert get_captured_dollar_cost() is None
        client.close()

    def test_unpriced_response_resets_previous_dollar_cost(self):
        """An unpriced response must not read as the previous request's cost."""
        from uipath.llm_client.httpx_client import UiPathHttpxClient
        from uipath.llm_client.utils.dollar_cost import get_captured_dollar_cost

        client = UiPathHttpxClient(base_url="https://example.com")
        priced = self._json_response({"associated_dollar_cost": 0.002145})
        unpriced = self._json_response({"choices": []})

        with patch.object(Client, "send", side_effect=[priced, unpriced]):
            client.send(self._opted_in_request(), stream=False)
            assert get_captured_dollar_cost() == 0.002145
            client.send(self._opted_in_request(), stream=False)
            assert get_captured_dollar_cost() is None
        client.close()


class TestUiPathHttpxAsyncClientSend:
    @pytest.mark.asyncio
    async def test_case_variant_headers_are_merged_before_send(self):
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        sent_requests = []

        async def handler(request: Request) -> Response:
            sent_requests.append(request)
            return Response(200, request=request)

        client = UiPathHttpxAsyncClient(
            base_url="https://example.com",
            transport=MockTransport(handler),
        )
        request = Request(
            "POST",
            "https://example.com/test",
            headers=Headers(
                [
                    (b"User-Agent", b"provider-sdk"),
                    (b"user-agent", b"custom-httpx-client"),
                ]
            ),
        )
        await client.send(request)

        sent_request = sent_requests[0]
        assert sent_request.headers.get_list("user-agent") == ["custom-httpx-client"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_case_variant_headers_are_merged_before_auth(self):
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient

        auth = RecordingHeaderAuth()

        async def handler(request: Request) -> Response:
            assert request.headers.get_list("user-agent") == auth.signed_user_agents
            assert request.headers["x-signed-user-agents"] == "custom-httpx-client"
            return Response(200, request=request)

        client = UiPathHttpxAsyncClient(
            auth=auth,
            base_url="https://example.com",
            transport=MockTransport(handler),
        )
        request = Request(
            "POST",
            "https://example.com/test",
            headers=Headers(
                [
                    (b"User-Agent", b"provider-sdk"),
                    (b"user-agent", b"custom-httpx-client"),
                ]
            ),
        )

        await client.send(request)

        assert auth.signed_user_agents == ["custom-httpx-client"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_dollar_cost_captured_when_opted_in(self):
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient
        from uipath.llm_client.utils.dollar_cost import get_captured_dollar_cost

        client = UiPathHttpxAsyncClient(base_url="https://example.com")
        request = Request(
            "POST",
            "https://example.com/test",
            headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"},
        )

        async def handler(request: Request) -> Response:
            return Response(
                200,
                request=request,
                headers={"content-type": "application/json"},
                content=b'{"associated_dollar_cost": 0.002145}',
            )

        client._transport = MockTransport(handler)
        await client.send(request, stream=False)
        assert get_captured_dollar_cost() == 0.002145
        await client.aclose()

    @pytest.mark.asyncio
    async def test_dollar_cost_captured_from_trailing_sse_frame(self):
        from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient
        from uipath.llm_client.utils.dollar_cost import get_captured_dollar_cost

        events = [
            b'data: {"id": "1"}\n\n',
            b"data: [DONE]\n\n",
            b'data: {"associated_dollar_cost": 0.002145}\n\n',
        ]

        async def handler(request: Request) -> Response:
            return Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=LazyByteStream(events),
            )

        client = UiPathHttpxAsyncClient(
            base_url="https://example.com", transport=MockTransport(handler)
        )
        async with client.stream(
            "POST", "/test", headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"}
        ) as response:
            assert b"".join([chunk async for chunk in response.aiter_bytes()]) == b"".join(events)
        assert get_captured_dollar_cost() == 0.002145
        await client.aclose()
