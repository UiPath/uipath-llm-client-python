"""Tests for captured response headers functionality.

Tests that LLM Gateway response headers are captured and surfaced
in LangChain's response_metadata on AIMessage objects.
"""

import json
import os
from unittest.mock import patch

import httpx
import pytest
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat

from uipath.llm_client.httpx_client import (
    UiPathHttpxAsyncClient,
    UiPathHttpxClient,
)
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.headers import (
    extract_matching_headers,
    get_captured_response_headers,
    set_captured_response_headers,
)

# ============================================================================
# Fixtures
# ============================================================================

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}

SAMPLE_GATEWAY_HEADERS = {
    "X-UiPath-RequestId": "req-123",
    "X-UiPath-TraceId": "trace-456",
    "X-UiPath-ModelVersion": "2024-01-01",
    "Content-Type": "application/json",
    "X-RateLimit-Remaining": "99",
}

CHAT_RESPONSE_JSON = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}

STREAM_CHUNKS = [
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    },
]


class MockTransport(httpx.BaseTransport):
    """Mock transport that returns configurable responses with custom headers."""

    def __init__(
        self,
        response_json: dict | None = None,
        response_headers: dict[str, str] | None = None,
        stream_chunks: list[dict] | None = None,
    ):
        self._response_json = response_json or CHAT_RESPONSE_JSON
        self._response_headers = response_headers or SAMPLE_GATEWAY_HEADERS
        self._stream_chunks = stream_chunks

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        headers = dict(self._response_headers)
        if self._stream_chunks and request.headers.get("X-UiPath-Streaming-Enabled") == "true":
            lines = []
            for chunk in self._stream_chunks:
                lines.append(f"data: {json.dumps(chunk)}\n")
            lines.append("data: [DONE]\n")
            content = "\n".join(lines).encode()
            headers["content-type"] = "text/event-stream"
            return httpx.Response(200, content=content, headers=headers)

        content = json.dumps(self._response_json).encode()
        headers["content-type"] = "application/json"
        return httpx.Response(200, content=content, headers=headers)


class MockAsyncTransport(httpx.AsyncBaseTransport):
    """Async version of MockTransport."""

    def __init__(
        self,
        response_json: dict | None = None,
        response_headers: dict[str, str] | None = None,
        stream_chunks: list[dict] | None = None,
    ):
        self._sync = MockTransport(response_json, response_headers, stream_chunks)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._sync.handle_request(request)


@pytest.fixture(autouse=True)
def clear_singletons():
    """Clear singleton instances before each test."""
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


@pytest.fixture
def llmgw_settings():
    with patch.dict(os.environ, LLMGW_ENV, clear=True):
        return LLMGatewaySettings()


def _make_normalized_chat(
    settings: LLMGatewaySettings,
    response_headers: dict[str, str] | None = None,
    stream_chunks: list[dict] | None = None,
    captured_headers: tuple[str, ...] = ("x-uipath-",),
) -> UiPathChat:
    """Create a UiPathChat with a mock transport."""
    transport = MockTransport(
        response_headers=response_headers or SAMPLE_GATEWAY_HEADERS,
        stream_chunks=stream_chunks or STREAM_CHUNKS,
    )
    chat = UiPathChat(
        model="gpt-4o",
        settings=settings,
        captured_headers=captured_headers,
    )
    # Replace the cached httpx client with one using our mock transport
    sync_client = UiPathHttpxClient(
        base_url="https://cloud.uipath.com/test-org-id/test-tenant-id/llmgateway_/api/chat/completions",
        model_name="gpt-4o",
        transport=transport,
        captured_headers=captured_headers,
    )
    async_transport = MockAsyncTransport(
        response_headers=response_headers or SAMPLE_GATEWAY_HEADERS,
        stream_chunks=stream_chunks or STREAM_CHUNKS,
    )
    async_client = UiPathHttpxAsyncClient(
        base_url="https://cloud.uipath.com/test-org-id/test-tenant-id/llmgateway_/api/chat/completions",
        model_name="gpt-4o",
        transport=async_transport,
        captured_headers=captured_headers,
    )
    # Override cached properties by setting instance attributes directly
    # (cached_property stores values in instance.__dict__, so this takes precedence)
    object.__setattr__(chat, "uipath_sync_client", sync_client)
    object.__setattr__(chat, "uipath_async_client", async_client)
    return chat


# ============================================================================
# Test extract_matching_headers
# ============================================================================


class TestExtractMatchingHeaders:
    """Tests for the header extraction helper."""

    def test_extracts_matching_prefixes(self):
        headers = httpx.Headers(SAMPLE_GATEWAY_HEADERS)
        result = extract_matching_headers(headers, ("x-uipath-",))
        assert "x-uipath-requestid" in result
        assert "x-uipath-traceid" in result
        assert "x-uipath-modelversion" in result
        assert "content-type" not in result
        assert "x-ratelimit-remaining" not in result

    def test_case_insensitive_matching(self):
        headers = httpx.Headers({"X-UiPath-Foo": "bar", "x-uipath-baz": "qux"})
        result = extract_matching_headers(headers, ("X-UIPATH-",))
        assert len(result) == 2

    def test_multiple_prefixes(self):
        headers = httpx.Headers(SAMPLE_GATEWAY_HEADERS)
        result = extract_matching_headers(headers, ("x-uipath-", "x-ratelimit-"))
        assert "x-uipath-requestid" in result
        assert "x-ratelimit-remaining" in result
        assert "content-type" not in result

    def test_empty_prefixes(self):
        headers = httpx.Headers(SAMPLE_GATEWAY_HEADERS)
        result = extract_matching_headers(headers, ())
        assert result == {}

    def test_no_matches(self):
        headers = httpx.Headers({"Content-Type": "application/json"})
        result = extract_matching_headers(headers, ("x-uipath-",))
        assert result == {}


# ============================================================================
# Test ContextVar functions
# ============================================================================


class TestContextVarFunctions:
    """Tests for get/set captured response headers."""

    def test_get_returns_empty_by_default(self):
        """ContextVar default is {}; get() should return {}."""
        set_captured_response_headers({})
        assert get_captured_response_headers() == {}

    def test_set_and_get(self):
        set_captured_response_headers({"X-UiPath-Foo": "bar"})
        assert get_captured_response_headers() == {"X-UiPath-Foo": "bar"}
        # Overwrite with empty clears
        set_captured_response_headers({})
        assert get_captured_response_headers() == {}

    def test_get_returns_copy(self):
        """Verify get_captured_response_headers returns a copy, not the ContextVar reference."""
        original = {"X-UiPath-Foo": "bar"}
        set_captured_response_headers(original)
        result = get_captured_response_headers()
        result["X-UiPath-New"] = "new"
        assert "X-UiPath-New" not in get_captured_response_headers()
        set_captured_response_headers({})


# ============================================================================
# Test UiPathHttpxClient header capture in send()
# ============================================================================


class TestHttpxClientHeaderCapture:
    """Tests that the httpx client captures headers in send()."""

    def test_send_captures_matching_headers(self):
        transport = MockTransport()
        client = UiPathHttpxClient(
            base_url="https://example.com",
            transport=transport,
            captured_headers=("x-uipath-",),
        )
        set_captured_response_headers({})
        client.get("/")
        captured = get_captured_response_headers()
        assert "x-uipath-requestid" in captured
        assert "x-uipath-traceid" in captured
        assert "content-type" not in captured
        client.close()

    def test_send_with_empty_captured_headers_does_not_capture(self):
        transport = MockTransport()
        client = UiPathHttpxClient(
            base_url="https://example.com",
            transport=transport,
            captured_headers=(),
        )
        set_captured_response_headers({})
        client.get("/")
        assert get_captured_response_headers() == {}
        client.close()

    def test_send_with_custom_prefixes(self):
        transport = MockTransport()
        client = UiPathHttpxClient(
            base_url="https://example.com",
            transport=transport,
            captured_headers=("x-ratelimit-",),
        )
        set_captured_response_headers({})
        client.get("/")
        captured = get_captured_response_headers()
        assert "x-ratelimit-remaining" in captured
        assert "x-uipath-requestid" not in captured
        client.close()

    @pytest.mark.asyncio
    async def test_async_send_captures_headers(self):
        async_transport = MockAsyncTransport()
        client = UiPathHttpxAsyncClient(
            base_url="https://example.com",
            transport=async_transport,
            captured_headers=("x-uipath-",),
        )
        set_captured_response_headers({})
        await client.get("/")
        captured = get_captured_response_headers()
        assert "x-uipath-requestid" in captured
        await client.aclose()


# ============================================================================
# Test Normalized Client (UiPathChat) header capture
# ============================================================================


class TestNormalizedClientHeaderCapture:
    """Tests for header capture in the normalized UiPathChat client."""

    def test_generate_captures_headers(self, llmgw_settings):
        chat = _make_normalized_chat(llmgw_settings)
        result = chat.invoke("Hello")
        assert "headers" in result.response_metadata
        gateway_headers = result.response_metadata["headers"]
        assert "x-uipath-requestid" in gateway_headers
        assert "x-uipath-traceid" in gateway_headers
        assert "content-type" not in gateway_headers

    @pytest.mark.asyncio
    async def test_agenerate_captures_headers(self, llmgw_settings):
        chat = _make_normalized_chat(llmgw_settings)
        result = await chat.ainvoke("Hello")
        assert "headers" in result.response_metadata
        gateway_headers = result.response_metadata["headers"]
        assert "x-uipath-requestid" in gateway_headers

    def test_stream_captures_headers_on_first_chunk(self, llmgw_settings):
        chat = _make_normalized_chat(llmgw_settings)
        chunks = list(chat.stream("Hello"))
        assert len(chunks) >= 1
        # First chunk should have gateway headers
        first_chunk = chunks[0]
        assert "headers" in first_chunk.response_metadata
        gateway_headers = first_chunk.response_metadata["headers"]
        assert "x-uipath-requestid" in gateway_headers
        # Later chunks should not have gateway headers
        if len(chunks) > 1:
            assert "headers" not in chunks[1].response_metadata

    @pytest.mark.asyncio
    async def test_astream_captures_headers_on_first_chunk(self, llmgw_settings):
        chat = _make_normalized_chat(llmgw_settings)
        chunks = []
        async for chunk in chat.astream("Hello"):
            chunks.append(chunk)
        assert len(chunks) >= 1
        first_chunk = chunks[0]
        assert "headers" in first_chunk.response_metadata

    def test_custom_prefixes(self, llmgw_settings):
        chat = _make_normalized_chat(
            llmgw_settings,
            captured_headers=("x-uipath-", "x-ratelimit-"),
        )
        result = chat.invoke("Hello")
        gateway_headers = result.response_metadata["headers"]
        assert "x-uipath-requestid" in gateway_headers
        assert "x-ratelimit-remaining" in gateway_headers

    def test_disabled_capture(self, llmgw_settings):
        chat = _make_normalized_chat(llmgw_settings, captured_headers=())
        result = chat.invoke("Hello")
        assert "headers" not in result.response_metadata

    def test_no_matching_headers(self, llmgw_settings):
        chat = _make_normalized_chat(
            llmgw_settings,
            response_headers={"Content-Type": "application/json"},
        )
        result = chat.invoke("Hello")
        # No matching headers, so the key should not be present
        assert "headers" not in result.response_metadata


# ============================================================================
# Test UiPathBaseChatModel wrapping (for passthrough clients)
# ============================================================================


class TestBaseChatModelWrapping:
    """Tests that UiPathBaseChatModel wrappers inject headers for passthrough clients.

    Since we can't easily instantiate real passthrough clients without vendor SDKs,
    we test the wrapping logic via the ContextVar mechanism directly.
    """

    def test_inject_gateway_headers_populates_result(self, llmgw_settings):
        """Test that _inject_gateway_headers reads from ContextVar."""
        chat = _make_normalized_chat(llmgw_settings)

        # Simulate what send() does: store headers in ContextVar
        set_captured_response_headers({"x-uipath-requestid": "test-123"})

        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="test", response_metadata={}))]
        )
        chat._inject_gateway_headers(result.generations)
        assert result.generations[0].message.response_metadata["headers"] == {
            "x-uipath-requestid": "test-123"
        }
        set_captured_response_headers({})

    def test_inject_gateway_headers_skipped_when_disabled(self, llmgw_settings):
        """Test that _inject_gateway_headers is skipped when captured_headers is empty."""
        chat = _make_normalized_chat(llmgw_settings, captured_headers=())
        set_captured_response_headers({"x-uipath-requestid": "test-123"})

        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="test", response_metadata={}))]
        )
        chat._inject_gateway_headers(result.generations)
        assert "headers" not in result.generations[0].message.response_metadata
        set_captured_response_headers({})
