"""Tests for LLM Gateway response capture into LangChain's response_metadata.

Covers captured response headers and the opt-in per-request dollar cost, for
both the non-streaming and the streaming paths.
"""

import json
import os
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

from tests.lazy_stream import LazyByteStream
from uipath.llm_client.httpx_client import (
    UiPathHttpxAsyncClient,
    UiPathHttpxClient,
)
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.dollar_cost import (
    INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER,
    set_captured_dollar_cost,
)
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
        stream_cost: float | None = None,
    ):
        self._response_json = response_json or CHAT_RESPONSE_JSON
        self._response_headers = response_headers or SAMPLE_GATEWAY_HEADERS
        self._stream_chunks = stream_chunks
        self._stream_cost = stream_cost

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        headers = dict(self._response_headers)
        if self._stream_chunks and request.headers.get("X-UiPath-Streaming-Enabled") == "true":
            events = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in self._stream_chunks]
            events.append(b"data: [DONE]\n\n")
            if self._stream_cost is not None:
                # As the gateway does: the cost frame trails the terminal event.
                cost_json = json.dumps({"associated_dollar_cost": self._stream_cost})
                events.append(f"data: {cost_json}\n\n".encode())
            headers["content-type"] = "text/event-stream"
            # One lazy chunk per event, so the SDK can stop at [DONE] before the cost frame.
            return httpx.Response(200, stream=LazyByteStream(events), headers=headers)

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
        stream_cost: float | None = None,
    ):
        self._sync = MockTransport(response_json, response_headers, stream_chunks, stream_cost)

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


# ============================================================================
# Test opt-in per-request dollar cost capture
# ============================================================================


def _make_passthrough_chat(
    settings: LLMGatewaySettings,
    response_json: dict,
    default_headers: dict[str, str] | None = None,
    stream_cost: float | None = None,
    **model_kwargs: Any,
) -> UiPathChatOpenAI:
    """Passthrough model with the real openai SDK in the loop, since that is what re-parses the body."""
    chat = UiPathChatOpenAI(
        model="gpt-4o-2024-11-20",
        settings=settings,
        model_details={},
        default_headers=default_headers,
        **model_kwargs,
    )
    chat.uipath_sync_client._transport = MockTransport(  # type: ignore[attr-defined]
        response_json=response_json, stream_chunks=STREAM_CHUNKS, stream_cost=stream_cost
    )
    chat.uipath_sync_client._mounts = {}  # type: ignore[attr-defined]
    chat.uipath_async_client._transport = MockAsyncTransport(  # type: ignore[attr-defined]
        response_json=response_json, stream_chunks=STREAM_CHUNKS, stream_cost=stream_cost
    )
    chat.uipath_async_client._mounts = {}  # type: ignore[attr-defined]
    return chat


OPT_IN_HEADERS = {INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"}
PRICED_RESPONSE_JSON = {**CHAT_RESPONSE_JSON, "associated_dollar_cost": 0.002145}


class TestDollarCostCapture:
    """The gateway's opt-in per-request dollar cost ends up in response_metadata."""

    def test_invoke_surfaces_cost_when_opted_in(self, llmgw_settings):
        chat = _make_passthrough_chat(
            llmgw_settings, response_json=PRICED_RESPONSE_JSON, default_headers=OPT_IN_HEADERS
        )
        result = chat.invoke("Hello")
        assert result.response_metadata["associated_dollar_cost"] == 0.002145

    @pytest.mark.asyncio
    async def test_ainvoke_surfaces_cost_when_opted_in(self, llmgw_settings):
        chat = _make_passthrough_chat(
            llmgw_settings, response_json=PRICED_RESPONSE_JSON, default_headers=OPT_IN_HEADERS
        )
        result = await chat.ainvoke("Hello")
        assert result.response_metadata["associated_dollar_cost"] == 0.002145

    def test_invoke_omits_cost_when_gateway_did_not_price_the_call(self, llmgw_settings):
        """Not priced must never surface as $0: the key must be absent."""
        chat = _make_passthrough_chat(
            llmgw_settings, response_json=CHAT_RESPONSE_JSON, default_headers=OPT_IN_HEADERS
        )
        result = chat.invoke("Hello")
        assert "associated_dollar_cost" not in result.response_metadata

    def test_invoke_omits_cost_when_not_opted_in(self, llmgw_settings):
        """Without the opt-in header the field cannot be gateway-authored."""
        chat = _make_passthrough_chat(llmgw_settings, response_json=PRICED_RESPONSE_JSON)
        result = chat.invoke("Hello")
        assert "associated_dollar_cost" not in result.response_metadata

    def test_stream_surfaces_cost_on_trailing_empty_chunk(self, llmgw_settings):
        """The cost is only known after the last vendor chunk, so it rides on one extra chunk."""
        chat = _make_passthrough_chat(
            llmgw_settings,
            response_json=CHAT_RESPONSE_JSON,
            default_headers=OPT_IN_HEADERS,
            stream_cost=0.002145,
        )
        chunks = list(chat.stream("Hello"))
        assert "".join(str(c.content) for c in chunks) == "Hello!"
        priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
        assert len(priced) == 1
        assert priced[0].content == ""
        assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145
        merged = chunks[0]
        for chunk in chunks[1:]:
            merged = merged + chunk
        assert merged.response_metadata["associated_dollar_cost"] == 0.002145

    @pytest.mark.asyncio
    async def test_astream_surfaces_cost_on_trailing_empty_chunk(self, llmgw_settings):
        chat = _make_passthrough_chat(
            llmgw_settings,
            response_json=CHAT_RESPONSE_JSON,
            default_headers=OPT_IN_HEADERS,
            stream_cost=0.002145,
        )
        chunks = [chunk async for chunk in chat.astream("Hello")]
        assert "".join(str(c.content) for c in chunks) == "Hello!"
        priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
        assert len(priced) == 1
        assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145

    def test_stream_omits_cost_when_gateway_did_not_price_the_call(self, llmgw_settings):
        chat = _make_passthrough_chat(
            llmgw_settings, response_json=CHAT_RESPONSE_JSON, default_headers=OPT_IN_HEADERS
        )
        chunks = list(chat.stream("Hello"))
        assert "".join(str(c.content) for c in chunks) == "Hello!"
        assert all("associated_dollar_cost" not in c.response_metadata for c in chunks)

    def test_invoke_routed_through_streaming_surfaces_cost(self, llmgw_settings):
        """streaming=True routes invoke() through _stream."""
        chat = _make_passthrough_chat(
            llmgw_settings,
            response_json=CHAT_RESPONSE_JSON,
            default_headers=OPT_IN_HEADERS,
            stream_cost=0.002145,
            streaming=True,
        )
        result = chat.invoke("Hello")
        assert result.content == "Hello!"
        assert result.response_metadata["associated_dollar_cost"] == 0.002145

    def test_inject_dollar_cost_populates_result(self, llmgw_settings):
        chat = _make_normalized_chat(llmgw_settings)
        set_captured_dollar_cost(0.002145)

        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="test", response_metadata={}))]
        )
        chat._inject_dollar_cost(result.generations)
        assert result.generations[0].message.response_metadata["associated_dollar_cost"] == 0.002145

    def test_inject_dollar_cost_skipped_when_absent(self, llmgw_settings):
        chat = _make_normalized_chat(llmgw_settings)

        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="test", response_metadata={}))]
        )
        chat._inject_dollar_cost(result.generations)
        assert "associated_dollar_cost" not in result.generations[0].message.response_metadata
