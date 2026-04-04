"""Tests for dynamic request header injection via UiPathDynamicHeadersCallback.

Tests that headers set via callbacks are injected into outgoing LLM gateway
requests through the ContextVar mechanism in httpx_client.send().

Design of OtelHeadersCallback:
  openinference-instrumentation-langchain intentionally does NOT attach its spans
  to the Python context (to avoid leaked contexts on errors). This means
  get_current_span() returns the *caller's* outer span — which is exactly the
  right behaviour for distributed tracing: the application creates a parent span,
  calls LangChain, and the callback propagates that parent span's context to the
  LLM gateway so the gateway request becomes part of the same trace.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from uipath_langchain_client import UiPathDynamicHeadersCallback
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)


@contextmanager
def active_span(tracer: Tracer, name: str) -> Iterator[Span]:
    """Start a named span and yield the current span via get_current_span()."""
    with tracer.start_as_current_span(name):
        yield trace.get_current_span()


# ============================================================================
# OtelHeadersCallback — reads the active span via get_current_span()
# ============================================================================


class OtelHeadersCallback(UiPathDynamicHeadersCallback):
    """Injects the active OTEL span's trace and span IDs into each LLM gateway request.

    Calls get_current_span() to read the span that is active in the caller's
    context (e.g. an outer application span). openinference-instrumentation-langchain
    creates its own child spans but deliberately does not attach them to the
    Python context, so get_current_span() always reflects the caller's outer span.

    This propagates the application's trace context to the LLM gateway, making
    the gateway request part of the same distributed trace.
    """

    def get_headers(self) -> dict[str, str]:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return {}
        return {
            "x-trace-id": format(ctx.trace_id, "032x"),
            "x-span-id": format(ctx.span_id, "016x"),
        }


# ============================================================================
# Fixtures & helpers
# ============================================================================

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}

_CHAT_RESPONSE = (
    b'{"id":"x","object":"chat.completion","created":1,"model":"gpt-4o",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},'
    b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
)


class RequestCapturingTransport(httpx.BaseTransport):
    """Sync transport that records the last request's headers."""

    def __init__(self):
        self.last_request_headers: dict[str, str] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request_headers = dict(request.headers)
        return httpx.Response(
            200, content=_CHAT_RESPONSE, headers={"content-type": "application/json"}
        )


class AsyncRequestCapturingTransport(httpx.AsyncBaseTransport):
    """Async transport that records the last request's headers."""

    def __init__(self):
        self._sync = RequestCapturingTransport()

    @property
    def last_request_headers(self) -> dict[str, str]:
        return self._sync.last_request_headers

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._sync.handle_request(request)


@pytest.fixture(scope="module")
def otel_exporter():
    """Module-scoped OTEL setup with LangChain instrumentation.

    TracerProvider + InMemorySpanExporter + LangChainInstrumentor are created
    once for the whole module. Individual tests clear the exporter via the
    `tracer` fixture before each run.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    propagate.set_global_textmap(TraceContextTextMapPropagator())
    LangChainInstrumentor().instrument()
    yield exporter
    LangChainInstrumentor().uninstrument()


@pytest.fixture
def tracer(otel_exporter: InMemorySpanExporter) -> Tracer:
    """Per-test tracer; clears the exporter so spans from previous tests don't leak."""
    otel_exporter.clear()
    return trace.get_tracer("uipath-test")


@pytest.fixture(autouse=True)
def clear_singletons():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


@pytest.fixture(autouse=True)
def reset_dynamic_headers():
    set_dynamic_request_headers({})
    yield
    set_dynamic_request_headers({})


@pytest.fixture
def llmgw_settings():
    with patch.dict(os.environ, LLMGW_ENV, clear=True):
        return LLMGatewaySettings()


def _make_chat_with_sync_transport(
    settings: LLMGatewaySettings,
    transport: RequestCapturingTransport,
) -> UiPathChat:
    chat = UiPathChat(model="gpt-4o", settings=settings)
    sync_client = UiPathHttpxClient(
        base_url="https://cloud.uipath.com/test-org-id/test-tenant-id/llmgateway_/api/chat/completions",
        model_name="gpt-4o",
        transport=transport,
    )
    object.__setattr__(chat, "uipath_sync_client", sync_client)
    return chat


def _make_chat_with_async_transport(
    settings: LLMGatewaySettings,
    transport: AsyncRequestCapturingTransport,
) -> UiPathChat:
    chat = UiPathChat(model="gpt-4o", settings=settings)
    async_client = UiPathHttpxAsyncClient(
        base_url="https://cloud.uipath.com/test-org-id/test-tenant-id/llmgateway_/api/chat/completions",
        model_name="gpt-4o",
        transport=transport,
    )
    object.__setattr__(chat, "uipath_async_client", async_client)
    return chat


# ============================================================================
# Test ContextVar helpers
# ============================================================================


class TestDynamicRequestHeadersContextVar:
    def test_default_is_empty(self):
        assert get_dynamic_request_headers() == {}

    def test_set_and_get(self):
        set_dynamic_request_headers({"X-Custom": "value"})
        assert get_dynamic_request_headers() == {"X-Custom": "value"}

    def test_set_empty_clears(self):
        set_dynamic_request_headers({"X-Custom": "value"})
        set_dynamic_request_headers({})
        assert get_dynamic_request_headers() == {}

    def test_get_returns_copy(self):
        set_dynamic_request_headers({"X-Custom": "value"})
        result = get_dynamic_request_headers()
        result["X-New"] = "other"
        assert "X-New" not in get_dynamic_request_headers()


# ============================================================================
# Test callback lifecycle
# ============================================================================


class TestCallbackLifecycle:
    def test_run_inline_is_true(self):
        """run_inline ensures on_chat_model_start runs in the caller's coroutine."""
        assert OtelHeadersCallback().run_inline is True

    def test_on_chat_model_start_sets_headers(self, tracer):
        """on_chat_model_start injects the active span's headers into the ContextVar."""
        cb = OtelHeadersCallback()
        with active_span(tracer, "llm-call"):
            cb.on_chat_model_start({}, [[]])
        assert "x-trace-id" in get_dynamic_request_headers()

    def test_on_chat_model_start_no_span_sets_empty(self):
        """When there is no active span, on_chat_model_start clears the ContextVar."""
        set_dynamic_request_headers({"x-stale": "value"})
        OtelHeadersCallback().on_chat_model_start({}, [[]])
        assert get_dynamic_request_headers() == {}


# ============================================================================
# Test OtelHeadersCallback.get_headers()
# ============================================================================


class TestOtelHeadersCallback:
    def test_no_active_span_returns_empty(self):
        assert OtelHeadersCallback().get_headers() == {}

    def test_active_span_injects_trace_and_span_ids(self, tracer):
        cb = OtelHeadersCallback()
        with active_span(tracer, "test-span") as span:
            headers = cb.get_headers()
        ctx = span.get_span_context()
        assert headers["x-trace-id"] == format(ctx.trace_id, "032x")
        assert headers["x-span-id"] == format(ctx.span_id, "016x")

    def test_trace_id_is_32_hex_chars(self, tracer):
        cb = OtelHeadersCallback()
        with active_span(tracer, "test-span"):
            headers = cb.get_headers()
        assert len(headers["x-trace-id"]) == 32
        assert all(c in "0123456789abcdef" for c in headers["x-trace-id"])

    def test_span_id_is_16_hex_chars(self, tracer):
        cb = OtelHeadersCallback()
        with active_span(tracer, "test-span"):
            headers = cb.get_headers()
        assert len(headers["x-span-id"]) == 16
        assert all(c in "0123456789abcdef" for c in headers["x-span-id"])

    def test_different_spans_produce_different_span_ids(self, tracer):
        cb = OtelHeadersCallback()
        with active_span(tracer, "span-a"):
            headers_a = cb.get_headers()
        with active_span(tracer, "span-b"):
            headers_b = cb.get_headers()
        assert headers_a["x-span-id"] != headers_b["x-span-id"]


# ============================================================================
# Test httpx client injects dynamic headers in send()
# ============================================================================


class TestHttpxClientDynamicHeaderInjection:
    def test_sync_client_injects_headers(self):
        transport = RequestCapturingTransport()
        client = UiPathHttpxClient(base_url="https://example.com", transport=transport)
        set_dynamic_request_headers({"x-custom-trace": "trace-123"})
        client.get("/")
        assert transport.last_request_headers.get("x-custom-trace") == "trace-123"
        client.close()

    def test_sync_client_no_injection_when_empty(self):
        transport = RequestCapturingTransport()
        client = UiPathHttpxClient(base_url="https://example.com", transport=transport)
        client.get("/")
        assert "x-custom-trace" not in transport.last_request_headers
        client.close()

    @pytest.mark.asyncio
    async def test_async_client_injects_headers(self):
        transport = AsyncRequestCapturingTransport()
        client = UiPathHttpxAsyncClient(base_url="https://example.com", transport=transport)
        set_dynamic_request_headers({"x-custom-trace": "trace-456"})
        await client.get("/")
        assert transport.last_request_headers.get("x-custom-trace") == "trace-456"
        await client.aclose()

    def test_dynamic_headers_can_override_defaults(self):
        transport = RequestCapturingTransport()
        client = UiPathHttpxClient(base_url="https://example.com", transport=transport)
        set_dynamic_request_headers({"X-UiPath-LLMGateway-TimeoutSeconds": "60"})
        client.get("/")
        assert transport.last_request_headers.get("x-uipath-llmgateway-timeoutseconds") == "60"
        client.close()


# ============================================================================
# End-to-end: openinference instruments LangChain, callback propagates outer span
#
# openinference does NOT attach its spans to the Python context (intentional
# design to avoid leaked contexts). get_current_span() therefore returns the
# caller's outer span — perfect for propagating application trace context to
# the LLM gateway.
# ============================================================================


class TestOpenInferenceEndToEnd:
    def test_outer_span_context_injected_into_request(self, otel_exporter, llmgw_settings, tracer):
        """Outer span is the current span; its context is injected into the LLM request.
        openinference creates child spans (same trace_id) but doesn't override the
        current span context, so get_current_span() returns the outer span."""
        transport = RequestCapturingTransport()
        chat = _make_chat_with_sync_transport(llmgw_settings, transport)
        cb = OtelHeadersCallback()

        with active_span(tracer, "my-app-operation") as outer:
            chat.invoke("Hello!", config={"callbacks": [cb]})

        # openinference created child spans
        spans = otel_exporter.get_finished_spans()
        assert len(spans) > 0

        outer_trace_id = format(outer.get_span_context().trace_id, "032x")
        outer_span_id = format(outer.get_span_context().span_id, "016x")

        # The outer span's context was injected into the HTTP request
        assert transport.last_request_headers.get("x-trace-id") == outer_trace_id
        assert transport.last_request_headers.get("x-span-id") == outer_span_id

        # All openinference spans belong to the same trace
        for span in spans:
            assert format(span.context.trace_id, "032x") == outer_trace_id

    @pytest.mark.asyncio
    async def test_async_outer_span_context_injected(self, otel_exporter, llmgw_settings, tracer):
        """Async path: outer span context propagates through ainvoke to the request."""
        transport = AsyncRequestCapturingTransport()
        chat = _make_chat_with_async_transport(llmgw_settings, transport)
        cb = OtelHeadersCallback()

        with active_span(tracer, "async-app-operation") as outer:
            await chat.ainvoke("Hello!", config={"callbacks": [cb]})

        outer_trace_id = format(outer.get_span_context().trace_id, "032x")
        outer_span_id = format(outer.get_span_context().span_id, "016x")

        assert transport.last_request_headers.get("x-trace-id") == outer_trace_id
        assert transport.last_request_headers.get("x-span-id") == outer_span_id

    def test_no_outer_span_no_headers_but_spans_still_created(self, otel_exporter, llmgw_settings):
        """Without an outer span, get_current_span() is invalid → no headers injected.
        openinference still creates its own root spans for observability."""
        transport = RequestCapturingTransport()
        chat = _make_chat_with_sync_transport(llmgw_settings, transport)
        cb = OtelHeadersCallback()

        chat.invoke("Hello!", config={"callbacks": [cb]})

        # openinference still exports spans
        spans = otel_exporter.get_finished_spans()
        assert len(spans) > 0

        # But no trace headers were injected (no outer span to propagate)
        assert "x-trace-id" not in transport.last_request_headers
        assert "x-span-id" not in transport.last_request_headers

    def test_headers_cleared_after_invoke(self, otel_exporter, llmgw_settings, tracer):
        """ContextVar is reset to empty after the call completes."""
        transport = RequestCapturingTransport()
        chat = _make_chat_with_sync_transport(llmgw_settings, transport)
        cb = OtelHeadersCallback()

        with active_span(tracer, "my-operation"):
            chat.invoke("Hello!", config={"callbacks": [cb]})

        assert get_dynamic_request_headers() == {}

    def test_sequential_calls_propagate_different_outer_spans(
        self, otel_exporter, llmgw_settings, tracer
    ):
        """Each call with a different outer span gets different span IDs in headers."""
        transport = RequestCapturingTransport()
        chat = _make_chat_with_sync_transport(llmgw_settings, transport)
        cb = OtelHeadersCallback()

        with active_span(tracer, "first-operation") as first:
            chat.invoke("First", config={"callbacks": [cb]})
        first_span_id = transport.last_request_headers.get("x-span-id")

        with active_span(tracer, "second-operation") as second:
            chat.invoke("Second", config={"callbacks": [cb]})
        second_span_id = transport.last_request_headers.get("x-span-id")

        assert first_span_id == format(first.get_span_context().span_id, "016x")
        assert second_span_id == format(second.get_span_context().span_id, "016x")
        assert first_span_id != second_span_id
