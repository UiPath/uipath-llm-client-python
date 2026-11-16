"""In-stream provider errors after HTTP 200 are typed and retried (PC-5454).

A streamed OpenAI / Azure OpenAI call can get HTTP 200 and then an error event
inside the stream ("Service Unavailable", "Our servers are currently
overloaded"). The openai SDK raises a bare ``openai.APIError`` with no response;
the transport retryer already let the 200 through. These tests drive the real
openai SDK and langchain-openai through a mock transport and check that:

* the error becomes a typed, ``in_stream`` :class:`UiPathServiceUnavailableError`;
* the chat model restarts the stream when the error arrives before any output;
* it does not restart once output has reached the caller, nor on HTTP errors the
  transport already retried, nor when retries are disabled.
"""

import json
import os
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest
from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.exceptions import (
    UiPathError,
    UiPathServiceUnavailableError,
)

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}

NO_WAIT = {"initial_delay": 0.0, "max_delay": 0.0, "jitter": 0.0}


@pytest.fixture(autouse=True)
def clear_singletons():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


@pytest.fixture
def llmgw_settings():
    with patch.dict(os.environ, LLMGW_ENV, clear=True):
        return LLMGatewaySettings()


# --- Responses API SSE bodies ------------------------------------------------


def _response_obj(status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-4o-mini-2024-07-18",
        "status": status,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": (
            {
                "input_tokens": 3,
                "output_tokens": 1,
                "total_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
            if status == "completed"
            else None
        ),
    }


_MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
}


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _created() -> dict[str, Any]:
    return {
        "type": "response.created",
        "sequence_number": 0,
        "response": _response_obj("in_progress", []),
    }


def _delta() -> dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hello",
        "logprobs": [],
    }


def _error(message: str) -> dict[str, Any]:
    # The shape openai._streaming raises APIError on: a top-level "error" object.
    return {
        "type": "error",
        "sequence_number": 2,
        "error": {"type": "service_unavailable_error", "code": None, "message": message},
    }


SUCCESS = _sse(
    [
        _created(),
        _delta(),
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": _response_obj("completed", [_MESSAGE]),
        },
    ]
)
FAIL_BEFORE_OUTPUT = _sse([_created(), _error("Service Unavailable")])
FAIL_AFTER_OUTPUT = _sse([_created(), _delta(), _error("Service Unavailable")])


def _stream_response(request: httpx.Request, body: bytes) -> httpx.Response:
    return httpx.Response(
        200, request=request, content=body, headers={"content-type": "text/event-stream"}
    )


class _Scripted(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Answers each request with the next scripted (status, body)."""

    def __init__(self, *script: tuple[int, bytes]) -> None:
        self.script = list(script)
        self.calls = 0

    def _next(self, request: httpx.Request) -> httpx.Response:
        status, body = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if status == 200:
            return _stream_response(request, body)
        return httpx.Response(status, request=request, json={"error": {"message": "busy"}})

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._next(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._next(request)


def _chat(settings: LLMGatewaySettings, transport: _Scripted, **kwargs: Any) -> UiPathChatOpenAI:
    kwargs.setdefault("retry_config", NO_WAIT)
    chat = UiPathChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        settings=settings,
        model_details={},
        api_flavor="responses",
        **kwargs,
    )
    # The openai SDK uses these httpx clients; rerouting their transport makes
    # the real SDK parse the scripted SSE.
    for client in (chat.uipath_sync_client, chat.uipath_async_client):
        client._transport = transport  # type: ignore[attr-defined]
        client._mounts = {}  # type: ignore[attr-defined]
    return chat


def _text(chunks: list[Any]) -> str:
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return merged.text


class TestInStreamErrorIsTyped:
    def test_error_before_output_becomes_in_stream_503_when_retries_exhausted(self, llmgw_settings):
        transport = _Scripted((200, FAIL_BEFORE_OUTPUT))
        chat = _chat(llmgw_settings, transport, max_retries=2)
        with pytest.raises(UiPathServiceUnavailableError) as info:
            list(chat.stream("hi"))
        exc = info.value
        assert exc.in_stream is True
        assert exc.status_code == 503
        assert exc.detail == "Service Unavailable"
        assert exc.error_code == "service_unavailable_error"
        assert isinstance(exc.__cause__, openai.APIError)
        assert not isinstance(exc, openai.APIError)
        assert transport.calls == 2  # max_retries attempts in total, like the transport

    def test_unrelated_in_stream_error_stays_untyped(self, llmgw_settings):
        body = _sse(
            [
                _created(),
                {
                    "type": "error",
                    "sequence_number": 1,
                    "error": {"type": "invalid_request_error", "message": "bad input"},
                },
            ]
        )
        transport = _Scripted((200, body))
        chat = _chat(llmgw_settings, transport)
        with pytest.raises(UiPathError) as info:
            list(chat.stream("hi"))
        assert type(info.value) is UiPathError
        assert transport.calls == 1


class TestInStreamErrorIsRetried:
    def test_sync_stream_retries_error_before_output(self, llmgw_settings):
        transport = _Scripted((200, FAIL_BEFORE_OUTPUT), (200, SUCCESS))
        chat = _chat(llmgw_settings, transport)
        chunks = list(chat.stream("hi"))
        assert transport.calls == 2
        assert _text(chunks) == "hello"
        # The failed attempt's response.created chunk is dropped, not merged in.
        created = [c for c in chunks if c.response_metadata.get("id") == "resp_1"]
        assert len(created) <= 1

    @pytest.mark.asyncio
    async def test_async_stream_retries_error_before_output(self, llmgw_settings):
        transport = _Scripted((200, FAIL_BEFORE_OUTPUT), (200, SUCCESS))
        chat = _chat(llmgw_settings, transport)
        chunks = [c async for c in chat.astream("hi")]
        assert transport.calls == 2
        assert _text(chunks) == "hello"

    def test_invoke_with_streaming_retries(self, llmgw_settings):
        transport = _Scripted((200, FAIL_BEFORE_OUTPUT), (200, SUCCESS))
        chat = _chat(llmgw_settings, transport, streaming=True)
        assert chat.invoke("hi").text == "hello"
        assert transport.calls == 2

    def test_overloaded_message_is_retried(self, llmgw_settings):
        body = _sse(
            [
                _created(),
                {
                    "type": "error",
                    "sequence_number": 1,
                    "error": {"message": "Our servers are currently overloaded"},
                },
            ]
        )
        transport = _Scripted((200, body), (200, SUCCESS))
        chat = _chat(llmgw_settings, transport)
        assert _text(list(chat.stream("hi"))) == "hello"
        assert transport.calls == 2


class TestInStreamErrorIsNotRetried:
    def test_error_after_output_is_raised_not_retried(self, llmgw_settings):
        transport = _Scripted((200, FAIL_AFTER_OUTPUT), (200, SUCCESS))
        chat = _chat(llmgw_settings, transport)
        received: list[Any] = []
        with pytest.raises(UiPathServiceUnavailableError) as info:
            for chunk in chat.stream("hi"):
                received.append(chunk)
        assert info.value.in_stream is True
        assert transport.calls == 1
        assert "hello" in _text(received)

    def test_retries_disabled(self, llmgw_settings):
        transport = _Scripted((200, FAIL_BEFORE_OUTPUT), (200, SUCCESS))
        chat = _chat(llmgw_settings, transport, max_retries=0)
        with pytest.raises(UiPathServiceUnavailableError):
            list(chat.stream("hi"))
        assert transport.calls == 1

    def test_http_503_is_retried_by_transport_only(self, llmgw_settings):
        # A real HTTP 503 belongs to the transport retryer (replaced by the mock
        # transport here); the stream retryer must not retry it again and
        # multiply the transport's attempts.
        transport = _Scripted((503, b""))
        chat = _chat(llmgw_settings, transport, max_retries=3)
        with pytest.raises(UiPathServiceUnavailableError) as info:
            list(chat.stream("hi"))
        assert info.value.in_stream is False
        assert transport.calls == 1
