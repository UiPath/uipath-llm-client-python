"""Tests that UiPathBaseChatModel re-tags provider SDK exceptions as UiPathError.

A vendor error (e.g. ``openai.BadRequestError``) raised by a passthrough client
is re-tagged in place so callers can catch it as the original vendor type, as
the matching UiPath semantic subclass (``UiPathBadRequestError``), or as the
``UiPathError`` root -- across sync/async generate and streaming.
"""

import json
import os
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathBadRequestError,
    UiPathError,
    UiPathRateLimitError,
)

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}


@pytest.fixture(autouse=True)
def clear_singletons():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


@pytest.fixture
def llmgw_settings():
    with patch.dict(os.environ, LLMGW_ENV, clear=True):
        return LLMGatewaySettings()


def _bad_request() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://example.com")
    resp = httpx.Response(400, request=req, json={"error": {"message": "bad"}})
    return openai.BadRequestError("bad", response=resp, body={"error": {"message": "bad"}})


class _RaisingChat(UiPathChat):
    """Passthrough-style chat whose core methods raise a configured vendor error."""

    boom: Any = None

    def _uipath_generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.boom

    async def _uipath_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.boom

    def _uipath_stream(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.boom
        yield  # pragma: no cover - marks this a generator

    async def _uipath_astream(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.boom
        yield  # pragma: no cover - marks this an async generator


def _make_chat(settings: LLMGatewaySettings) -> _RaisingChat:
    return _RaisingChat(model="gpt-4o", settings=settings, model_details={}, boom=_bad_request())


CATCH_TYPES = [openai.BadRequestError, UiPathBadRequestError, UiPathAPIError, UiPathError]


class TestProviderErrorWrapping:
    @pytest.mark.parametrize("exc_type", CATCH_TYPES)
    def test_generate(self, llmgw_settings, exc_type):
        chat = _make_chat(llmgw_settings)
        with pytest.raises(exc_type):
            chat.invoke("hi")

    @pytest.mark.parametrize("exc_type", CATCH_TYPES)
    @pytest.mark.asyncio
    async def test_agenerate(self, llmgw_settings, exc_type):
        chat = _make_chat(llmgw_settings)
        with pytest.raises(exc_type):
            await chat.ainvoke("hi")

    @pytest.mark.parametrize("exc_type", CATCH_TYPES)
    def test_stream(self, llmgw_settings, exc_type):
        chat = _make_chat(llmgw_settings)
        with pytest.raises(exc_type):
            list(chat.stream("hi"))

    @pytest.mark.parametrize("exc_type", CATCH_TYPES)
    @pytest.mark.asyncio
    async def test_astream(self, llmgw_settings, exc_type):
        chat = _make_chat(llmgw_settings)
        with pytest.raises(exc_type):
            async for _ in chat.astream("hi"):
                pass

    def test_presents_as_uipath_type_but_catchable_as_vendor(self, llmgw_settings):
        chat = _make_chat(llmgw_settings)
        with pytest.raises(UiPathError) as info:
            chat.invoke("hi")
        exc = info.value
        # The exception presents as the UiPath type...
        assert type(exc).__name__ == "UiPathBadRequestError"
        # ...while remaining catchable as the original vendor type.
        assert isinstance(exc, openai.BadRequestError)
        assert exc.status_code == 400


# ============================================================================
# End-to-end: the genuine openai SDK raises BadRequestError on a real 400, and
# UiPathBaseChatModel re-tags it so it is catchable as both types.
# ============================================================================

_OPENAI_400_BODY = {
    "error": {"message": "Invalid 'messages'", "type": "invalid_request_error", "code": None}
}


def _openai_400_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        400,
        request=request,
        content=json.dumps(_OPENAI_400_BODY).encode(),
        headers={"content-type": "application/json"},
    )


class _SyncMock400(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return _openai_400_response(request)


class _AsyncMock400(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return _openai_400_response(request)


def _make_openai_chat_returning_400(settings: LLMGatewaySettings) -> UiPathChatOpenAI:
    chat = UiPathChatOpenAI(model="gpt-4o-2024-11-20", settings=settings, model_details={})
    # The openai SDK uses chat.uipath_sync_client / uipath_async_client as its
    # http_client (same object), so rerouting their transport makes the real SDK
    # receive a 400 and raise a genuine openai.BadRequestError.
    chat.uipath_sync_client._transport = _SyncMock400()  # type: ignore[attr-defined]
    chat.uipath_sync_client._mounts = {}  # type: ignore[attr-defined]
    chat.uipath_async_client._transport = _AsyncMock400()  # type: ignore[attr-defined]
    chat.uipath_async_client._mounts = {}  # type: ignore[attr-defined]
    return chat


class TestRealOpenAISDKWrapping:
    def test_sync_invoke_catchable_both_ways(self, llmgw_settings):
        chat = _make_openai_chat_returning_400(llmgw_settings)
        with pytest.raises(openai.BadRequestError) as info:
            chat.invoke("hi")
        exc = info.value
        assert isinstance(exc, UiPathBadRequestError)
        assert isinstance(exc, UiPathAPIError)
        assert isinstance(exc, UiPathError)
        assert not isinstance(exc, UiPathRateLimitError)
        assert exc.status_code == 400

    def test_sync_invoke_catchable_as_uipath_semantic(self, llmgw_settings):
        chat = _make_openai_chat_returning_400(llmgw_settings)
        with pytest.raises(UiPathBadRequestError):
            chat.invoke("hi")

    @pytest.mark.asyncio
    async def test_async_invoke_catchable_both_ways(self, llmgw_settings):
        chat = _make_openai_chat_returning_400(llmgw_settings)
        with pytest.raises(UiPathBadRequestError) as info:
            await chat.ainvoke("hi")
        assert isinstance(info.value, openai.BadRequestError)
