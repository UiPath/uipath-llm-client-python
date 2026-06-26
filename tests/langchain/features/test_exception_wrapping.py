"""Tests that UiPathBaseChatModel converts provider SDK exceptions to UiPath errors.

A provider error (e.g. ``openai.BadRequestError``) raised by a passthrough client
is converted into the matching UiPath semantic subclass
(``UiPathBadRequestError``, ``UiPathRateLimitError``, …) — across sync/async
generate and streaming, for every provider. The result is a *pure* UiPath
exception (no vendor lineage); the original provider error is preserved as
``__cause__``.
"""

import json
import os
from typing import Any, Callable
from unittest.mock import patch

import anthropic
import httpx
import openai
import pytest
from google.genai.errors import ClientError as GenAIClientError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathAuthenticationError,
    UiPathBadRequestError,
    UiPathError,
    UiPathInternalServerError,
    UiPathPermissionDeniedError,
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


def _resp(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        request=httpx.Request("POST", "https://example.com"),
        headers=headers or {},
        json={"error": {"message": "boom"}},
    )


# --- builders for each provider's *native* exception shape ------------------


def _openai_exc(exc_cls: type[openai.APIStatusError], status: int, headers=None):
    return lambda: exc_cls("boom", response=_resp(status, headers), body=None)


def _anthropic_exc(exc_cls: type[anthropic.APIStatusError], status: int):
    return lambda: exc_cls("boom", response=_resp(status), body=None)


def _google_exc(status: int, headers=None):
    """langchain-google raises ChatGoogleGenerativeAIError *from* a genai error
    that holds the httpx response."""

    def build():
        cause = GenAIClientError(status, {"error": {"code": status}}, _resp(status, headers))
        err = ChatGoogleGenerativeAIError(f"Error calling model: {status}")
        err.__cause__ = cause
        return err

    return build


def _bedrock_exc(status: int):
    """The Bedrock shim's patched raise_for_status already yields a pure
    UiPathAPIError, so the chat model receives a UiPathError directly."""
    return lambda: UiPathAPIError.from_response(_resp(status))


# (provider, builds the native exc, expected pure UiPath type, already_uipath)
PROVIDER_CASES: list[tuple[str, Callable[[], Exception], type[UiPathError], bool]] = [
    ("openai", _openai_exc(openai.BadRequestError, 400), UiPathBadRequestError, False),
    (
        "openai",
        _openai_exc(openai.RateLimitError, 429, {"retry-after": "5"}),
        UiPathRateLimitError,
        False,
    ),
    ("anthropic", _anthropic_exc(anthropic.BadRequestError, 400), UiPathBadRequestError, False),
    (
        "anthropic",
        _anthropic_exc(anthropic.InternalServerError, 500),
        UiPathInternalServerError,
        False,
    ),
    ("google", _google_exc(400), UiPathBadRequestError, False),
    ("google", _google_exc(429, {"retry-after": "9"}), UiPathRateLimitError, False),
    # litellm / fireworks surface openai-style errors
    ("fireworks", _openai_exc(openai.AuthenticationError, 401), UiPathAuthenticationError, False),
    # bedrock: the shim already raised a pure UiPath error
    ("bedrock", _bedrock_exc(403), UiPathPermissionDeniedError, True),
]

CASE_IDS = [f"{name}-{exp.__name__}" for name, _, exp, _ in PROVIDER_CASES]


class _RaisingChat(UiPathChat):
    """Passthrough-style chat whose core methods raise a configured provider error."""

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


def _make_chat(settings: LLMGatewaySettings, boom: Exception) -> _RaisingChat:
    return _RaisingChat(model="gpt-4o", settings=settings, model_details={}, boom=boom)


class TestProviderErrorConversion:
    @pytest.mark.parametrize("name,build,expected,already", PROVIDER_CASES, ids=CASE_IDS)
    def test_generate(self, llmgw_settings, name, build, expected, already):
        native = build()
        chat = _make_chat(llmgw_settings, native)
        with pytest.raises(expected) as info:
            chat.invoke("hi")
        exc = info.value
        assert type(exc) is expected
        assert isinstance(exc, UiPathError)
        if already:
            # bedrock: the shim's UiPathError passes through untouched
            assert exc is native
        else:
            # the result is a *pure* UiPath error, original kept as cause
            assert exc.__cause__ is native

    @pytest.mark.parametrize("name,build,expected,already", PROVIDER_CASES, ids=CASE_IDS)
    @pytest.mark.asyncio
    async def test_agenerate(self, llmgw_settings, name, build, expected, already):
        chat = _make_chat(llmgw_settings, build())
        with pytest.raises(expected):
            await chat.ainvoke("hi")

    @pytest.mark.parametrize("name,build,expected,already", PROVIDER_CASES, ids=CASE_IDS)
    def test_stream(self, llmgw_settings, name, build, expected, already):
        chat = _make_chat(llmgw_settings, build())
        with pytest.raises(expected):
            list(chat.stream("hi"))

    @pytest.mark.parametrize("name,build,expected,already", PROVIDER_CASES, ids=CASE_IDS)
    @pytest.mark.asyncio
    async def test_astream(self, llmgw_settings, name, build, expected, already):
        chat = _make_chat(llmgw_settings, build())
        with pytest.raises(expected):
            async for _ in chat.astream("hi"):
                pass

    def test_no_vendor_lineage(self, llmgw_settings):
        """Converted errors are not catchable as their original vendor type."""
        chat = _make_chat(
            llmgw_settings, openai.BadRequestError("bad", response=_resp(400), body=None)
        )
        with pytest.raises(UiPathBadRequestError) as info:
            chat.invoke("hi")
        assert not isinstance(info.value, openai.APIError)

    def test_rate_limit_retry_after_preserved(self, llmgw_settings):
        chat = _make_chat(
            llmgw_settings,
            openai.RateLimitError("slow", response=_resp(429, {"retry-after": "5"}), body=None),
        )
        with pytest.raises(UiPathRateLimitError) as info:
            chat.invoke("hi")
        assert info.value.retry_after == 5.0

    def test_client_side_validation_error_becomes_root(self, llmgw_settings):
        """A non-HTTP error (e.g. a pydantic/validation error) maps to the root."""
        chat = _make_chat(llmgw_settings, ValueError("max_tokens not permitted"))
        with pytest.raises(UiPathError) as info:
            chat.invoke("hi")
        assert type(info.value) is UiPathError
        assert not isinstance(info.value, UiPathAPIError)
        assert isinstance(info.value.__cause__, ValueError)


# ============================================================================
# End-to-end: the genuine openai SDK raises BadRequestError on a real 400, and
# UiPathBaseChatModel converts it into a pure UiPathBadRequestError.
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


class TestRealOpenAISDKConversion:
    def test_sync_invoke_is_pure_uipath(self, llmgw_settings):
        chat = _make_openai_chat_returning_400(llmgw_settings)
        with pytest.raises(UiPathBadRequestError) as info:
            chat.invoke("hi")
        exc = info.value
        assert isinstance(exc, UiPathAPIError)
        assert isinstance(exc, UiPathError)
        assert not isinstance(exc, UiPathRateLimitError)
        assert not isinstance(exc, openai.APIError)  # vendor lineage dropped
        assert exc.status_code == 400
        # the genuine openai error is preserved as the cause
        assert isinstance(exc.__cause__, openai.BadRequestError)

    @pytest.mark.asyncio
    async def test_async_invoke_is_pure_uipath(self, llmgw_settings):
        chat = _make_openai_chat_returning_400(llmgw_settings)
        with pytest.raises(UiPathBadRequestError) as info:
            await chat.ainvoke("hi")
        assert not isinstance(info.value, openai.APIError)
        assert info.value.status_code == 400
