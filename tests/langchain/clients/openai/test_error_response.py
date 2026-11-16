"""A chat completions body that is an error object raises a typed UiPathError.

Some OpenAI-compatible endpoints (seen on bring-your-own model endpoints) answer
HTTP 200 with ``{"error": ...}`` instead of a completion. langchain-openai raises
a bare ``ValueError`` for it, which used to reach callers as an uncategorised
``UiPathError``. The UiPath OpenAI chat models raise a ``UiPathError`` carrying
``UiPathLLMErrorCode.PROVIDER_ERROR_RESPONSE`` and the endpoint's message.
"""

import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage
from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
)

from tests.langchain.transports import route_transport
from uipath.llm_client.settings import LLMGatewaySettings, UiPathBaseSettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathError,
    UiPathLLMErrorCode,
)

_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "byo-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_ChatClass = type[UiPathChatOpenAI] | type[UiPathAzureChatOpenAI]
CHAT_CLASSES: list[_ChatClass] = [UiPathChatOpenAI, UiPathAzureChatOpenAI]

_LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}


@pytest.fixture
def client_settings() -> Iterator[UiPathBaseSettings]:
    """Self-contained gateway settings: every HTTP call goes to a mock transport."""
    SingletonMeta._instances.clear()
    with patch.dict(os.environ, _LLMGW_ENV, clear=True):
        yield LLMGatewaySettings()
    SingletonMeta._instances.clear()


def _chat(chat_cls: _ChatClass, settings: UiPathBaseSettings, body: dict[str, Any]) -> Any:
    chat = chat_cls(model="byo-model", settings=settings, model_details={})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    route_transport(chat, httpx.MockTransport(handler), httpx.MockTransport(async_handler))
    return chat


@pytest.mark.parametrize("chat_cls", CHAT_CLASSES)
@pytest.mark.parametrize(
    "error,expected_detail",
    [
        # The shape seen in production: a bare string.
        ("apiUnexpected", "apiUnexpected"),
        # The OpenAI error object shape: its message is the detail.
        ({"message": "upstream model failed", "code": "server_error"}, "upstream model failed"),
        # An error object without a message keeps all of it.
        ({"code": "server_error"}, "{'code': 'server_error'}"),
    ],
)
def test_200_error_body_raises_provider_error_response(
    client_settings: UiPathBaseSettings, chat_cls: _ChatClass, error: Any, expected_detail: str
) -> None:
    chat = _chat(chat_cls, client_settings, {"error": error})

    with pytest.raises(UiPathError) as info:
        chat.invoke("hi")

    exc = info.value
    assert type(exc) is UiPathError
    assert not isinstance(exc, UiPathAPIError)
    assert exc.error_code == UiPathLLMErrorCode.PROVIDER_ERROR_RESPONSE
    assert exc.detail == expected_detail
    assert str(exc) == expected_detail


@pytest.mark.parametrize("chat_cls", CHAT_CLASSES)
@pytest.mark.asyncio
async def test_200_error_body_raises_provider_error_response_async(
    client_settings: UiPathBaseSettings, chat_cls: _ChatClass
) -> None:
    chat = _chat(chat_cls, client_settings, {"error": "apiUnexpected"})

    with pytest.raises(UiPathError) as info:
        await chat.ainvoke("hi")

    assert info.value.error_code == UiPathLLMErrorCode.PROVIDER_ERROR_RESPONSE
    assert info.value.detail == "apiUnexpected"


@pytest.mark.parametrize("chat_cls", CHAT_CLASSES)
@pytest.mark.parametrize("error", [None, "", {}])
def test_empty_error_field_is_not_an_error(
    client_settings: UiPathBaseSettings, chat_cls: _ChatClass, error: Any
) -> None:
    """langchain-openai ignores a falsy ``error`` field; so does the mixin."""
    chat = _chat(chat_cls, client_settings, {**_COMPLETION, "error": error})

    message = chat.invoke("hi")

    assert isinstance(message, AIMessage)
    assert message.content == "ok"


@pytest.mark.parametrize("chat_cls", CHAT_CLASSES)
def test_completion_is_unchanged(client_settings: UiPathBaseSettings, chat_cls: _ChatClass) -> None:
    chat = _chat(chat_cls, client_settings, _COMPLETION)

    message = chat.invoke("hi")

    assert isinstance(message, AIMessage)
    assert message.content == "ok"


def test_dict_response_with_error_raises(client_settings: UiPathBaseSettings) -> None:
    """``_create_chat_result`` also accepts a dict response (e.g. from callers)."""
    chat = UiPathChatOpenAI(model="byo-model", settings=client_settings, model_details={})

    with pytest.raises(UiPathError) as info:
        chat._create_chat_result({"error": {"message": "apiUnexpected"}})

    assert info.value.error_code == UiPathLLMErrorCode.PROVIDER_ERROR_RESPONSE
    assert info.value.detail == "apiUnexpected"
