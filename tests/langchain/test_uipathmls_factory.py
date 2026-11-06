"""Unit tests for ``get_chat_model`` on UiPath-hosted (``UiPathMls``) models such as path-mini.

Discovery lists path-mini with vendor ``UiPathMls`` and ``apiFlavor: null``. The gateway
serves it only on the raw route ``.../raw/vendor/uipathmls/model/{model}/completions`` in
OpenAI's chat completions format, and rejects streaming requests.
"""

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage
from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI
from uipath_langchain_client.factory import get_chat_model

from tests.langchain.transports import route_transport
from uipath.llm_client.settings import ApiFlavor, RoutingMode, UiPathBaseSettings

_PATH_MINI_INFO: dict[str, Any] = {
    "modelName": "path-mini",
    "vendor": "UiPathMls",
    "modelFamily": "Proprietary",
    "apiFlavor": None,
    "apiFlavors": ["OpenAiChatCompletions"],
    "modelSubscriptionType": "UiPathOwned",
    "modelDetails": {"maxOutputTokens": 0, "contextWindowTokens": 0},
}

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "path-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def path_mini_settings(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> UiPathBaseSettings:
    monkeypatch.setattr(
        client_settings, "get_model_info", lambda model_name, **kwargs: dict(_PATH_MINI_INFO)
    )
    return client_settings


def _route_and_capture(chat: Any) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_COMPLETION)

    route_transport(chat, httpx.MockTransport(handler))
    return requests


def test_factory_builds_openai_chat_on_the_uipathmls_raw_route(
    path_mini_settings: UiPathBaseSettings,
) -> None:
    chat = get_chat_model("path-mini", client_settings=path_mini_settings)

    assert isinstance(chat, UiPathChatOpenAI)
    assert chat.api_config.vendor_type == "uipathmls"
    assert chat.api_config.routing_mode == RoutingMode.PASSTHROUGH
    assert chat.api_config.api_flavor == ApiFlavor.CHAT_COMPLETIONS
    assert chat.use_responses_api is False
    assert chat.streaming is False
    assert chat.disable_streaming is True


def test_request_goes_to_uipathmls_route_with_thinking_off(
    path_mini_settings: UiPathBaseSettings,
) -> None:
    chat = get_chat_model("path-mini", client_settings=path_mini_settings, max_tokens=256)
    requests = _route_and_capture(chat)

    result = chat.invoke([HumanMessage(content="hi")])

    assert result.content == "ok"
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path.endswith("/raw/vendor/uipathmls/model/path-mini/completions")
    assert request.headers["X-UiPath-LlmGateway-ApiFlavor"] == ApiFlavor.CHAT_COMPLETIONS.value
    body = json.loads(request.content)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body
    assert not body.get("stream")


def test_streaming_is_forced_off_even_when_the_caller_enables_it(
    path_mini_settings: UiPathBaseSettings,
) -> None:
    chat = get_chat_model(
        "path-mini",
        client_settings=path_mini_settings,
        streaming=True,
        disable_streaming=False,
    )
    requests = _route_and_capture(chat)

    chunks = list(chat.stream([HumanMessage(content="hi")]))

    assert isinstance(chat, UiPathChatOpenAI)
    assert chat.streaming is False
    assert chat.disable_streaming is True
    assert "".join(str(c.content) for c in chunks) == "ok"
    assert len(requests) == 1
    assert not json.loads(requests[0].content).get("stream")


def test_caller_extra_body_is_kept_and_can_override_thinking(
    path_mini_settings: UiPathBaseSettings,
) -> None:
    chat = get_chat_model(
        "path-mini",
        client_settings=path_mini_settings,
        extra_body={
            "request_type": "agents",
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    requests = _route_and_capture(chat)

    chat.invoke([HumanMessage(content="hi")])

    body = json.loads(requests[0].content)
    assert body["request_type"] == "agents"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
