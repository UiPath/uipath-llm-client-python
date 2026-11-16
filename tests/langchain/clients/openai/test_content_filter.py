"""A content-filtered Azure OpenAI choice comes back as a message, not a ValueError.

Azure OpenAI answers 200 with ``finish_reason: "content_filter"`` when the
deployment's content filter stops the completion. langchain-openai's
``AzureChatOpenAI`` raises a bare ``ValueError`` for it, which carries no HTTP
response and so reached callers as an uncategorised ``UiPathError`` (PC-5436).
``UiPathAzureChatOpenAI`` returns the message with the finish reason in
``response_metadata``, as ``ChatOpenAI`` does, so callers can check it.
"""

from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from openai.types.chat import ChatCompletion
from uipath_langchain_client.clients.openai.chat_models import UiPathAzureChatOpenAI

from tests.langchain.transports import route_transport
from uipath.llm_client.settings import UiPathBaseSettings

_FILTER_RESULTS = {
    "hate": {"filtered": True, "severity": "high"},
    "self_harm": {"filtered": False, "severity": "safe"},
}


def _choice(index: int, finish_reason: str, content: str | None) -> dict[str, Any]:
    return {
        "index": index,
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content},
        "content_filter_results": _FILTER_RESULTS if finish_reason == "content_filter" else {},
    }


def _completion(*choices: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.2-2025-12-11",
        "choices": list(choices),
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


def _client() -> UiPathAzureChatOpenAI:
    return UiPathAzureChatOpenAI.model_construct(
        model_name="gpt-5.2-2025-12-11",
        client_settings=object(),
        output_version=None,
        use_responses_api=False,
    )


@pytest.mark.parametrize("typed", [True, False], ids=["openai-model", "dict"])
def test_content_filtered_choice_is_returned_with_its_finish_reason(typed: bool) -> None:
    raw = _completion(_choice(0, "content_filter", None))
    response: Any = ChatCompletion.model_validate(raw) if typed else raw

    result = _client()._create_chat_result(response)

    generation = result.generations[0]
    assert isinstance(generation.message, AIMessage)
    assert generation.message.content == ""
    assert generation.message.response_metadata["finish_reason"] == "content_filter"
    assert generation.generation_info is not None
    assert generation.generation_info["finish_reason"] == "content_filter"
    # AzureChatOpenAI's own post-processing still runs.
    assert generation.generation_info["content_filter_results"] == _FILTER_RESULTS
    assert result.llm_output is not None
    assert result.llm_output["model_name"] == "gpt-5.2-2025-12-11"
    assert generation.message.usage_metadata is not None
    assert generation.message.usage_metadata["total_tokens"] == 13


def test_typed_response_keeps_refusal() -> None:
    raw = _completion(_choice(0, "content_filter", None))
    raw["choices"][0]["message"]["refusal"] = "I can't help with that."

    result = _client()._create_chat_result(ChatCompletion.model_validate(raw))

    message = result.generations[0].message
    assert message.additional_kwargs["refusal"] == "I can't help with that."


def test_only_the_filtered_choice_carries_content_filter() -> None:
    raw = _completion(_choice(0, "stop", "hello"), _choice(1, "content_filter", None))

    result = _client()._create_chat_result(ChatCompletion.model_validate(raw))

    first, second = result.generations
    assert first.message.content == "hello"
    assert first.generation_info is not None
    assert first.generation_info["finish_reason"] == "stop"
    assert second.message.response_metadata["finish_reason"] == "content_filter"
    assert second.generation_info is not None
    assert second.generation_info["finish_reason"] == "content_filter"


def test_unfiltered_response_is_unchanged() -> None:
    raw = _completion(_choice(0, "stop", "hello"))

    result = _client()._create_chat_result(ChatCompletion.model_validate(raw))

    generation = result.generations[0]
    assert generation.message.content == "hello"
    assert generation.generation_info is not None
    assert generation.generation_info["finish_reason"] == "stop"


def _filtered_chat(client_settings: UiPathBaseSettings) -> UiPathAzureChatOpenAI:
    chat = UiPathAzureChatOpenAI(model="gpt-5.2-2025-12-11", settings=client_settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(_choice(0, "content_filter", None)))

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return handler(request)

    route_transport(
        chat,
        httpx.MockTransport(handler),
        httpx.MockTransport(async_handler),
    )
    return chat


def test_invoke_returns_the_filtered_message(client_settings: UiPathBaseSettings) -> None:
    message = _filtered_chat(client_settings).invoke([HumanMessage("hi")])

    assert isinstance(message, AIMessage)
    assert message.response_metadata["finish_reason"] == "content_filter"


async def test_ainvoke_returns_the_filtered_message(client_settings: UiPathBaseSettings) -> None:
    message = await _filtered_chat(client_settings).ainvoke([HumanMessage("hi")])

    assert isinstance(message, AIMessage)
    assert message.response_metadata["finish_reason"] == "content_filter"
