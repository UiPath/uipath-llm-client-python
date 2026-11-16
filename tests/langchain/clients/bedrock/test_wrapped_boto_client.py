"""Unit tests for ``WrappedBotoClient`` HTTP error surfacing.

The shim talks to the LLM Gateway over httpx instead of AWS. It must call
``raise_for_status()`` so gateway HTTP errors (e.g. 403 License-not-available)
propagate as exceptions, rather than being parsed as a normal result and then
mis-reported downstream (langchain_aws raises a misleading "No 'output' key"
``ValueError`` when the response lacks the expected fields).
"""

import json

import httpx
import pytest
from uipath_langchain_client.clients.bedrock.chat_models import UiPathChatBedrockConverse
from uipath_langchain_client.clients.bedrock.utils import WrappedBotoClient

from tests.aws_event_stream import (
    error_frame,
    event_frame,
    event_stream_response,
    exception_frame,
    raw_frame,
)
from tests.langchain.transports import route_transport
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathBadGatewayError,
    UiPathBadRequestError,
    UiPathInternalServerError,
    UiPathRateLimitError,
    UiPathServiceUnavailableError,
)

_ERROR_BODY = {
    "title": "License not available",
    "status": 403,
    "detail": "License not available for LLM usage.",
}


def _wrapped(handler: object) -> WrappedBotoClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return WrappedBotoClient(
        httpx_client=httpx.Client(transport=transport, base_url="http://gateway")
    )


def test_converse_raises_on_http_error() -> None:
    client = _wrapped(lambda request: httpx.Response(403, json=_ERROR_BODY))
    with pytest.raises(httpx.HTTPStatusError):
        client.converse(messages=[{"role": "user", "content": [{"text": "hi"}]}])


def test_converse_returns_body_on_success() -> None:
    payload = {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}}}
    client = _wrapped(lambda request: httpx.Response(200, json=payload))
    assert client.converse(messages=[]) == payload


def test_invoke_model_raises_on_http_error() -> None:
    client = _wrapped(lambda request: httpx.Response(403, json=_ERROR_BODY))
    with pytest.raises(httpx.HTTPStatusError):
        client.invoke_model(body=json.dumps({"prompt": "hi"}))


def test_converse_stream_raises_on_http_error() -> None:
    # The generator defers work until iterated, so the error surfaces on consume.
    client = _wrapped(lambda request: httpx.Response(403, json=_ERROR_BODY))
    stream = client.converse_stream(messages=[])["stream"]
    with pytest.raises(httpx.HTTPStatusError):
        list(stream)


def test_invoke_model_with_response_stream_raises_on_http_error() -> None:
    client = _wrapped(lambda request: httpx.Response(403, json=_ERROR_BODY))
    stream = client.invoke_model_with_response_stream(body=json.dumps({"prompt": "hi"}))["body"]
    with pytest.raises(httpx.HTTPStatusError):
        list(stream)


def test_converse_stream_hides_gateway_cost_event_from_langchain_aws() -> None:
    # langchain-aws raises on unknown events; the httpx client captures the cost itself.
    body = event_stream_response(
        event_frame("messageStart", {"role": "assistant"}),
        event_frame("messageStop", {"stopReason": "end_turn"}),
        event_frame("costMetadata", {"associated_dollar_cost": 0.002145}),
    )
    client = _wrapped(lambda request: body)
    events = list(client.converse_stream(messages=[])["stream"])
    assert events == [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


# Bedrock answers 200 and reports a failure part-way through the stream as an
# event-stream exception frame (``:message-type: exception``), which carries no
# ``:event-type``. It must surface as the provider error, not KeyError(':event-type').

_IN_STREAM_MESSAGE = (
    "The system encountered an unexpected error during processing. Try your request again."
)


@pytest.mark.parametrize(
    ("exception_type", "expected_class", "status"),
    [
        ("internalServerException", UiPathInternalServerError, 500),
        ("modelStreamErrorException", UiPathBadGatewayError, 502),
        ("serviceUnavailableException", UiPathServiceUnavailableError, 503),
        ("throttlingException", UiPathRateLimitError, 429),
        ("validationException", UiPathBadRequestError, 400),
        ("someFutureException", UiPathInternalServerError, 500),
    ],
)
def test_converse_stream_raises_provider_error_on_exception_frame(
    exception_type: str, expected_class: type[UiPathAPIError], status: int
) -> None:
    body = event_stream_response(
        event_frame("messageStart", {"role": "assistant"}),
        event_frame("contentBlockDelta", {"delta": {"text": "Hel"}, "contentBlockIndex": 0}),
        exception_frame(exception_type, {"message": _IN_STREAM_MESSAGE}),
    )
    client = _wrapped(lambda request: body)
    stream = client.converse_stream(messages=[])["stream"]
    received: list[dict] = []
    with pytest.raises(expected_class) as exc_info:
        for event in stream:
            received.append(event)
    # Events before the failure are still delivered.
    assert received == [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
    ]
    error = exc_info.value
    assert type(error) is expected_class
    assert error.status_code == status
    assert error.response.status_code == status
    assert error.error_code == exception_type
    assert error.body == {"type": exception_type, "message": _IN_STREAM_MESSAGE}
    assert _IN_STREAM_MESSAGE in error.response.text
    assert error.detail is not None and exception_type in error.detail
    assert str(error.request.url) == "http://gateway/"


def test_invoke_model_with_response_stream_raises_provider_error_on_exception_frame() -> None:
    body = event_stream_response(
        exception_frame("internalServerException", {"message": _IN_STREAM_MESSAGE}),
    )
    client = _wrapped(lambda request: body)
    stream = client.invoke_model_with_response_stream(body=json.dumps({"prompt": "hi"}))["body"]
    with pytest.raises(UiPathInternalServerError):
        list(stream)


def test_converse_stream_raises_provider_error_on_error_frame() -> None:
    body = event_stream_response(error_frame("ThrottlingException", "Too many tokens"))
    client = _wrapped(lambda request: body)
    with pytest.raises(UiPathRateLimitError) as exc_info:
        list(client.converse_stream(messages=[])["stream"])
    error = exc_info.value
    assert error.error_code == "ThrottlingException"
    assert error.body == {"type": "ThrottlingException", "message": "Too many tokens"}


def test_converse_stream_exception_frame_with_non_json_payload() -> None:
    frame = raw_frame(
        {":message-type": "exception", ":exception-type": "internalServerException"},
        b"upstream exploded",
    )
    client = _wrapped(lambda request: event_stream_response(frame))
    with pytest.raises(UiPathInternalServerError) as exc_info:
        list(client.converse_stream(messages=[])["stream"])
    assert exc_info.value.body == {
        "type": "internalServerException",
        "message": "upstream exploded",
    }


def test_converse_chat_stream_raises_provider_error_on_exception_frame(
    llmgw_settings: LLMGatewaySettings,
) -> None:
    """Through langchain-aws and the chat model's error wrapping, as the agent runtime calls it."""
    chat = UiPathChatBedrockConverse(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        settings=llmgw_settings,
        model_details={},
    )
    frames = [
        event_frame("messageStart", {"role": "assistant"}),
        event_frame("contentBlockDelta", {"delta": {"text": "Hel"}, "contentBlockIndex": 0}),
        exception_frame("internalServerException", {"message": _IN_STREAM_MESSAGE}),
    ]
    transport = httpx.MockTransport(lambda request: event_stream_response(*frames))
    route_transport(chat, transport, transport)
    with pytest.raises(UiPathInternalServerError) as exc_info:
        list(chat.stream("Hello"))
    assert exc_info.value.error_code == "internalServerException"
    assert _IN_STREAM_MESSAGE in exc_info.value.response.text
