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
from uipath_langchain_client.clients.bedrock.utils import WrappedBotoClient

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
