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

from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathPermissionDeniedError,
    patch_raise_for_status,
)

_ERROR_BODY = {
    "title": "License not available",
    "status": 403,
    "detail": "License not available for LLM usage.",
}

_HTML_403_BODY = (
    "<!DOCTYPE html><html><head><title>403 Forbidden</title></head>"
    "<body>403 Forbidden</body></html>"
)


def _wrapped(handler: object) -> WrappedBotoClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return WrappedBotoClient(
        httpx_client=httpx.Client(transport=transport, base_url="http://gateway")
    )


def _wrapped_patched(handler: object) -> WrappedBotoClient:
    """Mirror production: responses carry the patched ``raise_for_status``.

    ``UiPathHttpxClient`` runs ``patch_raise_for_status`` on every response so a
    non-2xx body surfaces as a typed ``UiPathAPIError`` (status + body excerpt)
    instead of a bare ``httpx.HTTPStatusError`` or an opaque ``JSONDecodeError``.
    """
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return WrappedBotoClient(
        httpx_client=httpx.Client(
            transport=transport,
            base_url="http://gateway",
            event_hooks={"response": [lambda response: patch_raise_for_status(response)]},
        )
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


def test_converse_surfaces_legible_error_for_non_json_body() -> None:
    """Regression for PC-4775: a 403 HTML body must not become a JSONDecodeError.

    Through the patched client the gateway error surfaces as a typed
    ``UiPathPermissionDeniedError`` carrying the status code and the HTML body
    excerpt, instead of the opaque ``json.decoder.JSONDecodeError`` that crashed
    the job before the ``raise_for_status`` guard.
    """
    client = _wrapped_patched(
        lambda request: httpx.Response(
            403, text=_HTML_403_BODY, headers={"content-type": "text/html"}
        )
    )
    with pytest.raises(UiPathPermissionDeniedError) as exc_info:
        client.converse(messages=[{"role": "user", "content": [{"text": "hi"}]}])
    error = exc_info.value
    assert error.status_code == 403
    assert error.body == _HTML_403_BODY
    assert "403 Forbidden" in str(error)


def test_converse_stream_surfaces_legible_error_for_non_json_body() -> None:
    """Regression for PC-4775 on the streaming path.

    The non-event 403 HTML body is read before the EventStreamBuffer touches it,
    so the patched ``raise_for_status`` surfaces a typed error rather than a
    checksum mismatch or an opaque ``JSONDecodeError``.
    """
    client = _wrapped_patched(
        lambda request: httpx.Response(
            403, text=_HTML_403_BODY, headers={"content-type": "text/html"}
        )
    )
    stream = client.converse_stream(messages=[])["stream"]
    with pytest.raises(UiPathPermissionDeniedError) as exc_info:
        list(stream)
    error = exc_info.value
    assert error.status_code == 403
    assert error.body == _HTML_403_BODY
    assert "403 Forbidden" in str(error)


def test_converse_typed_error_includes_json_body_excerpt() -> None:
    """A JSON gateway error body is preserved on the typed error for diagnosis."""
    client = _wrapped_patched(lambda request: httpx.Response(403, json=_ERROR_BODY))
    with pytest.raises(UiPathAPIError) as exc_info:
        client.converse(messages=[])
    assert exc_info.value.status_code == 403
    assert exc_info.value.body == _ERROR_BODY
