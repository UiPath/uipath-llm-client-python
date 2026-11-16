"""In-stream provider errors (HTTP 200, then an error event) — PC-5454.

``openai._streaming`` raises ``openai.APIError(message, request, body=error)``
with no response when an SSE event carries an ``error`` object. An overload /
service-unavailable event must become a typed, retryable
``UiPathServiceUnavailableError`` marked ``in_stream``; anything else keeps the
status-less ``UiPathError`` root.
"""

import openai
import pytest
from httpx import Request, Response
from tenacity import Retrying

from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathBadRequestError,
    UiPathError,
    UiPathServiceUnavailableError,
    as_uipath_error,
)
from uipath.llm_client.utils.retry import RetryConfig, build_in_stream_retryer

_REQUEST = Request("POST", "https://example.com/v1/responses")
_NO_WAIT: RetryConfig = {"initial_delay": 0.0, "max_delay": 0.0, "jitter": 0.0}


def _stream_error(message: str, **body: object) -> openai.APIError:
    return openai.APIError(message, _REQUEST, body={"message": message, **body})


class TestInStreamUnavailableClassification:
    @pytest.mark.parametrize(
        "error",
        [
            _stream_error("Service Unavailable"),
            _stream_error("Our servers are currently overloaded"),
            _stream_error(
                "The server is overloaded or not ready yet.", type="service_unavailable_error"
            ),
            _stream_error("Try again later", code="server_is_overloaded"),
            _stream_error("Overloaded", type="overloaded_error"),
        ],
    )
    def test_maps_to_in_stream_503(self, error: openai.APIError):
        converted = as_uipath_error(error)
        assert type(converted) is UiPathServiceUnavailableError
        assert converted.in_stream is True
        assert converted.status_code == 503
        assert converted.request is _REQUEST
        assert converted.detail == error.message
        assert converted.body == error.body
        assert converted.response.json() == {"error": error.body}
        assert error.message in converted.response.text
        assert not isinstance(converted, openai.APIError)

    def test_error_code_is_provider_code_or_type(self):
        assert (
            as_uipath_error(_stream_error("x", type="service_unavailable_error")).error_code
            == "service_unavailable_error"
        )
        assert (
            as_uipath_error(_stream_error("x", code="server_is_overloaded")).error_code
            == "server_is_overloaded"
        )

    def test_str_omits_body(self):
        converted = as_uipath_error(_stream_error("Service Unavailable", secret="pii"))
        assert "pii" not in str(converted)
        assert "503" in str(converted)

    @pytest.mark.parametrize(
        "error",
        [
            _stream_error("bad input", type="invalid_request_error"),
            _stream_error("An error occurred during streaming"),
            ValueError("service unavailable"),  # not a provider SDK error
        ],
    )
    def test_other_errors_stay_untyped(self, error: Exception):
        converted = as_uipath_error(error)
        assert type(converted) is UiPathError

    def test_real_http_status_wins(self):
        response = Response(400, request=_REQUEST)
        error = openai.BadRequestError("Service Unavailable", response=response, body=None)
        converted = as_uipath_error(error)
        assert type(converted) is UiPathBadRequestError
        assert converted.in_stream is False

    def test_regular_api_errors_are_not_in_stream(self):
        converted = UiPathAPIError.from_response(Response(503, request=_REQUEST))
        assert converted.in_stream is False


class TestInStreamRetryer:
    def test_disabled_when_no_retries(self):
        assert build_in_stream_retryer(max_retries=0, retry_config=None, logger=None) is None

    def test_retries_in_stream_errors_only(self):
        retryer = build_in_stream_retryer(max_retries=3, retry_config=_NO_WAIT, logger=None)
        assert isinstance(retryer, Retrying)

        calls = 0

        def in_stream_then_ok() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise as_uipath_error(_stream_error("Service Unavailable"))
            return "ok"

        assert retryer(in_stream_then_ok) == "ok"
        assert calls == 3

        calls = 0

        def http_503() -> str:
            nonlocal calls
            calls += 1
            raise UiPathAPIError.from_response(Response(503, request=_REQUEST))

        with pytest.raises(UiPathServiceUnavailableError):
            retryer(http_503)
        assert calls == 1

    def test_respects_retry_on_exceptions(self):
        # A caller that narrowed the retry types keeps that choice for in-stream errors.
        retryer = build_in_stream_retryer(
            max_retries=3,
            retry_config=RetryConfig(
                initial_delay=0.0,
                max_delay=0.0,
                jitter=0.0,
                retry_on_exceptions=(UiPathBadRequestError,),
            ),
            logger=None,
        )
        assert isinstance(retryer, Retrying)
        calls = 0

        def in_stream() -> None:
            nonlocal calls
            calls += 1
            raise as_uipath_error(_stream_error("Service Unavailable"))

        with pytest.raises(UiPathServiceUnavailableError):
            retryer(in_stream)
        assert calls == 1
