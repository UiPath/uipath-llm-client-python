"""Tests for UiPath exception classes."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from httpx import Headers, Request, Response

from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathAuthenticationError,
    UiPathBadRequestError,
    UiPathGatewayTimeoutError,
    UiPathInternalServerError,
    UiPathNotFoundError,
    UiPathPermissionDeniedError,
    UiPathRateLimitError,
    UiPathServiceUnavailableError,
    UiPathTooManyRequestsError,
    patch_raise_for_status,
)


class TestExceptions:
    """Tests for UiPath exception classes."""

    def test_exception_hierarchy(self):
        """Test all exceptions inherit from UiPathAPIError."""
        from httpx import HTTPStatusError

        assert issubclass(UiPathAPIError, HTTPStatusError)
        assert issubclass(UiPathAuthenticationError, UiPathAPIError)
        assert issubclass(UiPathRateLimitError, UiPathAPIError)

    def test_exception_status_codes(self):
        """Test exception classes have correct status codes."""
        assert UiPathBadRequestError.status_code == 400
        assert UiPathAuthenticationError.status_code == 401
        assert UiPathPermissionDeniedError.status_code == 403
        assert UiPathNotFoundError.status_code == 404
        assert UiPathRateLimitError.status_code == 429
        assert UiPathInternalServerError.status_code == 500

    def test_exception_from_response(self):
        """Test UiPathAPIError.from_response creates correct exception type."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {}  # Required for UiPathRateLimitError._parse_retry_after

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.status_code == 429

    def test_exception_from_response_with_retry_after(self):
        """Test UiPathRateLimitError parses Retry-After header."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {"retry-after": "30"}

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after == 30.0

    def test_exception_from_response_with_x_retry_after(self):
        """Test UiPathRateLimitError parses x-retry-after header."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {"x-retry-after": "45"}

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after == 45.0

    def test_exception_retry_after_none_when_not_present(self):
        """Test UiPathRateLimitError.retry_after is None when header missing."""
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limited"}
        mock_response.request = MagicMock(spec=Request)
        mock_response.headers = {}

        exc = UiPathAPIError.from_response(mock_response)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after is None


class TestExceptionDetails:
    """Tests for exception __str__, __repr__, body parsing, and patch_raise_for_status."""

    def _make_response(
        self, status_code, reason_phrase="Error", body_json=None, body_text="", headers=None
    ):
        mock_resp = MagicMock(spec=Response)
        mock_resp.status_code = status_code
        mock_resp.reason_phrase = reason_phrase
        mock_resp.request = MagicMock(spec=Request)
        mock_resp.headers = headers or {}
        mock_resp.text = body_text
        if body_json is not None:
            mock_resp.json.return_value = body_json
        else:
            from json import JSONDecodeError

            mock_resp.json.side_effect = JSONDecodeError("", "", 0)
        return mock_resp

    def test_str_format(self):
        resp = self._make_response(400, "Bad Request", body_json={"error": "invalid"})
        exc = UiPathAPIError.from_response(resp)
        s = str(exc)
        assert "UiPathBadRequestError" in s
        assert "Bad Request" in s
        assert "400" in s

    def test_repr_format(self):
        resp = self._make_response(404, "Not Found", body_json={"error": "missing"})
        exc = UiPathAPIError.from_response(resp)
        r = repr(exc)
        assert "UiPathNotFoundError" in r
        assert "status_code=404" in r

    def test_from_response_json_body(self):
        resp = self._make_response(500, "Server Error", body_json={"detail": "crash"})
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathInternalServerError)
        assert exc.body == {"detail": "crash"}

    def test_from_response_text_body_fallback(self):
        resp = self._make_response(500, "Server Error", body_text="plain text error")
        exc = UiPathAPIError.from_response(resp)
        assert exc.body == "plain text error"

    def test_from_response_no_body(self):
        resp = self._make_response(503, "Unavailable")
        resp.json.side_effect = Exception("unexpected")
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathServiceUnavailableError)
        assert exc.body is None

    def test_from_response_uses_response_request_when_none(self):
        resp = self._make_response(400, "Bad Request", body_json={})
        exc = UiPathAPIError.from_response(resp, request=None)
        assert exc.request is resp.request

    def test_from_response_uses_explicit_request(self):
        resp = self._make_response(400, "Bad Request", body_json={})
        custom_request = MagicMock(spec=Request)
        exc = UiPathAPIError.from_response(resp, request=custom_request)
        assert exc.request is custom_request

    def test_from_response_unmapped_status_code(self):
        resp = self._make_response(418, "I'm a teapot", body_json={})
        exc = UiPathAPIError.from_response(resp)
        assert type(exc) is UiPathAPIError
        assert exc.status_code == 418

    def test_all_status_code_mappings(self):
        mappings = {
            400: UiPathBadRequestError,
            401: UiPathAuthenticationError,
            403: UiPathPermissionDeniedError,
            404: UiPathNotFoundError,
            429: UiPathRateLimitError,
            500: UiPathInternalServerError,
            503: UiPathServiceUnavailableError,
            504: UiPathGatewayTimeoutError,
            529: UiPathTooManyRequestsError,
        }
        for code, expected_cls in mappings.items():
            resp = self._make_response(code, "Error", body_json={})
            exc = UiPathAPIError.from_response(resp)
            assert isinstance(exc, expected_cls), f"Status {code} -> {type(exc).__name__}"

    def test_patch_raise_for_status_converts_exception(self):
        from httpx import HTTPStatusError

        resp = MagicMock(spec=Response)
        resp.status_code = 404
        resp.reason_phrase = "Not Found"
        resp.json.return_value = {}
        resp.request = MagicMock(spec=Request)
        resp.headers = {}
        original = MagicMock(
            side_effect=HTTPStatusError("err", request=resp.request, response=resp)
        )
        resp.raise_for_status = original

        patched = patch_raise_for_status(resp)
        with pytest.raises(UiPathNotFoundError):
            patched.raise_for_status()

    def test_patch_raise_for_status_passes_on_success(self):
        resp = MagicMock(spec=Response)
        resp.status_code = 200
        original = MagicMock(return_value=resp)
        resp.raise_for_status = original

        patched = patch_raise_for_status(resp)
        result = patched.raise_for_status()
        assert result is resp

    def test_retry_after_http_date_format(self):
        future = datetime.now(timezone.utc).replace(microsecond=0)
        future = future.replace(second=future.second + 30 if future.second < 30 else future.second)
        date_str = future.strftime("%a, %d %b %Y %H:%M:%S GMT")

        resp = self._make_response(
            429, "Rate Limit", body_json={}, headers={"retry-after": date_str}
        )
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathRateLimitError)
        # The retry_after should be a positive number (seconds until future date)
        assert exc.retry_after is not None
        assert exc.retry_after >= 0

    def test_retry_after_unparseable_returns_none(self):
        resp = self._make_response(
            429, "Rate Limit", body_json={}, headers={"retry-after": "not-a-date-or-number"}
        )
        exc = UiPathAPIError.from_response(resp)
        assert isinstance(exc, UiPathRateLimitError)
        assert exc.retry_after is None


class TestRateLimitRetryAfterParsing:
    """Tests for UiPathRateLimitError._parse_retry_after."""

    def _make_429_response(self, headers=None):
        mock_resp = MagicMock(spec=Response)
        mock_resp.status_code = 429
        mock_resp.reason_phrase = "Too Many Requests"
        mock_resp.request = MagicMock(spec=Request)
        mock_resp.headers = Headers(headers or {})
        mock_resp.json.return_value = {}
        return mock_resp

    def test_parses_integer_seconds_from_retry_after(self):
        """retry-after header with integer seconds is parsed correctly."""
        resp = self._make_429_response(headers={"retry-after": "120"})
        result = UiPathRateLimitError._parse_retry_after(resp)
        assert result == 120.0

    def test_parses_float_seconds_from_retry_after(self):
        """retry-after header with float seconds is parsed correctly."""
        resp = self._make_429_response(headers={"retry-after": "2.5"})
        result = UiPathRateLimitError._parse_retry_after(resp)
        assert result == 2.5

    def test_parses_x_retry_after_as_fallback(self):
        """x-retry-after header is used when retry-after is absent."""
        resp = self._make_429_response(headers={"x-retry-after": "30"})
        result = UiPathRateLimitError._parse_retry_after(resp)
        assert result == 30.0

    def test_parses_http_date_format(self):
        """retry-after with HTTP-date format returns positive delay."""
        future = datetime.now(timezone.utc) + timedelta(seconds=60)
        date_str = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        resp = self._make_429_response(headers={"retry-after": date_str})
        result = UiPathRateLimitError._parse_retry_after(resp)
        assert result is not None
        assert result > 0

    def test_returns_none_when_no_header_present(self):
        """Returns None when neither retry-after nor x-retry-after is set."""
        resp = self._make_429_response(headers={})
        result = UiPathRateLimitError._parse_retry_after(resp)
        assert result is None

    def test_returns_none_for_unparseable_value(self):
        """Returns None for values that are neither numbers nor valid dates."""
        resp = self._make_429_response(headers={"retry-after": "not-valid"})
        result = UiPathRateLimitError._parse_retry_after(resp)
        assert result is None

    def test_retry_after_prefers_standard_header(self):
        """retry-after takes precedence over x-retry-after."""
        resp = self._make_429_response(headers={"retry-after": "10", "x-retry-after": "99"})
        result = UiPathRateLimitError._parse_retry_after(resp)
        assert result == 10.0

    def test_retry_after_property_on_exception(self):
        """retry_after property is set from the response header."""
        resp = self._make_429_response(headers={"retry-after": "42"})
        exc = UiPathRateLimitError(
            "rate limited",
            request=resp.request,
            response=resp,
        )
        assert exc.retry_after == 42.0


class TestPatchRaiseForStatus:
    """Tests for patch_raise_for_status utility."""

    def test_patched_response_raises_uipath_error_on_error_status(self):
        """Patched response raises UiPathAPIError subclass on HTTP error."""
        from httpx import HTTPStatusError

        mock_resp = MagicMock(spec=Response)
        mock_resp.status_code = 404
        mock_resp.reason_phrase = "Not Found"
        mock_resp.json.return_value = {"error": "not found"}
        mock_resp.request = MagicMock(spec=Request)
        mock_resp.headers = {}
        original = MagicMock(
            side_effect=HTTPStatusError("err", request=mock_resp.request, response=mock_resp)
        )
        mock_resp.raise_for_status = original

        patched = patch_raise_for_status(mock_resp)
        with pytest.raises(UiPathAPIError) as exc_info:
            patched.raise_for_status()
        assert isinstance(exc_info.value, UiPathNotFoundError)

    def test_patched_response_returns_response_on_success(self):
        """Patched response returns the response object on 2xx status."""
        mock_resp = MagicMock(spec=Response)
        mock_resp.status_code = 200
        original = MagicMock(return_value=mock_resp)
        mock_resp.raise_for_status = original

        patched = patch_raise_for_status(mock_resp)
        result = patched.raise_for_status()
        assert result is mock_resp

    def test_patched_replaces_original_method(self):
        """The raise_for_status method is replaced, not wrapped additively."""
        mock_resp = MagicMock(spec=Response)
        mock_resp.status_code = 200
        original = MagicMock(return_value=mock_resp)
        mock_resp.raise_for_status = original

        patched = patch_raise_for_status(mock_resp)
        assert patched.raise_for_status is not original
