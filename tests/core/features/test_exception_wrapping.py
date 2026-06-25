"""Tests for the UiPathError root and as_uipath_error / wrap_provider_errors.

These cover the unified-taxonomy behaviour: a provider error is converted into
the matching UiPath exception (status mapped from any ``httpx.Response`` in the
cause chain) and the original provider error is preserved as ``__cause__``.
"""

import pytest
from httpx import HTTPStatusError, Request, Response

from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathAuthenticationError,
    UiPathBadRequestError,
    UiPathError,
    UiPathRateLimitError,
    as_uipath_error,
    wrap_provider_errors,
)


class _FakeVendorError(Exception):
    """Mimics the shape of openai/anthropic errors (mutable, httpx response)."""

    def __init__(
        self,
        message: str,
        status_code: int,
        response: Response,
        body: object = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response
        self.body = body


def _response(status: int, headers: dict[str, str] | None = None) -> Response:
    return Response(status, request=Request("POST", "https://example.com"), headers=headers or {})


class TestUiPathErrorHierarchy:
    def test_uipath_error_is_root_of_api_errors(self):
        assert issubclass(UiPathAPIError, UiPathError)
        assert issubclass(UiPathBadRequestError, UiPathError)
        assert issubclass(UiPathRateLimitError, UiPathError)

    def test_api_error_still_httpx(self):
        assert issubclass(UiPathAPIError, HTTPStatusError)

    def test_rate_limit_retry_after_lazy_from_response(self):
        exc = UiPathRateLimitError(
            "slow down",
            request=Request("POST", "https://example.com"),
            response=_response(429, {"retry-after": "30"}),
        )
        assert exc.retry_after == 30.0


class TestAsUiPathError:
    def test_maps_429_to_rate_limit(self):
        err = _FakeVendorError("slow", 429, _response(429, {"retry-after": "12"}))
        converted = as_uipath_error(err)
        assert type(converted) is UiPathRateLimitError
        assert isinstance(converted, UiPathAPIError)
        assert isinstance(converted, UiPathError)
        assert converted.status_code == 429
        assert converted.retry_after == 12.0

    def test_maps_400_to_bad_request(self):
        converted = as_uipath_error(_FakeVendorError("bad", 400, _response(400)))
        assert type(converted) is UiPathBadRequestError

    def test_maps_401_to_authentication(self):
        converted = as_uipath_error(_FakeVendorError("no", 401, _response(401)))
        assert type(converted) is UiPathAuthenticationError

    def test_result_is_pure_uipath_not_vendor_type(self):
        err = _FakeVendorError("bad", 400, _response(400))
        converted = as_uipath_error(err)
        # The conversion drops the vendor lineage entirely.
        assert not isinstance(converted, _FakeVendorError)
        assert _FakeVendorError not in type(converted).__mro__
        assert converted is not err

    def test_unmapped_status_becomes_generic_api_error(self):
        converted = as_uipath_error(_FakeVendorError("teapot", 418, _response(418)))
        assert type(converted) is UiPathAPIError
        assert not isinstance(converted, UiPathBadRequestError)
        assert converted.status_code == 418

    def test_status_recovered_from_cause_chain(self):
        # google shape: a wrapper raised ``from`` the response-bearing error.
        inner = _FakeVendorError("bad", 400, _response(400))
        outer = RuntimeError("Error calling model")
        outer.__cause__ = inner
        converted = as_uipath_error(outer)
        assert type(converted) is UiPathBadRequestError
        assert converted.status_code == 400

    def test_status_recovered_from_context_chain(self):
        inner = _FakeVendorError("bad", 401, _response(401))
        outer = RuntimeError("wrapped")
        outer.__context__ = inner
        assert type(as_uipath_error(outer)) is UiPathAuthenticationError

    def test_no_http_response_gets_root_only(self):
        class _NoResponseError(Exception):
            pass

        converted = as_uipath_error(_NoResponseError("boom"))
        assert type(converted) is UiPathError
        assert not isinstance(converted, UiPathAPIError)

    def test_non_httpx_response_attribute_ignored(self):
        # A ``.response`` that is not an httpx.Response (e.g. botocore's dict)
        # must not be mistaken for one.
        class _BotoLike(Exception):
            response = {"ResponseMetadata": {"HTTPStatusCode": 400}}

        converted = as_uipath_error(_BotoLike("boom"))
        assert type(converted) is UiPathError

    def test_existing_uipath_error_passthrough(self):
        existing = UiPathAPIError.from_response(_response(500))
        assert as_uipath_error(existing) is existing

    def test_presents_as_uipath_type(self):
        converted = as_uipath_error(_FakeVendorError("bad", 400, _response(400)))
        assert type(converted).__name__ == "UiPathBadRequestError"
        assert str(converted).startswith("UiPathBadRequestError")
        assert "Status Code: 400" in str(converted)


class TestWrapProviderErrors:
    def _raise(self, exc: Exception):
        with wrap_provider_errors():
            raise exc

    def test_vendor_error_converted_and_catchable_as_uipath(self):
        for exc_type in (UiPathBadRequestError, UiPathAPIError, UiPathError):
            with pytest.raises(exc_type):
                self._raise(_FakeVendorError("bad", 400, _response(400)))

    def test_vendor_error_not_catchable_as_vendor_type(self):
        with pytest.raises(UiPathBadRequestError) as info:
            self._raise(_FakeVendorError("bad", 400, _response(400)))
        assert not isinstance(info.value, _FakeVendorError)
        # Original provider error is preserved as the cause.
        assert isinstance(info.value.__cause__, _FakeVendorError)

    def test_uipath_error_passes_through_unchanged(self):
        original = UiPathAPIError.from_response(_response(503))
        with pytest.raises(UiPathAPIError) as info:
            self._raise(original)
        assert info.value is original
        assert info.value.__cause__ is None

    def test_builtin_exception_becomes_root_and_chained(self):
        with pytest.raises(UiPathError) as info:
            with wrap_provider_errors():
                raise ValueError("boom")
        assert type(info.value) is UiPathError
        assert not isinstance(info.value, ValueError)
        assert isinstance(info.value.__cause__, ValueError)

    def test_generator_exit_not_wrapped(self):
        with pytest.raises(GeneratorExit):
            with wrap_provider_errors():
                raise GeneratorExit()

    def test_keyboard_interrupt_not_wrapped(self):
        with pytest.raises(KeyboardInterrupt):
            with wrap_provider_errors():
                raise KeyboardInterrupt()
