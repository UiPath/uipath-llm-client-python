"""Tests for the UiPathError root and as_uipath_error / wrap_provider_errors.

These cover the unified-taxonomy behaviour: a provider error is re-tagged so it
is catchable as its original type, as the matching UiPath semantic subclass, and
as the UiPathError root.
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
        # No custom __init__ anymore: retry_after is parsed from self.response.
        exc = UiPathRateLimitError(
            "slow down",
            request=Request("POST", "https://example.com"),
            response=_response(429, {"retry-after": "30"}),
        )
        assert exc.retry_after == 30.0


class TestAsUiPathError:
    def test_maps_429_to_rate_limit_in_place(self):
        err = _FakeVendorError("slow", 429, _response(429, {"retry-after": "12"}))
        tagged = as_uipath_error(err)
        assert tagged is err  # re-tagged in place, same object
        assert isinstance(tagged, UiPathRateLimitError)
        assert isinstance(tagged, _FakeVendorError)
        assert isinstance(tagged, UiPathAPIError)
        assert isinstance(tagged, UiPathError)
        assert tagged.status_code == 429
        assert tagged.retry_after == 12.0

    def test_maps_400_to_bad_request(self):
        err = _FakeVendorError("bad", 400, _response(400))
        tagged = as_uipath_error(err)
        assert isinstance(tagged, UiPathBadRequestError)
        assert isinstance(tagged, _FakeVendorError)

    def test_maps_401_to_authentication(self):
        err = _FakeVendorError("no", 401, _response(401))
        tagged = as_uipath_error(err)
        assert isinstance(tagged, UiPathAuthenticationError)

    def test_unmapped_status_becomes_generic_api_error(self):
        err = _FakeVendorError("teapot", 418, _response(418))
        tagged = as_uipath_error(err)
        assert type(tagged) is not UiPathAPIError  # it's the tagged subclass
        assert isinstance(tagged, UiPathAPIError)
        assert not isinstance(tagged, UiPathBadRequestError)
        assert tagged.status_code == 418

    def test_no_http_response_gets_root_marker_only(self):
        class _NoResponseError(Exception):
            pass

        err = _NoResponseError("boom")
        tagged = as_uipath_error(err)
        assert isinstance(tagged, UiPathError)
        assert isinstance(tagged, _NoResponseError)
        assert not isinstance(tagged, UiPathAPIError)

    def test_existing_uipath_error_passthrough(self):
        existing = UiPathAPIError.from_response(_response(500))
        assert as_uipath_error(existing) is existing

    def test_presents_as_uipath_type_but_keeps_vendor_type(self):
        err = _FakeVendorError("bad", 400, _response(400))
        tagged = as_uipath_error(err)
        # Presents as the UiPath semantic type...
        assert type(tagged).__name__ == "UiPathBadRequestError"
        assert type(tagged).__module__ == UiPathBadRequestError.__module__
        # ...while still being the original vendor type.
        assert isinstance(tagged, _FakeVendorError)
        assert _FakeVendorError in type(tagged).__mro__

    def test_root_marker_presents_as_uipath_error(self):
        class _NoResponseError(Exception):
            pass

        tagged = as_uipath_error(_NoResponseError("boom"))
        assert type(tagged).__name__ == "UiPathError"
        assert isinstance(tagged, _NoResponseError)

    def test_str_uses_uipath_format_after_tag(self):
        err = _FakeVendorError("bad", 400, _response(400))
        tagged = as_uipath_error(err)
        assert str(tagged).startswith("UiPathBadRequestError")
        assert "Status Code: 400" in str(tagged)

    def test_tagged_class_is_cached(self):
        a = as_uipath_error(_FakeVendorError("x", 400, _response(400)))
        b = as_uipath_error(_FakeVendorError("y", 400, _response(400)))
        assert type(a) is type(b)


class TestWrapProviderErrors:
    def _raise(self, exc: Exception):
        with wrap_provider_errors():
            raise exc

    def test_vendor_error_catchable_every_way(self):
        for exc_type in (
            _FakeVendorError,
            UiPathBadRequestError,
            UiPathAPIError,
            UiPathError,
        ):
            with pytest.raises(exc_type):
                self._raise(_FakeVendorError("bad", 400, _response(400)))

    def test_uipath_error_passes_through_unchanged(self):
        original = UiPathAPIError.from_response(_response(503))
        with pytest.raises(UiPathAPIError) as info:
            self._raise(original)
        assert info.value is original

    def test_builtin_exception_rebuilt_and_chained(self):
        with pytest.raises(UiPathError) as info:
            with wrap_provider_errors():
                raise ValueError("boom")
        assert isinstance(info.value, ValueError)
        assert isinstance(info.value.__cause__, ValueError)
        assert info.value.__cause__ is not info.value

    def test_generator_exit_not_wrapped(self):
        with pytest.raises(GeneratorExit):
            with wrap_provider_errors():
                raise GeneratorExit()

    def test_keyboard_interrupt_not_wrapped(self):
        with pytest.raises(KeyboardInterrupt):
            with wrap_provider_errors():
                raise KeyboardInterrupt()
