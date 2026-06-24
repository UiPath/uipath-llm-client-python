"""
Error Utilities for UiPath LLM Client

This module defines custom exception classes for UiPath API errors.
Each exception class corresponds to a specific HTTP status code, allowing
for precise error handling in application code.

These exceptions inherit from both UiPathError and httpx.HTTPStatusError, so
they can be caught by a UiPath-wide ``except UiPathError`` handler, by
status-specific UiPath handlers, or by generic httpx error handlers.

For the LangChain passthrough chat models, vendor SDK exceptions (e.g.
``openai.BadRequestError``) are additionally re-tagged via as_uipath_error so
they too are catchable as UiPathError without losing their original type.

The UiPathAPIError.from_response() factory method automatically creates
the appropriate exception type based on the HTTP response status code.

Example:
    >>> try:
    ...     response = client.uipath_request(request_body=data)
    ... except UiPathRateLimitError:
    ...     # Handle rate limiting with exponential backoff
    ...     pass
    ... except UiPathAuthenticationError:
    ...     # Handle auth failure - refresh token
    ...     pass
    ... except UiPathAPIError as e:
    ...     # Handle other API errors
    ...     print(f"API Error: {e.status_code} - {e.message}")
"""

from collections.abc import Iterator
from contextlib import contextmanager
from json import JSONDecodeError
from typing import Literal, cast

from httpx import HTTPStatusError, Request, Response


class UiPathError(Exception):
    """Common base class for every error surfaced by the UiPath LLM client.

    Two distinct kinds of error are catchable as ``UiPathError``:

    * :class:`UiPathAPIError` and its status-specific subclasses, raised by the
      core HTTP client for non-2xx responses.
    * Vendor SDK exceptions (e.g. ``openai.BadRequestError``) raised by the
      LangChain passthrough chat models. These are re-tagged in place by
      :func:`as_uipath_error` so they remain catchable as *both* their original
      type and ``UiPathError``.

    Catch ``UiPathError`` to handle any UiPath LLM failure regardless of which
    backend or provider produced it::

        try:
            chat.invoke(...)
        except openai.BadRequestError:   # provider-specific handling still works
            ...
        except UiPathError:              # catch-all across every provider
            ...
    """


class UiPathAPIError(UiPathError, HTTPStatusError):
    """Base exception for all UiPath API errors.

    Inherits from :class:`UiPathError` (so it can be caught alongside wrapped
    provider errors) and ``httpx.HTTPStatusError`` (for compatibility with httpx
    error handling).

    Attributes:
        message: Human-readable error message (usually the HTTP reason phrase).
        status_code: The HTTP status code of the response.
        body: The response body (parsed JSON dict or raw string).
        request: The original httpx.Request object.
        response: The original httpx.Response object.
    """

    status_code: int

    def __init__(
        self,
        message: str,
        *,
        request: Request,
        response: Response,
        body: str | dict | None = None,
    ):
        super().__init__(message, request=request, response=response)
        self.status_code = response.status_code
        self.message = message
        self.body = body

    def _message(self) -> object:
        # ``message`` is set by __init__, but a vendor error re-tagged into this
        # class via as_uipath_error() bypasses __init__ — fall back to args.
        message = getattr(self, "message", None)
        if message is None:
            message = self.args[0] if self.args else ""
        return message

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self._message()} "
            f"(Status Code: {getattr(self, 'status_code', None)}) {getattr(self, 'body', None)}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self._message()!r}, "
            f"status_code={getattr(self, 'status_code', None)}, body={getattr(self, 'body', None)!r})"
        )

    @classmethod
    def from_response(cls, response: Response, request: Request | None = None) -> "UiPathAPIError":
        """Create an appropriate UiPathAPIError subclass from an httpx Response.

        Args:
            response: The httpx Response object.
            request: The original httpx Request object. Falls back to response.request if None.

        Returns:
            A UiPathAPIError instance (or subclass) matching the response status code.
        """
        status_code = response.status_code
        exception_class = _STATUS_CODE_TO_EXCEPTION.get(status_code, UiPathAPIError)
        try:
            body = response.json()
        except JSONDecodeError:
            body = response.text
        except Exception:
            body = None
        if request is None:
            request = response.request
        return exception_class(
            response.reason_phrase,
            response=response,
            request=request,
            body=body,
        )


class UiPathBadRequestError(UiPathAPIError):
    """HTTP 400 Bad Request error."""

    status_code: Literal[400] = 400  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathAuthenticationError(UiPathAPIError):
    """HTTP 401 Unauthorized error."""

    status_code: Literal[401] = 401  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathPermissionDeniedError(UiPathAPIError):
    """HTTP 403 Forbidden error."""

    status_code: Literal[403] = 403  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathNotFoundError(UiPathAPIError):
    """HTTP 404 Not Found error."""

    status_code: Literal[404] = 404  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathRequestTimeoutError(UiPathAPIError):
    """HTTP 408 Request Timeout error."""

    status_code: Literal[408] = 408  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathConflictError(UiPathAPIError):
    """HTTP 409 Conflict error."""

    status_code: Literal[409] = 409  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathRequestTooLargeError(UiPathAPIError):
    """HTTP 413 Payload Too Large error."""

    status_code: Literal[413] = 413  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathUnprocessableEntityError(UiPathAPIError):
    """HTTP 422 Unprocessable Entity error."""

    status_code: Literal[422] = 422  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathRateLimitError(UiPathAPIError):
    """HTTP 429 Too Many Requests error.

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header), or None.
    """

    status_code: Literal[429] = 429  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def retry_after(self) -> float | None:
        """Get the retry-after value in seconds, if available.

        Parsed lazily from ``self.response`` so the value is correct both for
        instances built via ``__init__`` and for vendor errors re-tagged into
        this class by :func:`as_uipath_error` (which bypasses ``__init__``).
        """
        response = getattr(self, "response", None)
        if not isinstance(response, Response):
            return None
        return self._parse_retry_after(response)

    @staticmethod
    def _parse_retry_after(response: Response) -> float | None:
        """Parse the Retry-After or x-retry-after header from the response.

        The Retry-After header can be either:
        - A number of seconds (e.g., "120")
        - An HTTP-date (e.g., "Wed, 21 Oct 2015 07:28:00 GMT")

        Args:
            response: The httpx Response object.

        Returns:
            The number of seconds to wait, or None if not present/parseable.
        """
        import time
        from datetime import datetime, timezone

        # Check both header variants (case-insensitive in httpx)
        retry_after_value = response.headers.get("retry-after")
        if retry_after_value is None:
            retry_after_value = response.headers.get("x-retry-after")

        if retry_after_value is None:
            return None

        # Try parsing as integer (seconds)
        try:
            return float(retry_after_value)
        except ValueError:
            pass

        # Try parsing as HTTP-date (RFC 7231 IMF-fixdate format)
        # Example: "Wed, 21 Oct 2015 07:28:00 GMT"
        try:
            retry_date = datetime.strptime(retry_after_value, "%a, %d %b %Y %H:%M:%S GMT")
            retry_date = retry_date.replace(tzinfo=timezone.utc)
            delay = retry_date.timestamp() - time.time()
            return max(0.0, delay)  # Don't return negative delays
        except ValueError:
            pass

        return None


class UiPathInternalServerError(UiPathAPIError):
    """HTTP 500 Internal Server Error."""

    status_code: Literal[500] = 500  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathBadGatewayError(UiPathAPIError):
    """HTTP 502 Bad Gateway error."""

    status_code: Literal[502] = 502  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathServiceUnavailableError(UiPathAPIError):
    """HTTP 503 Service Unavailable error."""

    status_code: Literal[503] = 503  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathGatewayTimeoutError(UiPathAPIError):
    """HTTP 504 Gateway Timeout error."""

    status_code: Literal[504] = 504  # pyright: ignore[reportIncompatibleVariableOverride]


class UiPathTooManyRequestsError(UiPathAPIError):
    """HTTP 529 Too Many Requests (Anthropic overload) error."""

    status_code: Literal[529] = 529  # pyright: ignore[reportIncompatibleVariableOverride]


_STATUS_CODE_TO_EXCEPTION: dict[int, type[UiPathAPIError]] = {
    400: UiPathBadRequestError,
    401: UiPathAuthenticationError,
    403: UiPathPermissionDeniedError,
    404: UiPathNotFoundError,
    408: UiPathRequestTimeoutError,
    409: UiPathConflictError,
    413: UiPathRequestTooLargeError,
    422: UiPathUnprocessableEntityError,
    429: UiPathRateLimitError,
    500: UiPathInternalServerError,
    502: UiPathBadGatewayError,
    503: UiPathServiceUnavailableError,
    504: UiPathGatewayTimeoutError,
    529: UiPathTooManyRequestsError,
}


def patch_raise_for_status(response: Response) -> Response:
    """Patch response.raise_for_status() to raise UiPath-specific exceptions."""
    original_raise_for_status = response.raise_for_status

    def raise_for_status() -> Response:
        try:
            original_raise_for_status()
        except HTTPStatusError:
            raise UiPathAPIError.from_response(response)
        return response

    response.raise_for_status = raise_for_status
    return response


# Cache of (original exception class, UiPath base) -> synthesized subclass.
# Tagged classes are generated once per (original type, base) pair so
# ``isinstance`` results are stable and we never leak a new class per raised
# exception.
_uipath_tagged_classes: dict[tuple[type[Exception], type[Exception]], type[Exception]] = {}


def _uipath_tagged_class(cls: type[Exception], base: type[Exception]) -> type[Exception]:
    """Return a cached ``(base, cls)`` subclass for ``cls``.

    The synthesized class presents as the UiPath ``base``: it takes ``base``'s
    ``__name__`` and ``__module__`` (so ``type(exc).__name__`` reads e.g.
    ``UiPathBadRequestError`` in reprs and tracebacks) while remaining a subclass
    of the original vendor exception ``cls`` -- so it stays catchable as both.
    The original vendor type remains visible via ``__bases__`` / the MRO.
    """
    key = (cls, base)
    tagged = _uipath_tagged_classes.get(key)
    if tagged is None:
        tagged = cast(
            "type[Exception]",
            type(base.__name__, (base, cls), {"__module__": base.__module__}),
        )
        _uipath_tagged_classes[key] = tagged
    return tagged


def _uipath_base_for(exc: Exception) -> type[Exception]:
    """Pick the UiPath base class to tag ``exc`` with.

    When ``exc`` carries a real ``httpx.Response`` (openai, anthropic, httpx,
    litellm, fireworks, ...), map its HTTP status code onto the matching
    :class:`UiPathAPIError` subclass so the error is catchable by its semantic
    UiPath type (e.g. ``UiPathRateLimitError``) across every provider. An
    unmapped status still becomes a generic :class:`UiPathAPIError`.

    When no httpx response is available (botocore, google, plain exceptions), we
    cannot safely claim HTTP semantics, so we tag with the :class:`UiPathError`
    root marker only — still catchable as ``UiPathError`` plus the original type.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, Response):
        return UiPathError
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = response.status_code
    return _STATUS_CODE_TO_EXCEPTION.get(status_code, UiPathAPIError)


def as_uipath_error(exc: Exception) -> Exception:
    """Tag ``exc`` so it is catchable as both its original type and ``UiPathError``.

    The tag is chosen by :func:`_uipath_base_for`: an HTTP-shaped vendor error is
    tagged with the matching :class:`UiPathAPIError` subclass (so a provider's
    429 becomes a ``UiPathRateLimitError``, a 400 a ``UiPathBadRequestError``,
    etc.), while anything else is tagged with the :class:`UiPathError` root.

    For mutable exception types (all vendor SDK exceptions: openai, anthropic,
    botocore, google, httpx) the instance is re-tagged in place via ``__class__``
    assignment, preserving its message, attributes and traceback. The returned
    object *is* ``exc``.

    For immutable built-in exception types (e.g. ``ValueError``) whose memory
    layout forbids ``__class__`` reassignment, a fresh instance of the tagged
    subclass is rebuilt from ``exc.args`` and returned instead; callers should
    chain it with ``raise ... from exc``. If the rebuild fails, a plain
    :class:`UiPathError` carrying the original message is returned as a last
    resort.

    ``UiPathError`` instances (including :class:`UiPathAPIError`) are returned
    unchanged.
    """
    if isinstance(exc, UiPathError):
        return exc
    base = _uipath_base_for(exc)
    tagged_cls = _uipath_tagged_class(type(exc), base)
    try:
        exc.__class__ = tagged_cls
        return exc
    except TypeError:
        try:
            return tagged_cls(*exc.args)
        except Exception:
            return UiPathError(str(exc))


@contextmanager
def wrap_provider_errors() -> Iterator[None]:
    """Re-raise provider/SDK exceptions tagged as :class:`UiPathError`.

    Any ``Exception`` raised inside the ``with`` block is re-tagged via
    :func:`as_uipath_error` so callers can catch it as either its original type
    (e.g. ``openai.BadRequestError``) or ``UiPathError``. ``UiPathError``
    instances pass through untouched, and non-``Exception`` ``BaseException``
    subclasses (``GeneratorExit``, ``KeyboardInterrupt``, ``SystemExit``) are
    never wrapped.
    """
    try:
        yield
    except UiPathError:
        raise
    except Exception as exc:
        tagged = as_uipath_error(exc)
        if tagged is exc:
            raise
        raise tagged from exc


__all__ = [
    "UiPathError",
    "UiPathAPIError",
    "UiPathBadRequestError",
    "UiPathAuthenticationError",
    "UiPathPermissionDeniedError",
    "UiPathNotFoundError",
    "UiPathRequestTimeoutError",
    "UiPathConflictError",
    "UiPathRequestTooLargeError",
    "UiPathUnprocessableEntityError",
    "UiPathRateLimitError",
    "UiPathInternalServerError",
    "UiPathBadGatewayError",
    "UiPathServiceUnavailableError",
    "UiPathGatewayTimeoutError",
    "UiPathTooManyRequestsError",
    "as_uipath_error",
    "wrap_provider_errors",
]
