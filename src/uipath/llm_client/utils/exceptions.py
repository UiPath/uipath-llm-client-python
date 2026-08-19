"""
Error Utilities for UiPath LLM Client

This module defines custom exception classes for UiPath API errors.
Each exception class corresponds to a specific HTTP status code, allowing
for precise error handling in application code.

These exceptions inherit from both UiPathError and httpx.HTTPStatusError, so
they can be caught by a UiPath-wide ``except UiPathError`` handler, by
status-specific UiPath handlers, or by generic httpx error handlers.

For the LangChain passthrough chat models, vendor SDK exceptions (e.g.
``openai.BadRequestError``) are converted into the matching UiPath exception by
:func:`wrap_provider_errors`, so callers handle one taxonomy regardless of which
provider produced the error.

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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from json import JSONDecodeError
from typing import Literal

from httpx import HTTPStatusError, Request, Response

_UNSUPPORTED_MIME_MARKER = "Unsupported MIME type"


class UiPathLLMErrorCode(StrEnum):
    """Stable semantic error codes surfaced by the UiPath LLM client.

    Values are provider-independent and stable across supported LLM providers.
    """

    UNSUPPORTED_MIME_TYPE = "UNSUPPORTED_MIME_TYPE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    EXECUTION_DEADLINE_EXCEEDED = "EXECUTION_DEADLINE_EXCEEDED"


class UiPathError(Exception):
    """Common base class for every error surfaced by the UiPath LLM client.

    Everything the client can raise is catchable as ``UiPathError``:

    * :class:`UiPathAPIError` and its status-specific subclasses, raised by the
      core HTTP client for non-2xx responses.
    * Provider SDK exceptions (e.g. ``openai.BadRequestError``) raised by the
      LangChain passthrough chat models, which :func:`wrap_provider_errors`
      converts into the matching UiPath exception (mapping the HTTP status when
      the error carries an ``httpx.Response``, else the ``UiPathError`` root).
      The original provider exception is preserved as ``__cause__``.

    Catch ``UiPathError`` to handle any UiPath LLM failure regardless of which
    backend or provider produced it::

        try:
            chat.invoke(...)
        except UiPathRateLimitError as e:   # same semantic class for every provider
            backoff(e.retry_after)
        except UiPathError:                 # catch-all across every provider
            ...

    ``UiPathError`` carries normalized semantic context:

    * ``error_code`` — a stable, machine-readable identifier, when known.
    * ``detail`` — a best-effort diagnostic detail from the mapped lower-level
      error.
    """

    error_code: UiPathLLMErrorCode | str | None
    detail: str | None

    def __init__(
        self,
        detail: str | None = None,
        *,
        error_code: UiPathLLMErrorCode | str | None = None,
    ) -> None:
        Exception.__init__(self, detail or "")
        self.error_code = error_code
        self.detail = detail


class ModelNotFoundError(UiPathError, ValueError):
    """Raised when a model name is absent from the discovery API response.

    This is a client-side lookup failure — the model was never returned by the
    discovery endpoint — and is therefore distinct from the HTTP-status
    ``UiPath*Error`` classes (e.g. :class:`UiPathNotFoundError`, which is an
    actual HTTP 404 carrying a response).

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working, and :class:`UiPathError` so it is catchable alongside every other
    UiPath LLM error and carries the stable
    :attr:`UiPathLLMErrorCode.MODEL_NOT_FOUND` code.
    """

    def __init__(self, detail: str | None = None) -> None:
        UiPathError.__init__(self, detail, error_code=UiPathLLMErrorCode.MODEL_NOT_FOUND)


class UiPathExecutionDeadlineError(UiPathError):
    """Raised when the run's execution deadline has passed before an LLM call.

    Raised by the retryable transports before a request attempt (initial or
    retry) when the time budget declared via
    :func:`uipath.llm_client.utils.deadline.set_execution_deadline` is
    exhausted. Deliberately not retryable: the run is out of time and must
    fail cleanly inside its execution window instead of being force-killed by
    the control plane.
    """

    def __init__(self) -> None:
        super().__init__(
            "The run's execution deadline has passed; no time budget remains for this LLM call.",
            error_code=UiPathLLMErrorCode.EXECUTION_DEADLINE_EXCEEDED,
        )


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
        retry_after: Seconds to wait before retrying (from the Retry-After /
            x-retry-after header), or None. A server can attach Retry-After to
            any status code, so this lives on the base class — the retry layer
            treats its presence as an explicit retry request.
    """

    def __init__(
        self,
        message: str,
        *,
        request: Request,
        response: Response,
        body: str | dict | None = None,
        error_code: UiPathLLMErrorCode | str | None = None,
        detail: str | None = None,
    ):
        UiPathError.__init__(self, detail, error_code=error_code)
        HTTPStatusError.__init__(self, message, request=request, response=response)
        self.status_code = response.status_code
        self.message = message
        self.body = body

    @property
    def retry_after(self) -> float | None:
        """Get the retry-after value in seconds, if available.

        Parsed lazily from ``self.response`` (the Retry-After / x-retry-after
        header).
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

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self.message} "
            f"(Status Code: {self.status_code}) {self.body}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"status_code={self.status_code}, body={self.body!r})"
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
        retry_after: Seconds to wait before retrying (from Retry-After header),
            or None. Inherited from :class:`UiPathAPIError`.
    """

    status_code: Literal[429] = 429  # pyright: ignore[reportIncompatibleVariableOverride]


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


class UiPathOriginTimeoutError(UiPathAPIError):
    """HTTP 524 A Timeout Occurred (Cloudflare origin timeout) error."""

    status_code: Literal[524] = 524  # pyright: ignore[reportIncompatibleVariableOverride]


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
    524: UiPathOriginTimeoutError,
    529: UiPathTooManyRequestsError,
}


def patch_raise_for_status(response: Response) -> Response:
    """Patch response.raise_for_status() to raise UiPath-specific exceptions.

    The httpx ``HTTPStatusError`` is routed through :func:`wrap_provider_errors`
    so direct ``raise_for_status()`` callers (the core normalized client, the
    raw ``uipath_request``/``uipath_stream`` API, the Bedrock shim) go through
    the *same* conversion and status mapping as provider SDK exceptions — a
    single entry point. The original ``HTTPStatusError`` is preserved as
    ``__cause__``.
    """
    original_raise_for_status = response.raise_for_status

    def raise_for_status() -> Response:
        with wrap_provider_errors():
            original_raise_for_status()
        return response

    response.raise_for_status = raise_for_status
    return response


def _iter_error_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` then its ``__cause__``/``__context__`` ancestors, once each.

    Providers wrap the underlying error at different depths: openai/anthropic
    raise an error that carries the ``httpx.Response`` directly, while
    langchain-google re-raises ``ChatGoogleGenerativeAIError`` ``from`` the
    underlying ``google.genai`` error (which holds the response). Walking the
    chain lets a single rule recover the HTTP status from either shape.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


_UNSUPPORTED_MIME_MATCHERS: tuple[Callable[[BaseException], bool], ...] = (
    lambda err: isinstance(err, ValueError) and _UNSUPPORTED_MIME_MARKER in str(err),
)


def _as_unsupported_mime_type_error(
    exc: BaseException,
) -> UiPathError | None:
    for err in _iter_error_chain(exc):
        if any(matches(err) for matches in _UNSUPPORTED_MIME_MATCHERS):
            return UiPathError(
                error_code=UiPathLLMErrorCode.UNSUPPORTED_MIME_TYPE,
                detail=str(err),
            )
    return None


_ClientSideClassifier = Callable[[BaseException], UiPathError | None]

# Classifiers for provider errors raised *before* an HTTP response exists
# (client-side request rejection). Consulted in order, but only after the chain
# is confirmed to carry no httpx.Response — HTTP status stays authoritative.
_CLIENT_SIDE_CLASSIFIERS: list[_ClientSideClassifier] = [
    _as_unsupported_mime_type_error,
]


def as_uipath_error(exc: Exception) -> UiPathError:
    """Convert a provider/SDK exception into the matching UiPath exception.

    HTTP status is authoritative: ``exc`` and its cause chain are walked for an
    ``httpx.Response`` first. When one is found, its status code is mapped onto
    the matching :class:`UiPathAPIError` subclass (a provider's 429 becomes a
    :class:`UiPathRateLimitError`, a 400 a :class:`UiPathBadRequestError`, …) so
    semantic handling is identical across providers; an unmapped status becomes
    a generic :class:`UiPathAPIError`. A real response outranks any client-side
    classifier match elsewhere in the chain, which may be incidental
    ``__context__`` rather than the actual failure.

    Only when no response is available anywhere in the chain (a genuinely
    client-side rejection) is ``exc`` offered to each classifier in
    ``_CLIENT_SIDE_CLASSIFIERS``; a match yields :class:`UiPathError` with a
    stable semantic ``error_code`` and diagnostic ``detail``.

    Otherwise the :class:`UiPathError` root is returned — still catchable as
    ``UiPathError``. ``UiPathError`` instances are returned unchanged.

    The returned exception is a *new* object; callers should chain it to the
    original via ``raise ... from exc`` to preserve the provider error as
    ``__cause__``.
    """
    if isinstance(exc, UiPathError):
        return exc
    for err in _iter_error_chain(exc):
        response = getattr(err, "response", None)
        if isinstance(response, Response):
            return UiPathAPIError.from_response(response)
    for classify in _CLIENT_SIDE_CLASSIFIERS:
        if typed_error := classify(exc):
            return typed_error
    return UiPathError(str(exc))


@contextmanager
def wrap_provider_errors() -> Iterator[None]:
    """Convert provider/SDK exceptions into UiPath exceptions.

    Any ``Exception`` raised inside the ``with`` block is converted via
    :func:`as_uipath_error` into the matching :class:`UiPathAPIError` subclass
    (or the :class:`UiPathError` root when no HTTP response is available) and
    re-raised, chained to the original via ``raise ... from``.

    ``UiPathError`` instances pass through untouched — the core HTTP client and
    the Bedrock shim already raise them, so there is a single conversion point.
    Non-``Exception`` ``BaseException`` subclasses (``GeneratorExit``,
    ``KeyboardInterrupt``, ``SystemExit``) are never wrapped.
    """
    try:
        yield
    except UiPathError:
        raise
    except Exception as exc:
        raise as_uipath_error(exc) from exc


__all__ = [
    "UiPathError",
    "UiPathLLMErrorCode",
    "ModelNotFoundError",
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
    "UiPathOriginTimeoutError",
    "UiPathTooManyRequestsError",
    "as_uipath_error",
    "wrap_provider_errors",
]
