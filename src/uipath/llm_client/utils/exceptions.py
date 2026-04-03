"""
Error Utilities for UiPath LLM Client

This module defines custom exception classes for UiPath API errors.
Each exception class corresponds to a specific HTTP status code, allowing
for precise error handling in application code.

These exceptions inherit from httpx.HTTPStatusError, so they can be caught
by both UiPath-specific handlers and generic httpx error handlers.

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

from json import JSONDecodeError
from typing import Literal

from httpx import HTTPStatusError, Request, Response


class UiPathAPIError(HTTPStatusError):
    """Base exception for all UiPath API errors.

    Inherits from httpx.HTTPStatusError for compatibility with httpx error handling.

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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message} (Status Code: {self.status_code}) {self.body}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code}, body={self.body!r})"

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

    def __init__(
        self,
        message: str,
        *,
        request: Request,
        response: Response,
        body: str | dict | None = None,
    ):
        super().__init__(message, request=request, response=response, body=body)
        self._retry_after = self._parse_retry_after(response)

    @property
    def retry_after(self) -> float | None:
        """Get the retry-after value in seconds, if available."""
        return self._retry_after

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
    409: UiPathConflictError,
    413: UiPathRequestTooLargeError,
    422: UiPathUnprocessableEntityError,
    429: UiPathRateLimitError,
    500: UiPathInternalServerError,
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


__all__ = [
    "UiPathAPIError",
    "UiPathBadRequestError",
    "UiPathAuthenticationError",
    "UiPathPermissionDeniedError",
    "UiPathNotFoundError",
    "UiPathConflictError",
    "UiPathRequestTooLargeError",
    "UiPathUnprocessableEntityError",
    "UiPathRateLimitError",
    "UiPathInternalServerError",
    "UiPathServiceUnavailableError",
    "UiPathGatewayTimeoutError",
    "UiPathTooManyRequestsError",
]
