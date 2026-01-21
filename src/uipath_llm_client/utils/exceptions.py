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

    status_code: int = 400


class UiPathAuthenticationError(UiPathAPIError):
    """HTTP 401 Unauthorized error."""

    status_code: int = 401


class UiPathPermissionDeniedError(UiPathAPIError):
    """HTTP 403 Forbidden error."""

    status_code: int = 403


class UiPathNotFoundError(UiPathAPIError):
    """HTTP 404 Not Found error."""

    status_code: int = 404


class UiPathConflictError(UiPathAPIError):
    """HTTP 409 Conflict error."""

    status_code: int = 409


class UiPathRequestTooLargeError(UiPathAPIError):
    """HTTP 413 Payload Too Large error."""

    status_code: int = 413


class UiPathUnprocessableEntityError(UiPathAPIError):
    """HTTP 422 Unprocessable Entity error."""

    status_code: int = 422


class UiPathRateLimitError(UiPathAPIError):
    """HTTP 429 Too Many Requests error."""

    status_code: int = 429


class UiPathInternalServerError(UiPathAPIError):
    """HTTP 500 Internal Server Error."""

    status_code: int = 500


class UiPathServiceUnavailableError(UiPathAPIError):
    """HTTP 503 Service Unavailable error."""

    status_code: int = 503


class UiPathGatewayTimeoutError(UiPathAPIError):
    """HTTP 504 Gateway Timeout error."""

    status_code: int = 504


class UiPathTooManyRequestsError(UiPathAPIError):
    """HTTP 529 Too Many Requests (Anthropic overload) error."""

    status_code: int = 529


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
