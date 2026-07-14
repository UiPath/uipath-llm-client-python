"""
Retry Utilities for UiPath LLM Client.

This module provides retry logic for HTTP requests with configurable
exponential backoff and jitter. It uses tenacity for retry handling
and integrates with httpx transports.

The retry logic automatically respects the `Retry-After` or `x-retry-after`
HTTP headers when present in error responses. If the header specifies a wait
time, that value is used (capped at max_delay). Otherwise, exponential backoff
with jitter is applied.

Example:
    >>> from uipath.llm_client.utils.retry import RetryableHTTPTransport, RetryConfig
    >>>
    >>> # Configure retry behavior
    >>> retry_config: RetryConfig = {
    ...     "initial_delay": 1.0,
    ...     "max_delay": 30.0,
    ...     "jitter": 0.5,
    ... }
    >>>
    >>> # Create transport with retry logic
    >>> transport = RetryableHTTPTransport(
    ...     max_retries=3,
    ...     retry_config=retry_config,
    ...     logger=logging.getLogger(__name__),
    ... )
    >>>
    >>> # Use with httpx client
    >>> client = httpx.Client(transport=transport)
"""

import logging
from typing import Any, Callable, NotRequired

from httpx import (
    AsyncHTTPTransport,
    ConnectError,
    ConnectTimeout,
    HTTPTransport,
    ReadTimeout,
    Request,
    Response,
)
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_any,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base
from typing_extensions import TypedDict

from uipath.llm_client.utils.exceptions import (
    UiPathAPIError,
    UiPathBadGatewayError,
    UiPathGatewayTimeoutError,
    UiPathOriginTimeoutError,
    UiPathRateLimitError,
    UiPathRequestTimeoutError,
    UiPathServiceUnavailableError,
    UiPathTooManyRequestsError,
)

# Default retry configuration values, aligned with the legacy
# uipath-langchain-python chat-client retryers (BedrockRetryer) they replaced.
# Status codes retried by default: 408, 429, 502, 503, 504, 524, 529.
# Connection-level failures (connect errors, connect/read timeouts — the httpx
# equivalents of botocore's EndpointConnectionError / ConnectTimeoutError /
# ReadTimeoutError) are retried too. Independently of status, any error
# response carrying a Retry-After header is treated as an explicit server
# request to retry (see _build_retryer).
_DEFAULT_RETRY_ON_EXCEPTIONS: tuple[type[Exception], ...] = (
    UiPathRequestTimeoutError,
    UiPathRateLimitError,
    UiPathBadGatewayError,
    UiPathServiceUnavailableError,
    UiPathGatewayTimeoutError,
    UiPathOriginTimeoutError,
    UiPathTooManyRequestsError,
    ConnectError,
    ConnectTimeout,
    ReadTimeout,
)
_DEFAULT_INITIAL_DELAY: float = 5.0
_DEFAULT_MAX_DELAY: float = 120.0
_DEFAULT_EXP_BASE: float = 2.0
_DEFAULT_JITTER: float = 1.0


class wait_retry_after_with_fallback(wait_base):
    """Custom wait strategy that uses Retry-After header when available.

    This wait strategy checks if the exception is a ``UiPathAPIError`` whose
    response carried a Retry-After / x-retry-after header (any status code)
    and uses that value. If not available, falls back to exponential backoff
    with jitter.

    Attributes:
        fallback_wait: The fallback wait strategy (exponential backoff with jitter).
        max_delay: Maximum delay in seconds (caps retry-after values).
    """

    def __init__(
        self,
        *,
        initial: float,
        max: float,
        exp_base: float,
        jitter: float,
    ) -> None:
        """Initialize the wait strategy.

        Args:
            initial: Initial delay for exponential backoff.
            max: Maximum delay in seconds (also caps retry-after values).
            exp_base: Exponential backoff base multiplier.
            jitter: Random jitter to add to delays.
        """
        self.fallback_wait = wait_exponential_jitter(
            initial=initial,
            max=max,
            exp_base=exp_base,
            jitter=jitter,
        )
        self.max_delay = max

    def __call__(self, retry_state: RetryCallState) -> float:
        """Calculate the wait time for the next retry.

        Args:
            retry_state: The current retry state from tenacity.

        Returns:
            The number of seconds to wait before the next retry.
        """
        # Honor Retry-After from any API error, not just 429 — servers attach
        # it to 5xx (and occasionally other) responses as an explicit wait hint.
        if retry_state.outcome is not None and retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            if isinstance(exception, UiPathAPIError) and exception.retry_after is not None:
                # Use retry-after value, but cap at max_delay
                return min(exception.retry_after, self.max_delay)

        # Fall back to exponential backoff with jitter
        return self.fallback_wait(retry_state)


class RetryConfig(TypedDict):
    """Configuration for retry behavior on failed requests.

    All fields are optional and have sensible defaults when not provided.

    Attributes:
        retry_on_exceptions: Tuple of exception types to retry on.
            Defaults to the typed exceptions for HTTP 408, 429, 502, 503, 504,
            524, 529 plus httpx connection failures (``ConnectError``,
            ``ConnectTimeout``, ``ReadTimeout``). Independently of this tuple,
            any error response carrying a Retry-After header is retried.
        initial_delay: Initial delay in seconds before first retry.
            Defaults to 5.0.
        max_delay: Maximum delay in seconds between retries.
            Defaults to 120.0.
        exp_base: Exponential backoff base multiplier.
            Defaults to 2.0.
        jitter: Random jitter in seconds to add to delay.
            Defaults to 1.0.

    Example:
        >>> config: RetryConfig = {
        ...     "retry_on_exceptions": (UiPathRateLimitError,),
        ...     "initial_delay": 1.0,
        ...     "max_delay": 30.0,
        ...     "exp_base": 2.0,
        ...     "jitter": 0.5,
        ... }
    """

    retry_on_exceptions: NotRequired[tuple[type[Exception], ...]]
    initial_delay: NotRequired[float]
    max_delay: NotRequired[float]
    exp_base: NotRequired[float]
    jitter: NotRequired[float]


def _build_retryer(
    *,
    max_retries: int,
    retry_config: RetryConfig | None,
    logger: logging.Logger | None,
    async_mode: bool = False,
) -> Retrying | AsyncRetrying | None:
    """Build a tenacity retryer from configuration.

    Args:
        max_retries: Maximum number of retry attempts. Returns None if < 1 (i.e., 0 or negative).
        retry_config: Configuration for retry behavior. Uses defaults if not provided.
        logger: Logger for retry attempt warnings.
        async_mode: If True, returns AsyncRetrying; otherwise returns Retrying.

    Returns:
        A configured Retrying/AsyncRetrying instance, or None if retries disabled.
    """
    if max_retries < 1:
        return None

    cfg = retry_config or {}
    retry_on = cfg.get("retry_on_exceptions", _DEFAULT_RETRY_ON_EXCEPTIONS)
    initial_delay = cfg.get("initial_delay", _DEFAULT_INITIAL_DELAY)
    max_delay = cfg.get("max_delay", _DEFAULT_MAX_DELAY)
    exp_base = cfg.get("exp_base", _DEFAULT_EXP_BASE)
    jitter = cfg.get("jitter", _DEFAULT_JITTER)

    before_sleep: Callable[..., Any] | None = None
    if logger is not None:
        before_sleep = before_sleep_log(logger, logging.WARNING)

    retryer_class = AsyncRetrying if async_mode else Retrying
    return retryer_class(
        stop=stop_after_attempt(max_retries),
        wait=wait_retry_after_with_fallback(
            initial=initial_delay,
            max=max_delay,
            exp_base=exp_base,
            jitter=jitter,
        ),
        # Retry on the configured exception types, and additionally on ANY
        # error response carrying a Retry-After header — the server explicitly
        # asked for a retry, regardless of status code (legacy-client parity).
        retry=retry_any(
            retry_if_exception_type(retry_on),
            retry_if_exception(_server_requested_retry),
        ),
        reraise=True,
        before_sleep=before_sleep,
    )


def _server_requested_retry(exception: BaseException) -> bool:
    """True when an error response carries a Retry-After / x-retry-after hint."""
    return isinstance(exception, UiPathAPIError) and exception.retry_after is not None


class RetryableHTTPTransport(HTTPTransport):
    """HTTP transport with automatic retry on failures.

    Wraps httpx.HTTPTransport to add retry logic with exponential backoff.
    Retries are triggered on specific exception types (default: rate limit errors).

    Attributes:
        retryer: The tenacity Retrying instance, or None if retries disabled.
    """

    retryer: Retrying | None

    def __init__(
        self,
        *args: Any,
        max_retries: int = 0,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the retryable transport.

        Args:
            max_retries: Maximum number of retry attempts. Set to 0 (default) to disable retries.
            retry_config: Configuration for retry behavior. Uses defaults if not provided.
            logger: Logger for retry attempt warnings.
            *args: Positional arguments passed to HTTPTransport.
            **kwargs: Keyword arguments passed to HTTPTransport.
        """
        super().__init__(*args, **kwargs)
        self.retryer = _build_retryer(  # type: ignore[assignment]
            max_retries=max_retries,
            retry_config=retry_config,
            logger=logger,
            async_mode=False,
        )

    def handle_request(self, request: Request) -> Response:
        """Handle an HTTP request with retry logic.

        Args:
            request: The httpx Request to send.

        Returns:
            The httpx Response. Returns error responses after retries are exhausted
            instead of raising exceptions.
        """
        if self.retryer is None:
            return super().handle_request(request)

        parent_handle = super().handle_request

        def _send() -> Response:
            response = parent_handle(request)
            if response.is_error:
                raise UiPathAPIError.from_response(response, request)
            return response

        try:
            return self.retryer(_send)
        except UiPathAPIError as e:
            return e.response


class RetryableAsyncHTTPTransport(AsyncHTTPTransport):
    """Async HTTP transport with automatic retry on failures.

    Wraps httpx.AsyncHTTPTransport to add retry logic with exponential backoff.
    Retries are triggered on specific exception types (default: rate limit errors).

    Attributes:
        retryer: The tenacity AsyncRetrying instance, or None if retries disabled.
    """

    retryer: AsyncRetrying | None

    def __init__(
        self,
        *args: Any,
        max_retries: int = 0,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the retryable async transport.

        Args:
            max_retries: Maximum number of retry attempts. Set to 0 (default) to disable retries.
            retry_config: Configuration for retry behavior. Uses defaults if not provided.
            logger: Logger for retry attempt warnings.
            *args: Positional arguments passed to AsyncHTTPTransport.
            **kwargs: Keyword arguments passed to AsyncHTTPTransport.
        """
        super().__init__(*args, **kwargs)
        self.retryer = _build_retryer(  # type: ignore[assignment]
            max_retries=max_retries,
            retry_config=retry_config,
            logger=logger,
            async_mode=True,
        )

    async def handle_async_request(self, request: Request) -> Response:
        """Handle an async HTTP request with retry logic.

        Args:
            request: The httpx Request to send.

        Returns:
            The httpx Response. Returns error responses after retries are exhausted
            instead of raising exceptions.
        """
        if self.retryer is None:
            return await super().handle_async_request(request)

        parent_handle = super().handle_async_request

        async def _send() -> Response:
            response = await parent_handle(request)
            if response.is_error:
                raise UiPathAPIError.from_response(response, request)
            return response

        try:
            return await self.retryer(_send)
        except UiPathAPIError as e:
            return e.response


__all__ = [
    "RetryConfig",
    "RetryableHTTPTransport",
    "RetryableAsyncHTTPTransport",
]
