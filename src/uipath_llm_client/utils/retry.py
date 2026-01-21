"""
Retry Utilities for UiPath LLM Client.

This module provides retry logic for HTTP requests with configurable
exponential backoff and jitter. It uses tenacity for retry handling
and integrates with httpx transports.

Example:
    >>> from uipath_llm_client.utils.retry import RetryableHTTPTransport, RetryConfig
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
from typing import Any, Callable, NotRequired, TypedDict

from httpx import AsyncHTTPTransport, HTTPTransport, Request, Response
from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from uipath_llm_client.utils.exceptions import UiPathAPIError, UiPathRateLimitError

__all__ = [
    "RetryConfig",
    "RetryableHTTPTransport",
    "RetryableAsyncHTTPTransport",
]


# Default retry configuration values
_DEFAULT_RETRY_ON_EXCEPTIONS: tuple[type[UiPathAPIError], ...] = (UiPathRateLimitError,)
_DEFAULT_INITIAL_DELAY: float = 2.0
_DEFAULT_MAX_DELAY: float = 60.0
_DEFAULT_EXP_BASE: float = 2.0
_DEFAULT_JITTER: float = 1.0


class RetryConfig(TypedDict):
    """Configuration for retry behavior on failed requests.

    All fields are optional and have sensible defaults when not provided.

    Attributes:
        retry_on_exceptions: Tuple of exception types to retry on.
            Defaults to (UiPathRateLimitError,).
        initial_delay: Initial delay in seconds before first retry.
            Defaults to 2.0.
        max_delay: Maximum delay in seconds between retries.
            Defaults to 60.0.
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

    retry_on_exceptions: NotRequired[tuple[type[UiPathAPIError], ...]]
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
        max_retries: Maximum number of retry attempts. Returns None if <= 1.
        retry_config: Configuration for retry behavior. Uses defaults if not provided.
        logger: Logger for retry attempt warnings.
        async_mode: If True, returns AsyncRetrying; otherwise returns Retrying.

    Returns:
        A configured Retrying/AsyncRetrying instance, or None if retries disabled.
    """
    if max_retries <= 1:
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
        wait=wait_exponential_jitter(
            initial=initial_delay,
            max=max_delay,
            exp_base=exp_base,
            jitter=jitter,
        ),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
        before_sleep=before_sleep,
    )


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
        max_retries: int = 1,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the retryable transport.

        Args:
            max_retries: Maximum number of retry attempts. Set to 1 to disable retries.
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
        parent_handle = super().handle_request

        def _send() -> Response:
            response = parent_handle(request)
            if response.is_error:
                raise UiPathAPIError.from_response(response, request)
            return response

        try:
            if self.retryer is not None:
                return self.retryer(_send)
            else:
                return _send()
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
        max_retries: int = 1,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the retryable async transport.

        Args:
            max_retries: Maximum number of retry attempts. Set to 1 to disable retries.
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
        parent_handle = super().handle_async_request

        async def _send() -> Response:
            response = await parent_handle(request)
            if response.is_error:
                raise UiPathAPIError.from_response(response, request)
            return response

        try:
            if self.retryer is not None:
                return await self.retryer(_send)
            else:
                return await _send()
        except UiPathAPIError as e:
            return e.response
