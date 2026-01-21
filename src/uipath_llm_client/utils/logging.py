"""
Logging utilities for UiPath LLM Client.

This module provides logging configuration and helpers for tracking
request/response timing and errors in httpx event hooks.

Example:
    >>> logging_config = LoggingConfig(
    ...     model_name="gpt-4",
    ...     logger=logging.getLogger(__name__),
    ...     api_config=api_config,
    ... )
    >>> # Use with httpx event hooks
    >>> client = httpx.Client(
    ...     event_hooks={
    ...         "request": [logging_config.log_request_duration],
    ...         "response": [logging_config.log_response_duration],
    ...     }
    ... )
"""

import logging
import time

from httpx import Request, Response
from pydantic import BaseModel, ConfigDict

from uipath_llm_client.settings import UiPathAPIConfig


class LoggingConfig(BaseModel):
    """Configuration for request/response logging.

    This class provides methods suitable for use as httpx event hooks
    to track request duration and log errors.

    Attributes:
        model_name: The LLM model name for logging context.
        logger: The logger instance to use. If None, logging is disabled.
        api_config: API configuration for logging request type context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str | None = None
    logger: logging.Logger | None = None
    api_config: UiPathAPIConfig | None = None

    def log_request_duration(self, request: Request) -> None:
        """Record request start time for duration tracking.

        Stores the start time in request.extensions for later retrieval
        when the response is received.

        Args:
            request: The httpx Request object.
        """
        if self.logger is not None:
            request.extensions["start_time"] = time.monotonic()

    def log_response_duration(self, response: Response) -> None:
        """Calculate and log the request duration.

        Retrieves the start time from the request extensions and logs
        the total request duration along with contextual information.

        Args:
            response: The httpx Response object.
        """
        if self.logger is None:
            return

        start_time = response.request.extensions.get("start_time")
        if start_time is None:
            return

        duration = time.monotonic() - start_time
        client_type = self.api_config.client_type if self.api_config is not None else "unknown"
        api_type = self.api_config.api_type if self.api_config is not None else "unknown"
        request_type = f"{client_type} - {api_type}"

        self.logger.info(
            "[uipath_llm_client] Request to %s took %.2f seconds.",
            response.request.url,
            duration,
            extra={
                "requestUrl": str(response.request.url),
                "duration": f"{duration:.2f}",
                "modelName": self.model_name,
                "requestType": request_type,
            },
        )

    async def alog_request_duration(self, request: Request) -> None:
        """Async event hook for recording request start time.

        This is an async wrapper for log_request_duration, suitable for
        use with httpx.AsyncClient event hooks.

        Args:
            request: The httpx Request object.
        """
        self.log_request_duration(request)

    async def alog_response_duration(self, response: Response) -> None:
        """Async event hook for logging request duration.

        This is an async wrapper for log_response_duration, suitable for
        use with httpx.AsyncClient event hooks.

        Args:
            response: The httpx Response object.
        """
        self.log_response_duration(response)

    def log_error(self, response: Response) -> None:
        """Log an error if the response indicates a failure.

        Checks if the response has a non-2xx status code and logs
        an error message with the status code and reason phrase.

        Args:
            response: The httpx Response object.
        """
        if self.logger is None:
            return

        if response.is_error:
            self.logger.error(
                "[uipath_llm_client] Error querying LLM: %s (%s)",
                response.reason_phrase,
                response.status_code,
            )

    async def alog_error(self, response: Response) -> None:
        """Async event hook for logging errors.

        This is an async wrapper for log_error, suitable for use with
        httpx.AsyncClient event hooks.

        Args:
            response: The httpx Response object.
        """
        self.log_error(response)
