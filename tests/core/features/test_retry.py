"""Tests for retry logic."""

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.utils.exceptions import (
    UiPathBadGatewayError,
    UiPathGatewayTimeoutError,
    UiPathInternalServerError,
    UiPathRateLimitError,
    UiPathRequestTimeoutError,
    UiPathServiceUnavailableError,
    UiPathTooManyRequestsError,
)
from uipath.llm_client.utils.retry import (
    _DEFAULT_RETRY_ON_EXCEPTIONS,
    RetryableAsyncHTTPTransport,
    RetryableHTTPTransport,
    RetryConfig,
)

_NO_DELAY_CONFIG: RetryConfig = {"initial_delay": 0, "max_delay": 0, "jitter": 0}


class TestDefaultRetryOnExceptions:
    """Pins the default retry set so HTTP 408/429/502/503/504/529 stay retryable."""

    def test_default_set_covers_required_status_codes(self):
        assert set(_DEFAULT_RETRY_ON_EXCEPTIONS) == {
            UiPathRequestTimeoutError,
            UiPathRateLimitError,
            UiPathBadGatewayError,
            UiPathServiceUnavailableError,
            UiPathGatewayTimeoutError,
            UiPathTooManyRequestsError,
        }

    def test_internal_server_error_is_not_retried_by_default(self):
        assert UiPathInternalServerError not in _DEFAULT_RETRY_ON_EXCEPTIONS


class TestRetryConfig:
    """Tests for RetryConfig TypedDict."""

    def test_retry_config_defaults(self):
        """Test RetryConfig can be created with defaults."""
        config: RetryConfig = {}
        assert config.get("initial_delay") is None  # Will use default
        assert config.get("max_delay") is None
        assert config.get("jitter") is None

    def test_retry_config_custom_values(self):
        """Test RetryConfig with custom values."""
        config: RetryConfig = {
            "initial_delay": 1.0,
            "max_delay": 30.0,
            "jitter": 0.5,
            "exp_base": 2.0,
            "retry_on_exceptions": (UiPathRateLimitError,),
        }
        assert config["initial_delay"] == 1.0
        assert config["max_delay"] == 30.0
        assert config["jitter"] == 0.5
        assert config["exp_base"] == 2.0


class TestRetryableHTTPTransport:
    """Tests for RetryableHTTPTransport."""

    def test_transport_inherits_from_http_transport(self):
        """Test transport inherits from HTTPTransport."""
        from httpx import HTTPTransport

        assert issubclass(RetryableHTTPTransport, HTTPTransport)

    def test_transport_no_retry_when_max_retries_0(self):
        """Test no retry logic when max_retries is 0."""
        transport = RetryableHTTPTransport(max_retries=0)
        assert transport.retryer is None

    def test_transport_has_retryer_when_max_retries_1(self):
        """Test retryer is created when max_retries is 1."""
        transport = RetryableHTTPTransport(max_retries=1)
        assert transport.retryer is not None

    def test_transport_has_retryer_when_max_retries_gt_1(self):
        """Test retryer is created when max_retries > 1."""
        transport = RetryableHTTPTransport(max_retries=3)
        assert transport.retryer is not None

    def test_transport_with_custom_retry_config(self):
        """Test transport with custom retry config."""
        config: RetryConfig = {
            "initial_delay": 0.1,
            "max_delay": 1.0,
            "jitter": 0.1,
        }
        transport = RetryableHTTPTransport(max_retries=3, retry_config=config)
        assert transport.retryer is not None


class TestRetryableAsyncHTTPTransport:
    """Tests for RetryableAsyncHTTPTransport."""

    def test_async_transport_inherits_from_async_http_transport(self):
        """Test async transport inherits from AsyncHTTPTransport."""
        from httpx import AsyncHTTPTransport

        assert issubclass(RetryableAsyncHTTPTransport, AsyncHTTPTransport)

    def test_async_transport_no_retry_when_max_retries_0(self):
        """Test no retry logic when max_retries is 0."""
        transport = RetryableAsyncHTTPTransport(max_retries=0)
        assert transport.retryer is None

    def test_async_transport_has_retryer_when_max_retries_1(self):
        """Test retryer is created when max_retries is 1."""
        transport = RetryableAsyncHTTPTransport(max_retries=1)
        assert transport.retryer is not None

    def test_async_transport_has_retryer_when_max_retries_gt_1(self):
        """Test retryer is created when max_retries > 1."""
        transport = RetryableAsyncHTTPTransport(max_retries=3)
        assert transport.retryer is not None


class TestWaitRetryAfterWithFallback:
    """Tests for wait_retry_after_with_fallback strategy."""

    def test_uses_retry_after_from_rate_limit_error(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        # Create a mock retry state with a UiPathRateLimitError that has retry_after
        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathRateLimitError)
        exc.retry_after = 15.0
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        wait_time = strategy(retry_state)
        assert wait_time == 15.0

    def test_caps_retry_after_at_max_delay(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=10, exp_base=2, jitter=0)

        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathRateLimitError)
        exc.retry_after = 120.0
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        wait_time = strategy(retry_state)
        assert wait_time == 10.0  # Capped at max

    def test_fallback_to_exponential_backoff(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        # Non-rate-limit error
        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathInternalServerError)
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        with patch.object(strategy, "fallback_wait", return_value=4.0) as mock_fallback:
            wait_time = strategy(retry_state)
            assert wait_time == 4.0
            mock_fallback.assert_called_once_with(retry_state)

    def test_fallback_when_retry_after_is_none(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathRateLimitError)
        exc.retry_after = None
        mock_outcome.exception.return_value = exc

        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        with patch.object(strategy, "fallback_wait", return_value=2.0):
            wait_time = strategy(retry_state)
            assert wait_time == 2.0

    def test_fallback_when_no_outcome(self):
        from uipath.llm_client.utils.retry import wait_retry_after_with_fallback

        strategy = wait_retry_after_with_fallback(initial=1, max=60, exp_base=2, jitter=0)

        retry_state = MagicMock()
        retry_state.outcome = None

        with patch.object(strategy, "fallback_wait", return_value=1.0):
            wait_time = strategy(retry_state)
            assert wait_time == 1.0


class TestBuildRetryer:
    """Tests for _build_retryer function."""

    def test_returns_none_for_zero_retries(self):
        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=0, retry_config=None, logger=None)
        assert result is None

    def test_returns_none_for_negative_retries(self):
        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=-1, retry_config=None, logger=None)
        assert result is None

    def test_returns_retrying_for_sync_mode(self):
        from tenacity import Retrying

        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=3, retry_config=None, logger=None, async_mode=False)
        assert isinstance(result, Retrying)

    def test_returns_async_retrying_for_async_mode(self):
        from tenacity import AsyncRetrying

        from uipath.llm_client.utils.retry import _build_retryer

        result = _build_retryer(max_retries=3, retry_config=None, logger=None, async_mode=True)
        assert isinstance(result, AsyncRetrying)

    def test_with_custom_config(self):
        from tenacity import Retrying

        from uipath.llm_client.utils.retry import _build_retryer

        config: RetryConfig = {
            "initial_delay": 0.5,
            "max_delay": 5.0,
            "exp_base": 3.0,
            "jitter": 0.1,
            "retry_on_exceptions": (UiPathRateLimitError,),
        }
        result = _build_retryer(max_retries=2, retry_config=config, logger=None)
        assert isinstance(result, Retrying)

    def test_with_logger_adds_before_sleep(self):
        from uipath.llm_client.utils.retry import _build_retryer

        logger = logging.getLogger("test_retry_logger")
        result = _build_retryer(max_retries=2, retry_config=None, logger=logger)
        assert result is not None
        assert result.before_sleep is not None


class TestRetryOn504EndToEnd:
    """End-to-end coverage that a 504 response actually triggers the retry loop.

    The other test classes verify the *configuration* (default exception set,
    retryer wiring, wait math). These tests drive a real request through
    ``RetryableHTTPTransport.handle_request`` / ``handle_async_request`` and
    assert the underlying call fires ``max_retries`` times before the final 504
    response is returned.
    """

    def test_sync_504_retries_max_retries_times_then_returns_response(self):
        calls = 0

        def fake_504(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(504, content=b"{}", request=request)

        client = UiPathHttpxClient(
            base_url="https://example.com",
            max_retries=3,
            retry_config=_NO_DELAY_CONFIG,
        )
        try:
            with patch.object(httpx.HTTPTransport, "handle_request", fake_504):
                response = client.post("/anything", json={})
        finally:
            client.close()

        assert calls == 3
        assert response.status_code == 504

    def test_sync_max_retries_zero_makes_exactly_one_call(self):
        calls = 0

        def fake_504(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(504, content=b"{}", request=request)

        client = UiPathHttpxClient(base_url="https://example.com", max_retries=0)
        try:
            with patch.object(httpx.HTTPTransport, "handle_request", fake_504):
                response = client.post("/anything", json={})
        finally:
            client.close()

        assert calls == 1
        assert response.status_code == 504

    @pytest.mark.asyncio
    async def test_async_504_retries_max_retries_times_then_returns_response(self):
        calls = 0

        async def fake_504(
            self: httpx.AsyncHTTPTransport, request: httpx.Request
        ) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(504, content=b"{}", request=request)

        client = UiPathHttpxAsyncClient(
            base_url="https://example.com",
            max_retries=3,
            retry_config=_NO_DELAY_CONFIG,
        )
        try:
            with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", fake_504):
                response = await client.post("/anything", json={})
        finally:
            await client.aclose()

        assert calls == 3
        assert response.status_code == 504
