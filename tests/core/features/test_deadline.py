"""Tests for execution-deadline enforcement (PC-4871).

The deadline caps how long an LLM call (including its retries) may run so
serverless agent runs fail cleanly inside their execution window instead of
being force-killed by the control plane.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.utils.deadline import (
    LLM_GATEWAY_TIMEOUT_SECONDS_HEADER,
    clear_execution_deadline,
    get_execution_deadline,
    remaining_time_budget,
    set_execution_deadline,
)
from uipath.llm_client.utils.exceptions import UiPathExecutionDeadlineError
from uipath.llm_client.utils.retry import (
    RetryConfig,
    stop_when_deadline_exhausted,
    wait_retry_after_with_fallback,
)

_NO_DELAY_CONFIG: RetryConfig = {"initial_delay": 0, "max_delay": 0, "jitter": 0}


@pytest.fixture(autouse=True)
def _clean_deadline():
    """Ensure no deadline leaks between tests (ContextVar survives the test)."""
    clear_execution_deadline()
    yield
    clear_execution_deadline()


class TestDeadlineContext:
    """Tests for the deadline ContextVar helpers."""

    def test_no_deadline_by_default(self):
        assert get_execution_deadline() is None
        assert remaining_time_budget() is None

    def test_set_and_remaining(self):
        set_execution_deadline(120)
        remaining = remaining_time_budget()
        assert remaining is not None
        assert 118 < remaining <= 120

    def test_remaining_is_never_negative(self):
        set_execution_deadline(-30)
        assert remaining_time_budget() == 0.0

    def test_clear_unconditionally(self):
        set_execution_deadline(120)
        clear_execution_deadline()
        assert remaining_time_budget() is None

    def test_clear_with_token_restores_previous_value(self):
        set_execution_deadline(120)
        token = set_execution_deadline(30)
        clear_execution_deadline(token)
        remaining = remaining_time_budget()
        assert remaining is not None
        assert remaining > 100  # back to the 120s deadline


class TestFailFastOnlyWhenDeadlinePassed:
    """A passed deadline fails fast without an attempt."""

    def test_sync_raises_without_calling_transport_when_expired(self):
        calls = {"n": 0}

        def handler(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, content=b"{}", request=request)

        set_execution_deadline(-5)  # deadline already passed
        client = UiPathHttpxClient(base_url="https://example.com", max_retries=3)
        try:
            with patch.object(httpx.HTTPTransport, "handle_request", handler):
                with pytest.raises(UiPathExecutionDeadlineError):
                    client.post("/anything", json={})
        finally:
            client.close()

        assert calls["n"] == 0

    @pytest.mark.asyncio
    async def test_async_raises_without_calling_transport_when_expired(self):
        calls = {"n": 0}

        async def handler(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, content=b"{}", request=request)

        set_execution_deadline(-5)
        client = UiPathHttpxAsyncClient(base_url="https://example.com", max_retries=3)
        try:
            with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", handler):
                with pytest.raises(UiPathExecutionDeadlineError):
                    await client.post("/anything", json={})
        finally:
            await client.aclose()

        assert calls["n"] == 0

    def test_small_positive_budget_is_still_attempted(self):
        calls = {"n": 0}

        def handler(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, content=b"{}", request=request)

        set_execution_deadline(2)
        client = UiPathHttpxClient(base_url="https://example.com", max_retries=3)
        try:
            with patch.object(httpx.HTTPTransport, "handle_request", handler):
                response = client.post("/anything", json={})
        finally:
            client.close()

        assert calls["n"] == 1
        assert response.status_code == 200


class TestAttemptClamping:
    """Each attempt's server-side timeout limit shrinks to the remaining budget."""

    @staticmethod
    def _capturing_handler(captured: dict):
        def handler(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            captured["timeout"] = dict(request.extensions.get("timeout") or {})
            captured["header"] = request.headers.get(LLM_GATEWAY_TIMEOUT_SECONDS_HEADER)
            return httpx.Response(200, content=b"{}", request=request)

        return handler

    def test_header_lowered_to_remaining_budget(self):
        captured: dict = {}
        set_execution_deadline(60)
        client = UiPathHttpxClient(base_url="https://example.com", max_retries=3)
        try:
            with patch.object(
                httpx.HTTPTransport, "handle_request", self._capturing_handler(captured)
            ):
                client.post("/anything", json={})
        finally:
            client.close()

        # The default 895s timeout limit must be replaced by the ~60s remaining budget
        assert captured["header"] is not None
        assert 55 <= int(captured["header"]) <= 60

    def test_client_timeout_is_not_modified_by_deadline(self):
        captured: dict = {}
        set_execution_deadline(60)
        client = UiPathHttpxClient(base_url="https://example.com", max_retries=3)
        try:
            with patch.object(
                httpx.HTTPTransport, "handle_request", self._capturing_handler(captured)
            ):
                client.post("/anything", json={})
        finally:
            client.close()

        # httpx default read timeout (5.0, DEFAULT_TIMEOUT_CONFIG) survives untouched
        assert captured["timeout"].get("read") == 5.0

    def test_header_is_never_raised_above_configured_value(self):
        captured: dict = {}
        set_execution_deadline(100_000)  # far beyond the 895s default timeout limit
        client = UiPathHttpxClient(base_url="https://example.com", max_retries=3)
        try:
            with patch.object(
                httpx.HTTPTransport, "handle_request", self._capturing_handler(captured)
            ):
                client.post("/anything", json={})
        finally:
            client.close()

        assert captured["header"] == "895"

    def test_no_deadline_leaves_request_untouched(self):
        captured: dict = {}
        client = UiPathHttpxClient(base_url="https://example.com", max_retries=3)
        try:
            with patch.object(
                httpx.HTTPTransport, "handle_request", self._capturing_handler(captured)
            ):
                client.post("/anything", json={})
        finally:
            client.close()

        assert captured["header"] == "895"
        # httpx default read timeout is 5.0 from DEFAULT_TIMEOUT_CONFIG
        assert captured["timeout"].get("read") == 5.0


class TestRetryLoopStopsAtDeadline:
    """The retry loop is bounded by remaining wall-clock, not just attempts."""

    def test_stop_strategy_inactive_without_deadline(self):
        stop = stop_when_deadline_exhausted()
        assert stop(MagicMock()) is False

    def test_stop_strategy_active_when_deadline_passed(self):
        set_execution_deadline(-1)
        stop = stop_when_deadline_exhausted()
        assert stop(MagicMock()) is True

    def test_stop_strategy_inactive_while_budget_remains(self):
        set_execution_deadline(2)  # small but positive budget must not stop
        stop = stop_when_deadline_exhausted()
        assert stop(MagicMock()) is False

    def test_sync_504_stops_after_first_attempt_when_deadline_passes(self):
        """First attempt runs (ample budget), then the deadline passes and the
        retryer stops instead of exhausting the 5-attempt budget."""
        calls = {"n": 0}

        def handler(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            # Simulate the attempt consuming the whole remaining budget
            clear_execution_deadline()
            set_execution_deadline(-1)
            return httpx.Response(504, content=b"{}", request=request)

        set_execution_deadline(600)
        client = UiPathHttpxClient(
            base_url="https://example.com", max_retries=5, retry_config=_NO_DELAY_CONFIG
        )
        try:
            with patch.object(httpx.HTTPTransport, "handle_request", handler):
                response = client.post("/anything", json={})
        finally:
            client.close()

        assert calls["n"] == 1
        assert response.status_code == 504

    def test_sync_504_still_retries_full_budget_without_deadline(self):
        calls = {"n": 0}

        def handler(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(504, content=b"{}", request=request)

        client = UiPathHttpxClient(
            base_url="https://example.com", max_retries=5, retry_config=_NO_DELAY_CONFIG
        )
        try:
            with patch.object(httpx.HTTPTransport, "handle_request", handler):
                response = client.post("/anything", json={})
        finally:
            client.close()

        assert calls["n"] == 5
        assert response.status_code == 504


class TestWaitIsCappedToBudget:
    """Backoff sleeps never run past the deadline."""

    @staticmethod
    def _retry_state_without_retry_after():
        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock()
        exc.retry_after = None
        mock_outcome.exception.return_value = exc
        retry_state = MagicMock()
        retry_state.outcome = mock_outcome
        return retry_state

    def test_fallback_wait_capped_to_remaining_budget(self):
        strategy = wait_retry_after_with_fallback(initial=1, max=120, exp_base=2, jitter=0)
        set_execution_deadline(12)

        with patch.object(strategy, "fallback_wait", return_value=40.0):
            wait = strategy(self._retry_state_without_retry_after())

        # backoff wanted 40s but only ~12s of budget remain
        assert 11 <= wait <= 12

    def test_retry_after_capped_to_remaining_budget(self):
        strategy = wait_retry_after_with_fallback(initial=1, max=120, exp_base=2, jitter=0)
        set_execution_deadline(3)

        from uipath.llm_client.utils.exceptions import UiPathRateLimitError

        mock_outcome = MagicMock()
        mock_outcome.failed = True
        exc = MagicMock(spec=UiPathRateLimitError)
        exc.retry_after = 60.0
        mock_outcome.exception.return_value = exc
        retry_state = MagicMock()
        retry_state.outcome = mock_outcome

        wait = strategy(retry_state)
        assert 2 <= wait <= 3

    def test_wait_uncapped_without_deadline(self):
        strategy = wait_retry_after_with_fallback(initial=1, max=120, exp_base=2, jitter=0)

        with patch.object(strategy, "fallback_wait", return_value=42.0):
            wait = strategy(self._retry_state_without_retry_after())

        assert wait == 42.0
