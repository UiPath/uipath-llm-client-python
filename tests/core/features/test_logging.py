"""Tests for LoggingConfig request/response logging."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import Request, Response

from uipath.llm_client.settings import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode


class TestLoggingConfig:
    """Tests for LoggingConfig request/response logging."""

    def test_log_request_duration_stores_start_time(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_logging")
        config = LoggingConfig(logger=logger, model_name="gpt-4o")
        request = Request("GET", "https://example.com")
        config.log_request_duration(request)
        assert "start_time" in request.extensions
        assert isinstance(request.extensions["start_time"], float)

    def test_log_request_duration_no_op_without_logger(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        config = LoggingConfig(logger=None, model_name="gpt-4o")
        request = Request("GET", "https://example.com")
        config.log_request_duration(request)
        assert "start_time" not in request.extensions

    def test_log_response_duration_logs_info(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_logging_resp")
        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
        )
        config = LoggingConfig(logger=logger, model_name="gpt-4o", api_config=api_config)

        request = Request("GET", "https://example.com")
        request.extensions["start_time"] = time.monotonic() - 1.5

        response = MagicMock(spec=Response)
        response.request = request

        with patch.object(logger, "info") as mock_info:
            config.log_response_duration(response)
            mock_info.assert_called_once()
            args = mock_info.call_args
            assert "took" in args[0][0]

    def test_log_response_duration_no_start_time(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_no_start")
        config = LoggingConfig(logger=logger)

        request = Request("GET", "https://example.com")
        response = MagicMock(spec=Response)
        response.request = request

        with patch.object(logger, "info") as mock_info:
            config.log_response_duration(response)
            mock_info.assert_not_called()

    def test_log_response_duration_no_op_without_logger(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        config = LoggingConfig(logger=None)
        response = MagicMock(spec=Response)
        config.log_response_duration(response)  # Should not raise

    def test_log_error_logs_on_error_response(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_error_log")
        config = LoggingConfig(logger=logger)

        response = MagicMock(spec=Response)
        response.is_error = True
        response.reason_phrase = "Internal Server Error"
        response.status_code = 500

        with patch.object(logger, "error") as mock_error:
            config.log_error(response)
            mock_error.assert_called_once()

    def test_log_error_no_log_on_success(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_no_error")
        config = LoggingConfig(logger=logger)

        response = MagicMock(spec=Response)
        response.is_error = False

        with patch.object(logger, "error") as mock_error:
            config.log_error(response)
            mock_error.assert_not_called()

    def test_log_error_no_op_without_logger(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        config = LoggingConfig(logger=None)
        response = MagicMock(spec=Response)
        response.is_error = True
        config.log_error(response)  # Should not raise

    @pytest.mark.asyncio
    async def test_alog_request_duration_delegates_to_sync(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_async")
        config = LoggingConfig(logger=logger)
        request = Request("GET", "https://example.com")
        await config.alog_request_duration(request)
        assert "start_time" in request.extensions

    @pytest.mark.asyncio
    async def test_alog_error_delegates_to_sync(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_async_error")
        config = LoggingConfig(logger=logger)
        response = MagicMock(spec=Response)
        response.is_error = True
        response.reason_phrase = "Error"
        response.status_code = 500
        with patch.object(logger, "error") as mock_error:
            await config.alog_error(response)
            mock_error.assert_called_once()

    def test_log_response_unknown_api_config(self):
        from uipath.llm_client.utils.logging import LoggingConfig

        logger = logging.getLogger("test_unknown_config")
        config = LoggingConfig(logger=logger, model_name="test", api_config=None)

        request = Request("GET", "https://example.com")
        request.extensions["start_time"] = time.monotonic() - 0.1
        response = MagicMock(spec=Response)
        response.request = request

        with patch.object(logger, "info") as mock_info:
            config.log_response_duration(response)
            mock_info.assert_called_once()
            call_kwargs = mock_info.call_args
            assert "unknown" in str(call_kwargs)
