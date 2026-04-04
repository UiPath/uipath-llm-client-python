"""Tests for the shared httpx client builder utilities."""

from unittest.mock import MagicMock

from uipath.llm_client.clients.utils import build_httpx_async_client, build_httpx_client
from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings.base import UiPathAPIConfig


class TestBuildHttpxClient:
    def _make_mock_settings(self):
        settings = MagicMock()
        settings.build_base_url.return_value = "https://gateway.uipath.com/llm/v1"
        settings.build_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        settings.build_auth_pipeline.return_value = None
        return settings

    def test_returns_uipath_httpx_client(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="completions", routing_mode="normalized")

        client = build_httpx_client(
            model_name="gpt-4o",
            byo_connection_id=None,
            client_settings=settings,
            timeout=30.0,
            max_retries=3,
            default_headers=None,
            captured_headers=("x-uipath-",),
            retry_config=None,
            logger=None,
            api_config=api_config,
        )

        assert isinstance(client, UiPathHttpxClient)

    def test_with_auth_and_model_name(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="completions", routing_mode="normalized")

        client = build_httpx_client(
            model_name="gpt-4o",
            byo_connection_id=None,
            client_settings=settings,
            timeout=30.0,
            max_retries=0,
            default_headers={"X-Custom": "value"},
            captured_headers=("x-uipath-",),
            retry_config=None,
            logger=None,
            api_config=api_config,
        )

        assert isinstance(client, UiPathHttpxClient)
        settings.build_auth_headers.assert_called_once_with(
            model_name="gpt-4o", api_config=api_config
        )
        settings.build_base_url.assert_called_once_with(model_name="gpt-4o", api_config=api_config)

    def test_without_api_config(self):
        settings = self._make_mock_settings()

        client = build_httpx_client(
            model_name="gpt-4o",
            byo_connection_id=None,
            client_settings=settings,
            timeout=30.0,
            max_retries=0,
            default_headers=None,
            captured_headers=("x-uipath-",),
            retry_config=None,
            logger=None,
            api_config=None,
        )

        assert isinstance(client, UiPathHttpxClient)
        settings.build_auth_headers.assert_not_called()
        settings.build_base_url.assert_not_called()


class TestBuildHttpxAsyncClient:
    def _make_mock_settings(self):
        settings = MagicMock()
        settings.build_base_url.return_value = "https://gateway.uipath.com/llm/v1"
        settings.build_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        settings.build_auth_pipeline.return_value = None
        return settings

    def test_returns_uipath_httpx_async_client(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="completions", routing_mode="normalized")

        client = build_httpx_async_client(
            model_name="gpt-4o",
            byo_connection_id=None,
            client_settings=settings,
            timeout=30.0,
            max_retries=3,
            default_headers=None,
            captured_headers=("x-uipath-",),
            retry_config=None,
            logger=None,
            api_config=api_config,
        )

        assert isinstance(client, UiPathHttpxAsyncClient)

    def test_with_auth_and_model_name(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="embeddings", routing_mode="normalized")

        client = build_httpx_async_client(
            model_name="text-embedding-ada-002",
            byo_connection_id=None,
            client_settings=settings,
            timeout=60.0,
            max_retries=2,
            default_headers={"X-Custom": "header"},
            captured_headers=("x-uipath-",),
            retry_config=None,
            logger=None,
            api_config=api_config,
        )

        assert isinstance(client, UiPathHttpxAsyncClient)
        settings.build_auth_headers.assert_called_once_with(
            model_name="text-embedding-ada-002", api_config=api_config
        )
        settings.build_base_url.assert_called_once_with(
            model_name="text-embedding-ada-002", api_config=api_config
        )

    def test_without_api_config(self):
        settings = self._make_mock_settings()

        client = build_httpx_async_client(
            model_name="gpt-4o",
            byo_connection_id=None,
            client_settings=settings,
            timeout=30.0,
            max_retries=0,
            default_headers=None,
            captured_headers=("x-uipath-",),
            retry_config=None,
            logger=None,
            api_config=None,
        )

        assert isinstance(client, UiPathHttpxAsyncClient)
        settings.build_auth_headers.assert_not_called()
        settings.build_base_url.assert_not_called()
