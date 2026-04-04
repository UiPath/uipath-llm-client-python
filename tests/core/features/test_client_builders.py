"""Tests for UiPathHttpxClient / UiPathHttpxAsyncClient with client_settings."""

from unittest.mock import MagicMock

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings.base import UiPathAPIConfig


class TestUiPathHttpxClientWithSettings:
    def _make_mock_settings(self):
        settings = MagicMock()
        settings.build_base_url.return_value = "https://gateway.uipath.com/llm/v1"
        settings.build_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        settings.build_auth_pipeline.return_value = None
        return settings

    def test_returns_uipath_httpx_client(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="completions", routing_mode="normalized")

        client = UiPathHttpxClient(
            model_name="gpt-4o",
            client_settings=settings,
            api_config=api_config,
            max_retries=3,
        )

        assert isinstance(client, UiPathHttpxClient)

    def test_with_auth_and_model_name(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="completions", routing_mode="normalized")

        client = UiPathHttpxClient(
            model_name="gpt-4o",
            client_settings=settings,
            api_config=api_config,
            headers={"X-Custom": "value"},
        )

        assert isinstance(client, UiPathHttpxClient)
        settings.build_auth_headers.assert_called_once_with(
            model_name="gpt-4o", api_config=api_config
        )
        settings.build_base_url.assert_called_once_with(model_name="gpt-4o", api_config=api_config)

    def test_without_api_config(self):
        settings = self._make_mock_settings()

        client = UiPathHttpxClient(
            model_name="gpt-4o",
            client_settings=settings,
        )

        assert isinstance(client, UiPathHttpxClient)
        settings.build_auth_headers.assert_not_called()
        settings.build_base_url.assert_not_called()

    def test_without_client_settings(self):
        client = UiPathHttpxClient(model_name="gpt-4o")
        assert isinstance(client, UiPathHttpxClient)


class TestUiPathHttpxAsyncClientWithSettings:
    def _make_mock_settings(self):
        settings = MagicMock()
        settings.build_base_url.return_value = "https://gateway.uipath.com/llm/v1"
        settings.build_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        settings.build_auth_pipeline.return_value = None
        return settings

    def test_returns_uipath_httpx_async_client(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="completions", routing_mode="normalized")

        client = UiPathHttpxAsyncClient(
            model_name="gpt-4o",
            client_settings=settings,
            api_config=api_config,
            max_retries=3,
        )

        assert isinstance(client, UiPathHttpxAsyncClient)

    def test_with_auth_and_model_name(self):
        settings = self._make_mock_settings()
        api_config = UiPathAPIConfig(api_type="embeddings", routing_mode="normalized")

        client = UiPathHttpxAsyncClient(
            model_name="text-embedding-ada-002",
            client_settings=settings,
            api_config=api_config,
            headers={"X-Custom": "header"},
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

        client = UiPathHttpxAsyncClient(
            model_name="gpt-4o",
            client_settings=settings,
        )

        assert isinstance(client, UiPathHttpxAsyncClient)
        settings.build_auth_headers.assert_not_called()
        settings.build_base_url.assert_not_called()
