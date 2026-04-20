"""Tests for the merge behavior between class-level and instance default headers
on the langchain `UiPathBaseLLMClient`.

The class-level `class_default_headers` (timeout, 4xx-response policy) must
always be present. User-supplied `default_headers` are merged on top and win on
key collisions, but they must not remove class-level defaults.
"""

import os
from unittest.mock import patch

from uipath_langchain_client.clients.normalized.chat_models import UiPathChat

from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}


class TestClassDefaultHeadersAlwaysPresent:
    def setup_method(self):
        SingletonMeta._instances.clear()

    def teardown_method(self):
        SingletonMeta._instances.clear()

    def test_no_user_headers_preserves_class_defaults(self):
        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(model="gpt-4o", settings=LLMGatewaySettings())
            headers = chat.uipath_sync_client.headers
        assert headers.get("x-uipath-llmgateway-timeoutseconds") == "295"
        assert headers.get("x-uipath-llmgateway-allowfull4xxresponse") == "false"

    def test_user_headers_do_not_remove_class_defaults(self):
        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(
                model="gpt-4o",
                settings=LLMGatewaySettings(),
                default_headers={"x-my-custom": "value"},
            )
            headers = chat.uipath_sync_client.headers
        assert headers.get("x-uipath-llmgateway-timeoutseconds") == "295"
        assert headers.get("x-uipath-llmgateway-allowfull4xxresponse") == "false"
        assert headers.get("x-my-custom") == "value"

    def test_user_headers_override_class_defaults_on_collision(self):
        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(
                model="gpt-4o",
                settings=LLMGatewaySettings(),
                default_headers={"X-UiPath-LLMGateway-TimeoutSeconds": "60"},
            )
            headers = chat.uipath_sync_client.headers
        assert headers.get("x-uipath-llmgateway-timeoutseconds") == "60"
        assert headers.get("x-uipath-llmgateway-allowfull4xxresponse") == "false"

    def test_async_client_also_merges(self):
        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(
                model="gpt-4o",
                settings=LLMGatewaySettings(),
                default_headers={"x-my-custom": "async-value"},
            )
            headers = chat.uipath_async_client.headers
        assert headers.get("x-uipath-llmgateway-timeoutseconds") == "295"
        assert headers.get("x-uipath-llmgateway-allowfull4xxresponse") == "false"
        assert headers.get("x-my-custom") == "async-value"
