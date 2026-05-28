"""Tests pinning the default ``max_retries`` value on ``UiPathBaseLLMClient``.

The default must stay at 3 so that every LangChain chat/embedding client built
on top of ``UiPathBaseLLMClient`` (OpenAI, Anthropic, Bedrock, Vertex AI, etc.)
retries transient HTTP failures by default. Callers can still opt out by
passing ``max_retries=0`` explicitly.
"""

import os
from unittest.mock import patch

from uipath_langchain_client.clients.normalized.chat_models import UiPathChat

from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta
from uipath.llm_client.utils.retry import RetryableHTTPTransport

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}


class TestDefaultMaxRetries:
    def setup_method(self):
        SingletonMeta._instances.clear()

    def teardown_method(self):
        SingletonMeta._instances.clear()

    def test_default_max_retries_is_three(self):
        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(model="gpt-4o", settings=LLMGatewaySettings())
            assert chat.max_retries == 3
            transport = chat.uipath_sync_client._transport
            assert isinstance(transport, RetryableHTTPTransport)
            assert transport.retryer is not None

    def test_explicit_zero_disables_retries(self):
        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(model="gpt-4o", settings=LLMGatewaySettings(), max_retries=0)
            assert chat.max_retries == 0
            transport = chat.uipath_sync_client._transport
            assert isinstance(transport, RetryableHTTPTransport)
            assert transport.retryer is None
