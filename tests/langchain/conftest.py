"""Shared constants and fixtures for langchain tests.

Model configurations and completions_config/embeddings_config fixtures have been
distributed to per-provider conftest files under tests/langchain/clients/.
"""

import os
from unittest.mock import patch

import pytest
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.bedrock.embeddings import UiPathBedrockEmbeddings
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.google.embeddings import UiPathGoogleGenerativeAIEmbeddings
from uipath_langchain_client.clients.litellm.chat_models import UiPathChatLiteLLM
from uipath_langchain_client.clients.litellm.embeddings import UiPathLiteLLMEmbeddings
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.normalized.embeddings import UiPathEmbeddings
from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
)
from uipath_langchain_client.clients.openai.embeddings import (
    UiPathAzureOpenAIEmbeddings,
    UiPathOpenAIEmbeddings,
)
from uipath_langchain_client.clients.vertexai.chat_models import UiPathChatAnthropicVertex

from uipath.llm_client.settings import LLMGatewaySettings

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}


@pytest.fixture
def llmgw_settings():
    with patch.dict(os.environ, LLMGW_ENV, clear=True):
        return LLMGatewaySettings()


COMPLETION_CLIENTS_CLASSES = [
    UiPathChat,
    UiPathChatOpenAI,
    UiPathAzureChatOpenAI,
    UiPathChatGoogleGenerativeAI,
    UiPathChatAnthropic,
    UiPathChatAnthropicBedrock,
    UiPathChatAnthropicVertex,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
    UiPathChatLiteLLM,
]
EMBEDDINGS_CLIENTS_CLASSES = [
    UiPathEmbeddings,
    UiPathOpenAIEmbeddings,
    UiPathAzureOpenAIEmbeddings,
    UiPathGoogleGenerativeAIEmbeddings,
    UiPathBedrockEmbeddings,
    UiPathLiteLLMEmbeddings,
]

COMPLETION_MODEL_NAMES = [
    "gpt-4o-2024-11-20",
    "gpt-5.2-2025-12-11",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "claude-haiku-4-5@20251001",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
]

EMBEDDING_MODEL_NAMES = [
    "text-embedding-3-large",
    "gemini-embedding-001",
]
