"""Shared constants and fixtures for langchain tests.

Model configurations and completions_config/embeddings_config fixtures have been
distributed to per-provider conftest files under tests/langchain/clients/.
"""

from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.bedrock.embeddings import UiPathBedrockEmbeddings
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.google.embeddings import UiPathGoogleGenerativeAIEmbeddings
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
]
EMBEDDINGS_CLIENTS_CLASSES = [
    UiPathEmbeddings,
    UiPathOpenAIEmbeddings,
    UiPathAzureOpenAIEmbeddings,
    UiPathGoogleGenerativeAIEmbeddings,
    UiPathBedrockEmbeddings,
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
