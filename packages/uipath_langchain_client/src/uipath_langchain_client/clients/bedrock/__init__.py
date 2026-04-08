"""LangChain chat models and embeddings for AWS Bedrock via UiPath.

Requires the ``aws`` optional extra::

    uv add uipath-langchain-client[aws]
"""

from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.bedrock.embeddings import UiPathBedrockEmbeddings

__all__ = [
    "UiPathChatBedrock",
    "UiPathChatBedrockConverse",
    "UiPathChatAnthropicBedrock",
    "UiPathBedrockEmbeddings",
]
