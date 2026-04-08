"""LangChain chat models and embeddings for OpenAI/Azure OpenAI via UiPath.

Requires the ``openai`` optional extra::

    uv add uipath-langchain-client[openai]
"""

from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
)
from uipath_langchain_client.clients.openai.embeddings import (
    UiPathAzureOpenAIEmbeddings,
    UiPathOpenAIEmbeddings,
)

__all__ = [
    "UiPathChatOpenAI",
    "UiPathOpenAIEmbeddings",
    "UiPathAzureChatOpenAI",
    "UiPathAzureOpenAIEmbeddings",
]
