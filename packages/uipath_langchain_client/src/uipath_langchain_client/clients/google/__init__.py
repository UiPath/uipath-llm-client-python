"""LangChain chat models and embeddings for Google Gemini via UiPath.

Requires the ``google`` optional extra::

    uv add uipath-langchain-client[google]
"""

from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.google.embeddings import UiPathGoogleGenerativeAIEmbeddings

__all__ = ["UiPathChatGoogleGenerativeAI", "UiPathGoogleGenerativeAIEmbeddings"]
