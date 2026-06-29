"""LangChain chat models and embeddings for Fireworks AI via UiPath.

Requires the ``fireworks`` optional extra::

    uv add uipath-langchain-client[fireworks]
"""

from uipath_langchain_client.clients.fireworks.chat_models import UiPathChatFireworks
from uipath_langchain_client.clients.fireworks.embeddings import UiPathFireworksEmbeddings

__all__ = ["UiPathChatFireworks", "UiPathFireworksEmbeddings"]
