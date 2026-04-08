"""LangChain chat models and embeddings for Azure AI via UiPath.

Requires the ``azure`` optional extra::

    uv add uipath-langchain-client[azure]
"""

from uipath_langchain_client.clients.azure.chat_models import UiPathAzureAIChatCompletionsModel
from uipath_langchain_client.clients.azure.embeddings import UiPathAzureAIEmbeddingsModel

__all__ = ["UiPathAzureAIChatCompletionsModel", "UiPathAzureAIEmbeddingsModel"]
