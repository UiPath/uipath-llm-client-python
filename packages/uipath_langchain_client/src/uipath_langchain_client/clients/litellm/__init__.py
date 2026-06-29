"""UiPath LangChain LiteLLM chat model and embeddings.

Requires the ``litellm`` optional extra::

    uv add uipath-langchain-client[litellm]
"""

from uipath_langchain_client.clients.litellm.chat_models import UiPathChatLiteLLM
from uipath_langchain_client.clients.litellm.embeddings import UiPathLiteLLMEmbeddings

__all__ = ["UiPathChatLiteLLM", "UiPathLiteLLMEmbeddings"]
