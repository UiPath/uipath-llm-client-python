"""LangChain chat model for Anthropic Claude via Google Vertex AI and UiPath.

Requires the ``vertexai`` optional extra::

    uv add uipath-langchain-client[vertexai]
"""

from uipath_langchain_client.clients.vertexai.chat_models import UiPathChatAnthropicVertex

__all__ = ["UiPathChatAnthropicVertex"]
