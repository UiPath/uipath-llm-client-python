"""LangChain chat model for Anthropic Claude via UiPath.

Requires the ``anthropic`` optional extra::

    uv add uipath-langchain-client[anthropic]
"""

from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic

__all__ = ["UiPathChatAnthropic"]
