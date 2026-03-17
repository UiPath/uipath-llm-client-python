"""
UiPath LangChain Client

LangChain-compatible chat models and embeddings for accessing LLMs through
UiPath's infrastructure (AgentHub or LLM Gateway).

Quick Start:
    >>> from uipath_langchain_client import (
    ...     get_chat_model,
    ...     get_embedding_model,
    ...     get_default_client_settings,
    ... )
    >>>
    >>> # Get settings (auto-detects backend from environment)
    >>> settings = get_default_client_settings()
    >>>
    >>> # Chat model with auto-detected vendor
    >>> chat = get_chat_model(model_name="gpt-4o-2024-11-20", client_settings=settings)
    >>> response = chat.invoke("Hello!")
    >>>
    >>> # Embeddings model
    >>> embeddings = get_embedding_model(model_name="text-embedding-3-large", client_settings=settings)
    >>> vectors = embeddings.embed_documents(["Hello world"])

Settings:
    - get_default_client_settings(): Auto-detect backend from environment
    - PlatformSettings: UiPath Platform backend (AgentHub/Orchestrator, CLI-based auth)
    - LLMGatewaySettings: UiPath LLM Gateway backend (S2S auth)

Factory Functions:
    - get_chat_model(): Create a chat model with auto-detected vendor
    - get_embedding_model(): Create an embeddings model with auto-detected vendor
"""

from uipath_langchain_client.__version__ import __version__
from uipath_langchain_client.clients import UiPathChat, UiPathEmbeddings
from uipath_langchain_client.factory import get_chat_model, get_embedding_model
from uipath_langchain_client.settings import (
    LLMGatewaySettings,
    PlatformSettings,
    get_default_client_settings,
)

__all__ = [
    "__version__",
    "get_chat_model",
    "get_embedding_model",
    "UiPathChat",
    "UiPathEmbeddings",
    "get_default_client_settings",
    "LLMGatewaySettings",
    "PlatformSettings",
]
