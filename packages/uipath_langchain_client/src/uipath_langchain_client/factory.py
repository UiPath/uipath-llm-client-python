"""
Factory Module for UiPath LangChain Client

This module provides factory functions that automatically detect the appropriate
LangChain model class based on the model name and vendor. This simplifies usage
by eliminating the need to manually import provider-specific classes.

The factory queries UiPath's discovery endpoint to determine which vendor
(OpenAI, Google, Anthropic, etc.) provides a given model, then instantiates
the correct LangChain wrapper class.

Example:
    >>> from uipath_langchain_client import get_chat_model, get_embedding_model
    >>> from uipath_langchain_client.settings import get_default_client_settings
    >>>
    >>> settings = get_default_client_settings()
    >>>
    >>> # Auto-detect vendor from model name
    >>> chat = get_chat_model("gpt-4o-2024-11-20", settings)
    >>> embeddings = get_embedding_model("text-embedding-3-large", settings)
"""

from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from uipath_langchain_client.settings import UiPathBaseSettings, get_default_client_settings


def _get_model_info(
    model_name: str,
    client_settings: UiPathBaseSettings,
    byo_connection_id: str | None = None,
) -> dict[str, Any]:
    available_models = client_settings.get_available_models()

    matching_models = [m for m in available_models if m["modelName"].lower() == model_name.lower()]

    if byo_connection_id:
        matching_models = [
            m
            for m in matching_models
            if (byom_details := m.get("byomDetails"))
            and byom_details.get("integrationServiceConnectionId", "").lower()
            == byo_connection_id.lower()
        ]

    if not byo_connection_id and len(matching_models) > 1:
        matching_models = [m for m in matching_models if m.get("byomDetails") is None]

    if not matching_models:
        raise ValueError(
            f"Model {model_name} not found in available models the available models are: {[m['modelName'] for m in available_models]}"
        )

    return matching_models[0]


def get_chat_model(
    model_name: str,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    client_type: Literal["passthrough", "normalized"] = "passthrough",
    **model_kwargs: Any,
) -> BaseChatModel:
    """Factory function to create the appropriate LangChain chat model for a given model name.

    Automatically detects the model vendor and returns the correct LangChain model class.

    Args:
        model: Name of the model to use (e.g., "gpt-4", "claude-3-opus")
        client_type: Use "normalized" for provider-agnostic API or "passthrough" for vendor-specific
        **model_kwargs: Additional keyword arguments to pass to the model constructor

    Returns:
        A LangChain BaseChatModel instance configured for the specified model

    Raises:
        ValueError: If the model is not found in available models or vendor is not supported
    """
    client_settings = client_settings or get_default_client_settings()
    model_info = _get_model_info(model_name, client_settings, byo_connection_id)

    if client_type == "normalized":
        from uipath_langchain_client.clients.normalized.chat_models import (
            UiPathNormalizedChatModel,
        )

        return UiPathNormalizedChatModel(model=model_name, **model_kwargs)

    vendor_type = model_info["vendor"].lower()
    match vendor_type:
        case "openai":
            if "gpt" in model_name:
                from uipath_langchain_client.clients.openai.chat_models import (
                    UiPathAzureChatOpenAI,
                )

                return UiPathAzureChatOpenAI(
                    model=model_name,
                    client_settings=client_settings,
                    **model_kwargs,
                )
            else:
                raise ValueError(f"Invalid model name: {model_name} for vendor: {vendor_type}")
        case "vertexai":
            if "gemini" in model_name:
                from uipath_langchain_client.clients.google.chat_models import (
                    UiPathChatGoogleGenerativeAI,
                )

                return UiPathChatGoogleGenerativeAI(
                    model=model_name,
                    client_settings=client_settings,
                    **model_kwargs,
                )
            elif "claude" in model_name:
                from uipath_langchain_client.clients.anthropic.chat_models import (
                    UiPathChatAnthropic,
                )

                return UiPathChatAnthropic(
                    model=model_name,
                    client_settings=client_settings,
                    vendor_type="vertexai",
                    **model_kwargs,
                )
            else:
                raise ValueError(f"Invalid model name: {model_name} for vendor: {vendor_type}")
        case "awsbedrock":
            if "claude" in model_name:
                from uipath_langchain_client.clients.anthropic.chat_models import (
                    UiPathChatAnthropic,
                )

                return UiPathChatAnthropic(
                    model=model_name,
                    client_settings=client_settings,
                    vendor_type="awsbedrock",
                    **model_kwargs,
                )
            else:
                raise ValueError(f"Invalid model name: {model_name} for vendor: {vendor_type}")
        case _:
            raise ValueError(f"Invalid UiPath vendor type: {vendor_type}")


def get_embedding_model(
    model: str,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    client_type: Literal["passthrough", "normalized"] = "passthrough",
    **model_kwargs: Any,
) -> Embeddings:
    """Factory function to create the appropriate LangChain embeddings model.

    Automatically detects the model vendor and returns the correct LangChain embeddings class.

    Args:
        model: Name of the embeddings model (e.g., "text-embedding-3-large").
        client_settings: Client settings for authentication and routing.
        client_type: API mode - "normalized" for provider-agnostic API or
            "passthrough" for vendor-specific APIs.
        **model_kwargs: Additional arguments passed to the embeddings constructor.

    Returns:
        A LangChain Embeddings instance configured for the specified model.

    Raises:
        ValueError: If the model is not found or the vendor is not supported.

    Example:
        >>> settings = get_default_client_settings()
        >>> embeddings = get_embedding_model("text-embedding-3-large", settings)
        >>> vectors = embeddings.embed_documents(["Hello world"])
    """
    client_settings = client_settings or get_default_client_settings()
    model_info = _get_model_info(model, client_settings, byo_connection_id)

    if client_type == "normalized":
        from uipath_langchain_client.clients.normalized.embeddings import (
            UiPathNormalizedEmbeddings,
        )

        return UiPathNormalizedEmbeddings(
            model=model, client_settings=client_settings, **model_kwargs
        )

    vendor_type = model_info["vendor"].lower()
    match vendor_type:
        case "openai":
            from uipath_langchain_client.clients.openai.embeddings import (
                UiPathAzureOpenAIEmbeddings,
            )

            return UiPathAzureOpenAIEmbeddings(
                model=model, client_settings=client_settings, **model_kwargs
            )
        case "vertexai":
            from uipath_langchain_client.clients.google.embeddings import (
                UiPathGoogleGenerativeAIEmbeddings,
            )

            return UiPathGoogleGenerativeAIEmbeddings(
                model=model, client_settings=client_settings, **model_kwargs
            )
        case "awsbedrock":
            from uipath_langchain_client.clients.bedrock.embeddings import (
                UiPathBedrockEmbeddings,
            )

            return UiPathBedrockEmbeddings(
                model=model, client_settings=client_settings, **model_kwargs
            )
        case _:
            raise ValueError(f"Invalid UiPath Embeddings provider: {vendor_type}")
