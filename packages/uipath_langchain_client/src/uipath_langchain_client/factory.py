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
    >>> chat = get_chat_model(model_name="gpt-4o-2024-11-20", client_settings=settings)
    >>> embeddings = get_embedding_model(model_name="text-embedding-3-large", client_settings=settings)
"""

from typing import Any, Literal

from uipath_langchain_client.base_client import (
    UiPathBaseChatModel,
    UiPathBaseEmbeddings,
)
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
        matching_models = [
            m
            for m in matching_models
            if (
                (m.get("modelSubscriptionType", "") == "UiPathOwned")
                or (m.get("byomDetails") is None)
            )
        ]

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
) -> UiPathBaseChatModel:
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
    is_uipath_owned = model_info.get("modelSubscriptionType") == "UiPathOwned"
    if not is_uipath_owned:
        client_settings.validate_byo_model(model_info)

    if client_type == "normalized":
        from uipath_langchain_client.clients.normalized.chat_models import (
            UiPathChat,
        )

        return UiPathChat(
            model=model_name,
            settings=client_settings,
            byo_connection_id=byo_connection_id,
            **model_kwargs,
        )

    vendor_type = model_info["vendor"].lower()
    match vendor_type:
        case "openai":
            if is_uipath_owned:
                from uipath_langchain_client.clients.openai.chat_models import (
                    UiPathAzureChatOpenAI,
                )

                return UiPathAzureChatOpenAI(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )
            else:
                from uipath_langchain_client.clients.openai.chat_models import (
                    UiPathChatOpenAI,
                )

                return UiPathChatOpenAI(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )
        case "vertexai":
            if is_uipath_owned:
                if "claude" in model_name:
                    from uipath_langchain_client.clients.anthropic.chat_models import (
                        UiPathChatAnthropic,
                    )

                    return UiPathChatAnthropic(
                        model=model_name,
                        settings=client_settings,
                        vendor_type=vendor_type,
                        byo_connection_id=byo_connection_id,
                        **model_kwargs,
                    )
                elif "gemini" in model_name:
                    from uipath_langchain_client.clients.google.chat_models import (
                        UiPathChatGoogleGenerativeAI,
                    )

                    return UiPathChatGoogleGenerativeAI(
                        model=model_name,
                        settings=client_settings,
                        byo_connection_id=byo_connection_id,
                        **model_kwargs,
                    )
                else:
                    raise ValueError(
                        f"We don't have a client that currently supports this model: {model_name} on vendor: {vendor_type}"
                    )
            else:
                from uipath_langchain_client.clients.google.chat_models import (
                    UiPathChatGoogleGenerativeAI,
                )

                return UiPathChatGoogleGenerativeAI(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )
        case "awsbedrock":
            if is_uipath_owned:
                if "claude" in model_name:
                    from uipath_langchain_client.clients.anthropic.chat_models import (
                        UiPathChatAnthropic,
                    )

                    return UiPathChatAnthropic(
                        model=model_name,
                        settings=client_settings,
                        vendor_type=vendor_type,
                        byo_connection_id=byo_connection_id,
                        **model_kwargs,
                    )
                else:
                    from uipath_langchain_client.clients.bedrock.chat_models import (
                        UiPathChatBedrock,
                    )

                    return UiPathChatBedrock(
                        model=model_name,
                        settings=client_settings,
                        byo_connection_id=byo_connection_id,
                        **model_kwargs,
                    )

            else:
                from uipath_langchain_client.clients.bedrock.chat_models import (
                    UiPathChatBedrockConverse,
                )

                return UiPathChatBedrockConverse(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )
        case _:
            raise ValueError(
                f"Invalid vendor type: {vendor_type}, we don't currently have clients that support this vendor"
            )


def get_embedding_model(
    model_name: str,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    client_type: Literal["passthrough", "normalized"] = "passthrough",
    **model_kwargs: Any,
) -> UiPathBaseEmbeddings:
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
        >>> embeddings = get_embedding_model(model_name="text-embedding-3-large", client_settings=settings)
        >>> vectors = embeddings.embed_documents(["Hello world"])
    """
    client_settings = client_settings or get_default_client_settings()
    model_info = _get_model_info(model_name, client_settings, byo_connection_id)
    is_uipath_owned = model_info.get("modelSubscriptionType") == "UiPathOwned"
    if not is_uipath_owned:
        client_settings.validate_byo_model(model_info)

    if client_type == "normalized":
        from uipath_langchain_client.clients.normalized.embeddings import (
            UiPathEmbeddings,
        )

        return UiPathEmbeddings(
            model=model_name,
            settings=client_settings,
            byo_connection_id=byo_connection_id,
            **model_kwargs,
        )

    vendor_type = model_info["vendor"].lower()
    match vendor_type:
        case "openai":
            if is_uipath_owned:
                from uipath_langchain_client.clients.openai.embeddings import (
                    UiPathAzureOpenAIEmbeddings,
                )

                return UiPathAzureOpenAIEmbeddings(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )
            else:
                from uipath_langchain_client.clients.openai.embeddings import (
                    UiPathOpenAIEmbeddings,
                )

                return UiPathOpenAIEmbeddings(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )

        case "vertexai":
            from uipath_langchain_client.clients.google.embeddings import (
                UiPathGoogleGenerativeAIEmbeddings,
            )

            return UiPathGoogleGenerativeAIEmbeddings(
                model=model_name,
                settings=client_settings,
                byo_connection_id=byo_connection_id,
                **model_kwargs,
            )
        case "awsbedrock":
            from uipath_langchain_client.clients.bedrock.embeddings import (
                UiPathBedrockEmbeddings,
            )

            return UiPathBedrockEmbeddings(
                model=model_name,
                settings=client_settings,
                byo_connection_id=byo_connection_id,
                **model_kwargs,
            )
        case _:
            raise ValueError(
                f"Invalid vendor type: {vendor_type}, we don't currently have clients that support this vendor"
            )
