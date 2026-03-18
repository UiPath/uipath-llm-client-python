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

from typing import Any

from uipath_langchain_client.base_client import (
    UiPathBaseChatModel,
    UiPathBaseEmbeddings,
)
from uipath_langchain_client.settings import (
    ApiFlavor,
    RoutingMode,
    UiPathBaseSettings,
    VendorType,
    get_default_client_settings,
)


def _get_model_info(
    model_name: str,
    *,
    client_settings: UiPathBaseSettings,
    byo_connection_id: str | None = None,
    vendor_type: VendorType | str | None = None,
) -> dict[str, Any]:
    available_models = client_settings.get_available_models()

    matching_models = [m for m in available_models if m["modelName"].lower() == model_name.lower()]

    if vendor_type is not None:
        matching_models = [
            m for m in matching_models if m.get("vendor", "").lower() == str(vendor_type).lower()
        ]

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
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    api_flavor: ApiFlavor | str | None = None,
    **model_kwargs: Any,
) -> UiPathBaseChatModel:
    """Factory function to create the appropriate LangChain chat model for a given model name.

    Automatically detects the model vendor and returns the correct LangChain model class.

    Args:
        model_name: Name of the model to use (e.g., "gpt-4o", "claude-3-opus").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings for authentication and routing.
        routing_mode: Use RoutingMode.NORMALIZED for provider-agnostic API or
            RoutingMode.PASSTHROUGH for vendor-specific.
        vendor_type: Filter models by vendor type (e.g., VendorType.OPENAI).
            If not provided, auto-detected from the model discovery endpoint.
        api_flavor: Vendor-specific API flavor to use. Effects:
            - OpenAI: ApiFlavor.RESPONSES sets use_responses_api=True.
            - Bedrock Claude: Default uses UiPathChatAnthropicBedrock.
              ApiFlavor.CONVERSE uses UiPathChatBedrockConverse,
              ApiFlavor.INVOKE uses UiPathChatBedrock.
        **model_kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        A LangChain BaseChatModel instance configured for the specified model.

    Raises:
        ValueError: If the model is not found in available models or vendor is not supported.
    """
    client_settings = client_settings or get_default_client_settings()
    model_info = _get_model_info(
        model_name,
        client_settings=client_settings,
        byo_connection_id=byo_connection_id,
        vendor_type=vendor_type,
    )
    model_family = model_info.get("modelFamily", None)
    if model_family is not None:
        model_family = model_family.lower()
    is_uipath_owned = model_info.get("modelSubscriptionType") == "UiPathOwned"
    if not is_uipath_owned:
        client_settings.validate_byo_model(model_info)

    if routing_mode == RoutingMode.NORMALIZED:
        from uipath_langchain_client.clients.normalized.chat_models import (
            UiPathChat,
        )

        return UiPathChat(
            model=model_name,
            settings=client_settings,
            byo_connection_id=byo_connection_id,
            **model_kwargs,
        )

    discovered_vendor = model_info["vendor"].lower()
    match discovered_vendor:
        case VendorType.OPENAI:
            if api_flavor == ApiFlavor.RESPONSES:
                model_kwargs["use_responses_api"] = True

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
        case VendorType.VERTEXAI:
            if model_family == "anthropicclaude":
                from uipath_langchain_client.clients.anthropic.chat_models import (
                    UiPathChatAnthropic,
                )

                return UiPathChatAnthropic(
                    model=model_name,
                    settings=client_settings,
                    vendor_type=discovered_vendor,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )

            from uipath_langchain_client.clients.google.chat_models import (
                UiPathChatGoogleGenerativeAI,
            )

            return UiPathChatGoogleGenerativeAI(
                model=model_name,
                settings=client_settings,
                byo_connection_id=byo_connection_id,
                **model_kwargs,
            )
        case VendorType.AWSBEDROCK:
            if model_family == "anthropicclaude" and api_flavor is None:
                from uipath_langchain_client.clients.bedrock.chat_models import (
                    UiPathChatAnthropicBedrock,
                )

                return UiPathChatAnthropicBedrock(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )

            if api_flavor == ApiFlavor.INVOKE:
                from uipath_langchain_client.clients.bedrock.chat_models import (
                    UiPathChatBedrock,
                )

                return UiPathChatBedrock(
                    model=model_name,
                    settings=client_settings,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )

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
                f"Invalid vendor type: {discovered_vendor}, we don't currently have clients that support this vendor"
            )


def get_embedding_model(
    model_name: str,
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    **model_kwargs: Any,
) -> UiPathBaseEmbeddings:
    """Factory function to create the appropriate LangChain embeddings model.

    Automatically detects the model vendor and returns the correct LangChain embeddings class.

    Args:
        model_name: Name of the embeddings model (e.g., "text-embedding-3-large").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: Client settings for authentication and routing.
        routing_mode: API mode - RoutingMode.NORMALIZED for provider-agnostic API or
            RoutingMode.PASSTHROUGH for vendor-specific APIs.
        vendor_type: Filter models by vendor type (e.g., VendorType.OPENAI).
            If not provided, auto-detected from the model discovery endpoint.
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
    model_info = _get_model_info(
        model_name,
        client_settings=client_settings,
        byo_connection_id=byo_connection_id,
        vendor_type=vendor_type,
    )
    is_uipath_owned = model_info.get("modelSubscriptionType") == "UiPathOwned"
    if not is_uipath_owned:
        client_settings.validate_byo_model(model_info)

    if routing_mode == RoutingMode.NORMALIZED:
        from uipath_langchain_client.clients.normalized.embeddings import (
            UiPathEmbeddings,
        )

        return UiPathEmbeddings(
            model=model_name,
            settings=client_settings,
            byo_connection_id=byo_connection_id,
            **model_kwargs,
        )

    discovered_vendor = model_info["vendor"].lower()
    match discovered_vendor:
        case VendorType.OPENAI:
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

        case VendorType.VERTEXAI:
            from uipath_langchain_client.clients.google.embeddings import (
                UiPathGoogleGenerativeAIEmbeddings,
            )

            return UiPathGoogleGenerativeAIEmbeddings(
                model=model_name,
                settings=client_settings,
                byo_connection_id=byo_connection_id,
                **model_kwargs,
            )
        case VendorType.AWSBEDROCK:
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
                f"Invalid vendor type: {discovered_vendor}, we don't currently have clients that support this vendor"
            )
