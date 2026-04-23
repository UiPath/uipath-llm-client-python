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
    API_FLAVOR_TO_VENDOR_TYPE,
    BYOM_TO_ROUTING_FLAVOR,
    ApiFlavor,
    ModelFamily,
    RoutingMode,
    UiPathBaseSettings,
    VendorType,
    get_default_client_settings,
)
from uipath_langchain_client.utils import is_anthropic_model_name


def get_chat_model(
    model_name: str,
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    api_flavor: ApiFlavor | str | None = None,
    custom_class: type[UiPathBaseChatModel] | None = None,
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
        custom_class: A custom class to use for instantiating the chat model instead of the
            auto-detected one. Must be a subclass of UiPathBaseChatModel. When provided,
            the factory skips vendor detection and uses this class directly.
        **model_kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        A LangChain BaseChatModel instance configured for the specified model.

    Raises:
        ValueError: If the model is not found in available models or vendor is not supported.
    """
    client_settings = client_settings or get_default_client_settings()
    model_info = client_settings.get_model_info(
        model_name,
        byo_connection_id=byo_connection_id,
        vendor_type=vendor_type,
    )
    model_family = model_info.get("modelFamily", None)

    model_details = model_info.get("modelDetails")
    if model_details is not None:
        model_kwargs.setdefault("model_details", model_details)

    if custom_class is not None:
        return custom_class(
            model=model_name,
            settings=client_settings,
            byo_connection_id=byo_connection_id,
            **model_kwargs,
        )

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

    discovered_vendor_type = model_info.get("vendor", None)
    discovered_api_flavor = model_info.get("apiFlavor", None)
    if discovered_vendor_type is None and discovered_api_flavor is not None:
        discovered_vendor_type = API_FLAVOR_TO_VENDOR_TYPE.get(discovered_api_flavor, None)
    if discovered_vendor_type is None:
        raise ValueError("No vendor type or api flavor found in model info")
    discovered_vendor_type = discovered_vendor_type.lower()

    # Discovered api_flavor takes precedence over user-supplied api_flavor
    if discovered_api_flavor is not None:
        routing_flavor = BYOM_TO_ROUTING_FLAVOR.get(discovered_api_flavor)
        if routing_flavor is not None:
            api_flavor = routing_flavor
        else:
            api_flavor = discovered_api_flavor

    match discovered_vendor_type:
        case VendorType.OPENAI:
            # OpenAI chat defaults to the Responses API when no flavor is specified.
            if api_flavor is None:
                api_flavor = ApiFlavor.RESPONSES

            if model_family == ModelFamily.OPENAI:
                from uipath_langchain_client.clients.openai.chat_models import (
                    UiPathAzureChatOpenAI,
                )

                return UiPathAzureChatOpenAI(
                    model=model_name,
                    settings=client_settings,
                    api_flavor=api_flavor,
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
                    api_flavor=api_flavor,
                    byo_connection_id=byo_connection_id,
                    **model_kwargs,
                )
        case VendorType.VERTEXAI:
            if model_family == ModelFamily.ANTHROPIC_CLAUDE:
                from uipath_langchain_client.clients.anthropic.chat_models import (
                    UiPathChatAnthropic,
                )

                return UiPathChatAnthropic(
                    model=model_name,
                    settings=client_settings,
                    vendor_type=discovered_vendor_type,
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
            if (
                model_family == ModelFamily.ANTHROPIC_CLAUDE and api_flavor != ApiFlavor.CONVERSE
            ) or (api_flavor == ApiFlavor.INVOKE and is_anthropic_model_name(model_name)):
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
                f"Invalid vendor type: {discovered_vendor_type}, we don't currently have clients that support this vendor"
            )


def get_embedding_model(
    model_name: str,
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    custom_class: type[UiPathBaseEmbeddings] | None = None,
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
        custom_class: A custom class to use for instantiating the embedding model instead of
            the auto-detected one. Must be a subclass of UiPathBaseEmbeddings. When provided,
            the factory skips vendor detection and uses this class directly.
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
    model_info = client_settings.get_model_info(
        model_name,
        byo_connection_id=byo_connection_id,
        vendor_type=vendor_type,
    )
    model_family = model_info.get("modelFamily", None)

    model_details = model_info.get("modelDetails")
    if model_details is not None:
        model_kwargs.setdefault("model_details", model_details)

    if custom_class is not None:
        return custom_class(
            model=model_name,
            settings=client_settings,
            byo_connection_id=byo_connection_id,
            **model_kwargs,
        )

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

    discovered_vendor_type = model_info.get("vendor")
    discovered_api_flavor = model_info.get("apiFlavor")
    if discovered_vendor_type is None and discovered_api_flavor is not None:
        discovered_vendor_type = API_FLAVOR_TO_VENDOR_TYPE.get(discovered_api_flavor)
    if discovered_vendor_type is None:
        raise ValueError(
            f"No vendor type found in model info for embedding model '{model_name}'. "
            f"Model info returned: {model_info}"
        )
    discovered_vendor_type = discovered_vendor_type.lower()
    match discovered_vendor_type:
        case VendorType.OPENAI:
            if model_family == ModelFamily.OPENAI:
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
                f"Invalid vendor type: {discovered_vendor_type}, we don't currently have clients that support this vendor"
            )
