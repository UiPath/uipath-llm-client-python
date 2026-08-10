from collections.abc import Container, Mapping
from functools import cached_property
from typing import Any, Self

from pydantic import Field, model_validator

from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.settings import (
    ApiFlavor,
    ApiType,
    RoutingMode,
    UiPathAPIConfig,
    VendorType,
)

try:
    from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
    from langchain_aws.chat_models import ChatBedrock, ChatBedrockConverse
    from langchain_aws.chat_models import bedrock as _bedrock_module
    from langchain_aws.chat_models.anthropic import ChatAnthropicBedrock

    from uipath_langchain_client.clients.bedrock.utils import WrappedBotoClient

    _original_format_data_content_block = _bedrock_module._format_data_content_block

    def _patched_format_data_content_block(block: dict) -> dict:
        """Extended version that also handles file/document blocks for Anthropic API."""
        if block["type"] == "file":
            if "base64" not in block and block.get("source_type") != "base64":
                raise ValueError("File data only supported through in-line base64 format.")
            if "mime_type" not in block:
                raise ValueError("mime_type key is required for base64 data.")
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": block["mime_type"],
                    "data": block.get("base64") or block.get("data", ""),
                },
            }
        return _original_format_data_content_block(block)

    _bedrock_module._format_data_content_block = _patched_format_data_content_block

except ImportError as e:
    raise ImportError(
        "The 'bedrock' extra is required to use UiPathBedrockChatModel and UiPathBedrockChatModelConverse. "
        "Install it with: uv add uipath-langchain-client[bedrock]"
    ) from e


def _setup_model_id(values: Any) -> Any:
    if isinstance(values, dict) and "model_id" not in values:
        model = values.get("model") or values.get("model_name")
        if model:
            values = {**values, "model_id": model}
    return values


_CONVERSE_PASSTHROUGH_KEYS = frozenset({"output_config"})


def _partition_converse_settings(
    model_settings: Mapping[str, Any],
    native_fields: Container[str],
    disabled_params: Container[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split model settings into (direct, passthrough) for ChatBedrockConverse.

    Converse only takes provider params via additional_model_request_fields, not
    model_kwargs. Keys that aren't native fields go to passthrough; so does output_config
    (a field, but it must be nested anyway). Real fields are set directly, disabled
    dropped. Returns (setattr these, merge these into additional_model_request_fields).
    """
    direct: dict[str, Any] = {}
    passthrough: dict[str, Any] = {}
    for key, value in model_settings.items():
        if key in disabled_params:
            continue
        if key in native_fields and key not in _CONVERSE_PASSTHROUGH_KEYS:
            direct[key] = value
        else:
            passthrough[key] = value
    return direct, passthrough


class UiPathChatBedrockConverse(UiPathBaseChatModel, ChatBedrockConverse):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.AWSBEDROCK,
        api_flavor=ApiFlavor.CONVERSE,
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model_id: str = "PLACEHOLDER"
    client: Any = WrappedBotoClient()
    bedrock_client: Any = WrappedBotoClient()

    @model_validator(mode="before")
    @classmethod
    def setup_model_id(cls, values: Any) -> Any:
        return _setup_model_id(values)

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = WrappedBotoClient(self.uipath_sync_client)
        return self

    def _apply_model_settings(self) -> None:
        if not self.model_settings:
            return
        direct, passthrough = _partition_converse_settings(
            self.model_settings,
            native_fields=type(self).model_fields,
            disabled_params=self.disabled_params or {},
        )
        for key, value in direct.items():
            self._assign_validated(key, value)
        if passthrough:
            self.additional_model_request_fields = {
                **(self.additional_model_request_fields or {}),
                **passthrough,
            }


class UiPathChatBedrock(UiPathBaseChatModel, ChatBedrock):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.AWSBEDROCK,
        api_flavor=ApiFlavor.INVOKE,
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model_id: str = "PLACEHOLDER"
    client: Any = WrappedBotoClient()
    bedrock_client: Any = WrappedBotoClient()

    @model_validator(mode="before")
    @classmethod
    def setup_model_id(cls, values: Any) -> Any:
        return _setup_model_id(values)

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = WrappedBotoClient(self.uipath_sync_client)
        return self

    @property
    def _as_converse(self) -> UiPathChatBedrockConverse:
        raise NotImplementedError("You must instantiate the converse client directly")


class UiPathChatAnthropicBedrock(UiPathBaseChatModel, ChatAnthropicBedrock):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.AWSBEDROCK,
        api_flavor=ApiFlavor.INVOKE,
        freeze_base_url=True,
    )

    # Override fields to avoid typing issues and fix stuff
    stop_sequences: list[str] | None = Field(default=None, alias="stop")
    model: str = Field(default="", alias="model_name")
    default_request_timeout: float | None = None

    @cached_property
    def _client(self) -> AnthropicBedrock:
        return AnthropicBedrock(
            aws_access_key="PLACEHOLDER",
            aws_secret_key="PLACEHOLDER",
            aws_region="PLACEHOLDER",
            base_url=str(self.uipath_sync_client.base_url),
            default_headers=dict(self.uipath_sync_client.headers),
            max_retries=0,  # handled by the UiPathBaseChatModel
            http_client=self.uipath_sync_client,
        )

    @cached_property
    def _async_client(self) -> AsyncAnthropicBedrock:
        return AsyncAnthropicBedrock(
            aws_access_key="PLACEHOLDER",
            aws_secret_key="PLACEHOLDER",
            aws_region="PLACEHOLDER",
            base_url=str(self.uipath_async_client.base_url),
            default_headers=dict(self.uipath_async_client.headers),
            max_retries=0,  # handled by the UiPathBaseChatModel
            http_client=self.uipath_async_client,
        )
