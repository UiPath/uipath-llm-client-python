import re
from functools import cached_property
from typing import Any, Self

from langchain_core.language_models import LanguageModelInput
from pydantic import Field, model_validator
from typing_extensions import override

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


# Sampling parameters that Claude Opus 4+ (reasoning models) do not support.
# The API returns 400 if any of these are present in the request payload.
_CLAUDE_OPUS_4_UNSUPPORTED_PARAMS: frozenset[str] = frozenset({"temperature", "top_k", "top_p"})


def _is_claude_opus_4_or_above(model_name: str) -> bool:
    """Return True for Claude Opus 4+ models that reject sampling parameters."""
    return bool(re.search(r"claude-opus-4", model_name, re.IGNORECASE))


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
        if isinstance(values, dict) and "model_id" not in values:
            model = values.get("model") or values.get("model_name")
            if model:
                values = {**values, "model_id": model}
        return values

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = WrappedBotoClient(self.uipath_sync_client)
        return self


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
        if isinstance(values, dict) and "model_id" not in values:
            model = values.get("model") or values.get("model_name")
            if model:
                values = {**values, "model_id": model}
        return values

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

    @override
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if _is_claude_opus_4_or_above(self.model):
            for param in _CLAUDE_OPUS_4_UNSUPPORTED_PARAMS:
                payload.pop(param, None)
        return payload
