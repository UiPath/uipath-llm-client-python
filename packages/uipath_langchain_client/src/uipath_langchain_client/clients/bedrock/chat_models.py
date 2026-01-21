from typing import Any, Self

from pydantic import model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_aws.chat_models import ChatBedrock, ChatBedrockConverse
    from uipath_langchain_client.clients.bedrock.utils import WrappedBotoClient
except ImportError as e:
    raise ImportError(
        "The 'aws' extra is required to use UiPathBedrockChatModel and UiPathBedrockChatModelConverse. "
        "Install it with: uv add uipath-langchain-client[aws]"
    ) from e


class UiPathChatBedrockConverse(UiPathBaseLLMClient, ChatBedrockConverse):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="awsbedrock",
        api_flavor="converse",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model_id: str = "PLACEHOLDER"
    client: Any = WrappedBotoClient()
    bedrock_client: Any = WrappedBotoClient()

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.model_id = self.model_name
        self.client = WrappedBotoClient(self.uipath_sync_client)
        return self


class UiPathChatBedrock(UiPathBaseLLMClient, ChatBedrock):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="awsbedrock",
        api_flavor="invoke",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model_id: str = "PLACEHOLDER"
    client: Any = WrappedBotoClient()
    bedrock_client: Any = WrappedBotoClient()

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.model_id = self.model_name
        self.client = WrappedBotoClient(self.uipath_sync_client)
        return self

    @property
    def _as_converse(self) -> UiPathChatBedrockConverse:
        return UiPathChatBedrockConverse(
            model=self.model_name,
            client_settings=self.client_settings,
        )
