from typing import Self

from pydantic import model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_aws.embeddings import BedrockEmbeddings
    from uipath_langchain_client.clients.bedrock.utils import WrappedBotoClient
except ImportError as e:
    raise ImportError(
        "The 'aws' extra is required to use UiPathBedrockEmbeddings. "
        "Install it with: uv add uipath-langchain-client[aws]"
    ) from e


class UiPathBedrockEmbeddings(UiPathBaseLLMClient, BedrockEmbeddings):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="awsbedrock",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model_id: str = "PLACEHOLDER"
    region_name: str | None = "PLACEHOLDER"

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.model_id = self.model_name
        self.client = WrappedBotoClient(self.uipath_sync_client)
        return self
