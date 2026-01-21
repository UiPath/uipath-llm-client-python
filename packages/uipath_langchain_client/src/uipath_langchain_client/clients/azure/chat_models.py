from typing import Self

from pydantic import model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.aio import ChatCompletionsClient as ChatCompletionsClientAsync
    from azure.core.credentials import AzureKeyCredential
except ImportError as e:
    raise ImportError(
        "The 'azure' extra is required to use UiPathAzureAIChatCompletionsModel. "
        "Install it with: uv add uipath-langchain-client[azure]"
    ) from e


class UiPathAzureAIChatCompletionsModel(UiPathBaseLLMClient, AzureAIChatCompletionsModel):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="azure",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    endpoint: str | None = "PLACEHOLDER"

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        # TODO: finish implementation once we have a proper model in UiPath API
        self._client = ChatCompletionsClient(
            endpoint="PLACEHOLDER",
            credential=AzureKeyCredential("PLACEHOLDER"),
            model=self.model_name,
            **self.client_kwargs,
        )
        self._async_client = ChatCompletionsClientAsync(
            endpoint="PLACEHOLDER",
            credential=AzureKeyCredential("PLACEHOLDER"),
            model=self.model_name,
            **self.client_kwargs,
        )
        return self
