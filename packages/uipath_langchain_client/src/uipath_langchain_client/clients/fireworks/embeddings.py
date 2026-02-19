from typing import Self

from pydantic import model_validator

from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_fireworks.embeddings import FireworksEmbeddings
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'fireworks' extra is required to use UiPathFireworksEmbeddings. "
        'Install it with: uv add "uipath-langchain-client[fireworks]"'
    ) from e


class UiPathFireworksEmbeddings(UiPathBaseLLMClient, FireworksEmbeddings):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="openai",
        api_flavor="chat-completions",
        api_version="2025-03-01-preview",
        freeze_base_url=True,
    )

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        self.client = OpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        )
        self.async_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        )
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return [
            i.embedding for i in self.client.embeddings.create(input=texts, model=self.model).data
        ]

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs asynchronously."""
        return [
            i.embedding
            for i in (await self.async_client.embeddings.create(input=texts, model=self.model)).data
        ]

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text asynchronously."""
        return (await self.aembed_documents([text]))[0]
