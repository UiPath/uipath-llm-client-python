from langchain_core.embeddings import Embeddings
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig


class UiPathNormalizedEmbeddings(UiPathBaseLLMClient, Embeddings):
    """LangChain embeddings using the UiPath's normalized embeddings API.

    Provides a consistent interface for generating text embeddings across all
    embedding providers supported by UiPath AgentHub and LLM Gateway.
    """

    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="embeddings",
        client_type="normalized",
        freeze_base_url=True,
    )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.uipath_request(request_body={"input": texts})
        return [r["embedding"] for r in response.json()["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self.uipath_arequest(request_body={"input": texts})
        return [r["embedding"] for r in response.json()["data"]]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]
