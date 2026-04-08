"""UiPath LangChain embeddings model powered by LiteLLM.

Inherits ``UiPathBaseEmbeddings`` for UiPath httpx transport and
delegates to the core ``UiPathLiteLLM`` client for embeddings.

Example:
    >>> from uipath_langchain_client.clients.litellm import UiPathLiteLLMEmbeddings
    >>>
    >>> embeddings = UiPathLiteLLMEmbeddings(model="text-embedding-3-large")
    >>> vectors = embeddings.embed_documents(["Hello world"])
"""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from uipath.llm_client.clients.litellm import UiPathLiteLLM
from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType
from uipath_langchain_client.base_client import UiPathBaseEmbeddings
from uipath_langchain_client.settings import UiPathAPIConfig


class UiPathLiteLLMEmbeddings(UiPathBaseEmbeddings):
    """LangChain embeddings model that routes through UiPath LLM Gateway via LiteLLM.

    Args:
        model: The embedding model name (e.g., "text-embedding-3-large").
        settings: UiPath client settings. Defaults to environment-based settings.
        vendor_type: Filter/override vendor type from discovery.
        api_flavor: Override API flavor from discovery.
    """

    api_config: UiPathAPIConfig = Field(
        default_factory=lambda: UiPathAPIConfig(
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type=VendorType.OPENAI,  # placeholder — overridden by _setup_core
            api_type=ApiType.EMBEDDINGS,
            freeze_base_url=True,
        ),
    )

    vendor_type: VendorType | str | None = Field(default=None, exclude=True)
    api_flavor: ApiFlavor | str | None = Field(default=None, exclude=True)

    _core: UiPathLiteLLM | None = None

    @model_validator(mode="after")
    def _setup_core(self) -> Self:
        """Create the core UiPathLiteLLM client and sync api_config from discovery."""
        self._core = UiPathLiteLLM(
            model_name=self.model_name,
            byo_connection_id=self.byo_connection_id,
            client_settings=self.client_settings,
            vendor_type=self.vendor_type,
            api_flavor=self.api_flavor,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            captured_headers=self.captured_headers,
            retry_config=self.retry_config,
            logger=self.logger,
        )
        self.api_config = UiPathAPIConfig(
            vendor_type=self._core._api_config.vendor_type,
            api_flavor=self._core._api_config.api_flavor,
        )
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        assert self._core is not None
        response = self._core.embedding(input=texts)
        return [item["embedding"] for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        assert self._core is not None
        response = self._core.embedding(input=[text])
        return response.data[0]["embedding"]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        assert self._core is not None
        response = await self._core.aembedding(input=texts)
        return [item["embedding"] for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        assert self._core is not None
        response = await self._core.aembedding(input=[text])
        return response.data[0]["embedding"]
