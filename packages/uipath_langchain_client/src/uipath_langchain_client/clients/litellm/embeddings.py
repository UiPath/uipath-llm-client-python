from typing import Any, Self

from pydantic import Field, model_validator

from uipath_langchain_client.base_client import UiPathBaseEmbeddings

try:
    from langchain_litellm import LiteLLMEmbeddings
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'litellm' extra is required to use UiPathLiteLLMEmbeddings. "
        "Install it with: uv add uipath-langchain-client[litellm]"
    ) from e


class UiPathLiteLLMEmbeddings(UiPathBaseEmbeddings, LiteLLMEmbeddings):  # type: ignore[override]
    """LiteLLM embeddings routed through UiPath LLM Gateway.

    Combines UiPath's authentication and routing with LiteLLM's
    multi-provider embeddings interface. The api_config must be provided
    by the caller.
    """

    # Override LiteLLMEmbeddings' model field to align with UiPathBaseLLMClient.model_name
    model: str = Field(default="", alias="model_name")

    _uipath_openai_client: OpenAI | None = None
    _uipath_async_openai_client: AsyncOpenAI | None = None

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        base_url = str(self.uipath_sync_client.base_url).rstrip("/")

        self._uipath_openai_client = OpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=self.uipath_sync_client,
            base_url=base_url,
        )
        self._uipath_async_openai_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,
            http_client=self.uipath_async_client,
            base_url=base_url,
        )
        return self

    def _embedding_with_retry(self, **kwargs: Any) -> Any:
        """Override to inject UiPath OpenAI client into litellm embedding calls."""
        import litellm
        from langchain_litellm.embeddings.litellm import _create_retry_decorator

        kwargs["client"] = self._uipath_openai_client
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        def _embed() -> Any:
            return litellm.embedding(**kwargs)

        return _embed()

    async def _aembedding_with_retry(self, **kwargs: Any) -> Any:
        """Override to inject UiPath async OpenAI client into litellm embedding calls."""
        import litellm
        from langchain_litellm.embeddings.litellm import _create_retry_decorator

        kwargs["client"] = self._uipath_async_openai_client
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        async def _aembed() -> Any:
            return await litellm.aembedding(**kwargs)

        return await _aembed()
