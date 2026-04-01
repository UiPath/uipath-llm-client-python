from typing import Any, Optional, Self

from pydantic import Field, model_validator

from uipath_langchain_client.base_client import UiPathBaseChatModel

try:
    from langchain_litellm import ChatLiteLLM
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'litellm' extra is required to use UiPathChatLiteLLM. "
        "Install it with: uv add uipath-langchain-client[litellm]"
    ) from e

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)


class UiPathChatLiteLLM(UiPathBaseChatModel, ChatLiteLLM):  # type: ignore[override]
    """LiteLLM chat model routed through UiPath LLM Gateway.

    Combines UiPath's authentication and routing with LiteLLM's
    multi-provider chat interface. The api_config must be provided
    by the caller — provider routing is not auto-detected from model names.
    """

    # Override ChatLiteLLM's model field to align with UiPathBaseLLMClient.model_name (alias="model")
    model: str = Field(default="", alias="model_name")

    _uipath_openai_client: Optional[OpenAI] = None
    _uipath_async_openai_client: Optional[AsyncOpenAI] = None

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

    def completion_with_retry(
        self, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any
    ) -> Any:
        """Override to inject UiPath OpenAI client into litellm calls."""
        from langchain_litellm.chat_models.litellm import _create_retry_decorator

        kwargs["client"] = self._uipath_openai_client
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.completion(**kwargs)

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any
    ) -> Any:
        """Override to inject UiPath async OpenAI client into litellm calls."""
        from langchain_litellm.chat_models.litellm import _create_retry_decorator

        kwargs["client"] = self._uipath_async_openai_client
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            return await self.client.acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)
