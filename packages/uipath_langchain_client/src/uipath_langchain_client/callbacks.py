"""LangChain callbacks for dynamic per-request header injection."""

from abc import abstractmethod
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from uipath.llm_client.utils.headers import (
    set_dynamic_request_headers,
)


class UiPathDynamicHeadersCallback(BaseCallbackHandler):
    """Base callback for injecting dynamic headers into each LLM gateway request.

    Extend this class and implement ``get_headers()`` to return the headers to
    inject. The headers are stored in a ContextVar before each LLM call and read
    by the httpx client's ``send()`` method, so they flow transparently through
    the call stack regardless of which vendor SDK is in use.

    Example (OTEL trace propagation)::

        from opentelemetry import trace, propagate

        class OtelHeadersCallback(UiPathDynamicHeadersCallback):
            def get_headers(self) -> dict[str, str]:
                carrier: dict[str, str] = {}
                propagate.inject(carrier)
                return carrier

        chat = get_chat_model(model_name="gpt-4o", client_settings=settings)
        response = chat.invoke("Hello!", config={"callbacks": [OtelHeadersCallback()]})
    """

    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Return headers to inject into the next LLM gateway request."""
        ...

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        set_dynamic_request_headers(self.get_headers())

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        set_dynamic_request_headers(self.get_headers())

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        set_dynamic_request_headers({})

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        set_dynamic_request_headers({})
