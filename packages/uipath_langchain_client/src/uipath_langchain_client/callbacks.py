"""LangChain callbacks for dynamic per-request header injection."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from uipath.llm_client.utils.headers import set_dynamic_request_headers


class UiPathDynamicHeadersCallback(BaseCallbackHandler, ABC):
    """Base callback for injecting dynamic headers into each LLM gateway request.

    Extend this class and implement ``get_headers()`` to return the headers to
    inject. ``run_inline = True`` ensures ``on_chat_model_start`` is called
    directly in the caller's coroutine (not via ``asyncio.gather``), so the
    ContextVar mutation is visible when ``httpx.send()`` fires.

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

    run_inline: bool = True  # dispatch in the caller's coroutine, not via asyncio.gather

    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Return headers to inject into the next LLM gateway request."""
        ...

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        set_dynamic_request_headers(self.get_headers())

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        set_dynamic_request_headers(self.get_headers())

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        set_dynamic_request_headers({})

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        set_dynamic_request_headers({})
