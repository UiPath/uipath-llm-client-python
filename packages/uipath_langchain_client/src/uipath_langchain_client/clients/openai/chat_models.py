from collections.abc import Awaitable, Callable
from typing import Any, Self

from httpx import Request
from langchain_core.outputs import ChatResult
from pydantic import Field, SecretStr, model_validator

from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.clients.openai.tool_call_extras import (
    OpenAIToolCallExtrasMixin,
)
from uipath_langchain_client.clients.openai.utils import fix_url_and_api_flavor_header
from uipath_langchain_client.settings import (
    ApiFlavor,
    ApiType,
    RoutingMode,
    UiPathAPIConfig,
    VendorType,
)

try:
    from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required to use UiPathChatOpenAI and UiPathAzureChatOpenAI. "
        "Install it with: uv add uipath-langchain-client[openai]"
    ) from e


class UiPathChatOpenAI(UiPathBaseChatModel, OpenAIToolCallExtrasMixin, ChatOpenAI):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.OPENAI,
        api_version="2025-04-01-preview",
        freeze_base_url=False,
    )
    api_flavor: ApiFlavor | str | None = None

    # Override fields to avoid errors when instantiating the class
    openai_api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] = Field(
        alias="api_key", default=SecretStr("PLACEHOLDER")
    )

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        if self.api_flavor is not None:
            self.api_config.api_flavor = self.api_flavor
            # Lock LangChain's use_responses_api to match the discovered flavor.
            # Without this, features like reasoning={} or certain model names
            # silently switch LangChain to the Responses API, which would fail
            # if the model only supports chat-completions (or vice versa).
            if self.api_flavor == ApiFlavor.CHAT_COMPLETIONS:
                self.use_responses_api = False
            elif self.api_flavor == ApiFlavor.RESPONSES:
                self.use_responses_api = True
        base_url = str(self.uipath_sync_client.base_url).rstrip("/")
        locked_flavor = str(self.api_config.api_flavor) if self.api_config.api_flavor else None

        def on_request(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        async def on_request_async(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        self.uipath_sync_client.event_hooks["request"].append(on_request)
        self.uipath_async_client.event_hooks["request"].append(on_request_async)

        self.root_client = OpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
            base_url=base_url,
        )
        self.root_async_client = AsyncOpenAI(
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_async_client,
            base_url=base_url,
        )
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        return self


class UiPathAzureChatOpenAI(UiPathBaseChatModel, AzureChatOpenAI):  # type: ignore[override]
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.OPENAI,
        api_version="2025-04-01-preview",
        freeze_base_url=False,
    )
    api_flavor: ApiFlavor | str | None = None

    # Override fields to avoid errors when instantiating the class
    azure_endpoint: str | None = Field(default="PLACEHOLDER")
    openai_api_version: str | None = Field(default="PLACEHOLDER", alias="api_version")
    openai_api_key: SecretStr | None = Field(default=SecretStr("PLACEHOLDER"), alias="api_key")

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        if self.api_flavor is not None:
            self.api_config.api_flavor = self.api_flavor
            if self.api_flavor == ApiFlavor.CHAT_COMPLETIONS:
                self.use_responses_api = False
            elif self.api_flavor == ApiFlavor.RESPONSES:
                self.use_responses_api = True
        base_url = str(self.uipath_sync_client.base_url).rstrip("/")
        locked_flavor = str(self.api_config.api_flavor) if self.api_config.api_flavor else None

        def on_request(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        async def on_request_async(request: Request) -> None:
            fix_url_and_api_flavor_header(base_url, request, api_flavor=locked_flavor)

        self.uipath_sync_client.event_hooks["request"].append(on_request)
        self.uipath_async_client.event_hooks["request"].append(on_request_async)

        self.root_client = AzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_sync_client,
        )
        self.root_async_client = AsyncAzureOpenAI(
            azure_endpoint="PLACEHOLDER",
            api_version="PLACEHOLDER",
            api_key="PLACEHOLDER",
            max_retries=0,  # handled by the UiPath client
            http_client=self.uipath_async_client,
        )
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        return self

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Return a content-filtered choice as a message instead of raising.

        langchain-openai's ``AzureChatOpenAI`` raises a bare ``ValueError`` for a
        choice whose ``finish_reason`` is ``"content_filter"``. That error carries
        no HTTP response (the call returned 200), so it would reach callers as an
        uncategorised ``UiPathError``. ``ChatOpenAI`` and the Responses API path
        return the message instead, with the finish reason in
        ``response_metadata``; this does the same, so callers check the stop
        reason one way on every OpenAI route.
        """
        is_dict = isinstance(response, dict)
        choices = (response.get("choices") if is_dict else getattr(response, "choices", None)) or []
        filtered = {
            i
            for i, choice in enumerate(choices)
            if (choice.get("finish_reason") if is_dict else getattr(choice, "finish_reason", None))
            == "content_filter"
        }
        if not filtered:
            return super()._create_chat_result(response, generation_info)

        # Hide the finish reason from AzureChatOpenAI so it does not raise, keeping
        # the rest of its processing (model name, content filter results, and the
        # typed response's parsed/refusal fields), then put the finish reason back
        # on the filtered generations.
        if is_dict:
            unfiltered: Any = {
                **response,
                "choices": [
                    {**choice, "finish_reason": None} if i in filtered else choice
                    for i, choice in enumerate(choices)
                ],
            }
        else:
            unfiltered = response.model_copy(
                update={
                    "choices": [
                        choice.model_copy(update={"finish_reason": None})
                        if i in filtered
                        else choice
                        for i, choice in enumerate(choices)
                    ]
                }
            )
        result = super()._create_chat_result(unfiltered, generation_info)
        for i in filtered:
            generation = result.generations[i]
            generation.generation_info = {
                **(generation.generation_info or {}),
                "finish_reason": "content_filter",
            }
            generation.message.response_metadata["finish_reason"] = "content_filter"
        return result
