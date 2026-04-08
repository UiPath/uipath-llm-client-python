# pyright: reportAttributeAccessIssue=false
"""UiPath LangChain chat model powered by LiteLLM.

Inherits ``UiPathBaseChatModel`` and ``ChatLiteLLM`` — UiPath handles
authentication, routing, retries; LiteLLM handles provider-specific formatting.

Example:
    >>> from uipath_langchain_client.clients.litellm import UiPathChatLiteLLM
    >>>
    >>> chat = UiPathChatLiteLLM(model="gpt-5.2-2025-12-11")
    >>> response = chat.invoke("Hello!")
    >>> print(response.content)
"""

from typing import Any, Dict

from pydantic import Field, model_validator
from typing_extensions import Self

from uipath.llm_client.clients.litellm import UiPathLiteLLM
from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_litellm import ChatLiteLLM
except ImportError as e:
    raise ImportError(
        "The 'litellm' extra is required to use UiPathChatLiteLLM. "
        "Install it with: uv add uipath-langchain-client[litellm]"
    ) from e


class UiPathChatLiteLLM(UiPathBaseChatModel, ChatLiteLLM):  # type: ignore[override]
    """LangChain chat model that routes through UiPath LLM Gateway via LiteLLM.

    Discovers the model from the UiPath backend and uses LiteLLM for
    provider-specific request formatting. Authentication, URL routing,
    and retries are handled by the UiPath httpx client.

    Args:
        model: The model name (e.g., "gpt-5.2-2025-12-11", "gemini-2.5-flash").
        settings: UiPath client settings. Defaults to environment-based settings.
        vendor_type: Filter/override vendor type from discovery.
        api_flavor: Override API flavor (e.g., ApiFlavor.RESPONSES, ApiFlavor.CONVERSE).
    """

    api_config: UiPathAPIConfig = Field(
        default_factory=lambda: UiPathAPIConfig(
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type=VendorType.OPENAI,  # placeholder — overridden by _setup_uipath
            api_type=ApiType.COMPLETIONS,
            freeze_base_url=True,
        ),
    )

    vendor_type: VendorType | str | None = Field(default=None, exclude=True)
    api_flavor: ApiFlavor | str | None = Field(default=None, exclude=True)

    # Internal core client
    _core: UiPathLiteLLM | None = None

    @model_validator(mode="after")
    def _setup_uipath(self) -> Self:
        """Create a UiPathLiteLLM core client and inject into ChatLiteLLM."""
        core = UiPathLiteLLM(
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
        self._core = core

        # Sync api_config from discovery for UiPathBaseLLMClient httpx clients
        self.api_config = UiPathAPIConfig(
            vendor_type=core._api_config.vendor_type,
            api_flavor=core._api_config.api_flavor,
        )

        # Inject into ChatLiteLLM's fields
        self.custom_llm_provider = core._custom_llm_provider
        self.api_key = "PLACEHOLDER"
        self.api_base = str(core._completion_client.client.base_url)
        self.model_name = core._litellm_model  # type: ignore[assignment]

        return self

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Inject UiPath HTTPHandler into every litellm call."""
        params = super()._client_params
        if self._core is not None:
            params["client"] = self._core._completion_client
            params["num_retries"] = 0
            params["max_retries"] = 0
            params.update(self._core._extra_litellm_kwargs)
        return params

    @property
    def _llm_type(self) -> str:
        return "uipath-litellm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "provider": self._core._custom_llm_provider if self._core else None,
            "api_base": self.api_base,
        }
