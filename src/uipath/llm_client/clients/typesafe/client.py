import logging
from collections.abc import Mapping
from typing import Any

from httpx import Response

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import ApiType, RoutingMode
from uipath.llm_client.utils.retry import RetryConfig

JEV_DEFAULT_MODEL = "jev-latest"

# Raw vendor passthrough: .../raw/vendor/typesafe/model/{model}/completions
TYPESAFE_VENDOR_TYPE = "typesafe"
TYPESAFE_API_FLAVOR = "systemone"


def _build_api_config() -> UiPathAPIConfig:
    return UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=TYPESAFE_VENDOR_TYPE,
        api_flavor=TYPESAFE_API_FLAVOR,
        freeze_base_url=True,
    )


class UiPathJevClient:
    """Client for TypeSafe's Jev ``systemone`` endpoint.

    Calls the UiPath LLM Gateway raw vendor passthrough for vendor
    ``typesafe`` using ``client_settings`` (or the default settings). It uses
    the UiPath httpx clients, so retries (including 429/529), logging and
    UiPath exception mapping behave like the other vendor clients.

    Args:
        model_name: The Jev model name. Defaults to ``jev-latest``.
        client_settings: UiPath client settings. Defaults to the default settings.
        timeout: Client-side request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        default_headers: Additional headers to include in requests.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
    """

    def __init__(
        self,
        *,
        model_name: str = JEV_DEFAULT_MODEL,
        client_settings: UiPathBaseSettings | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.model_name = model_name
        gateway: dict[str, Any] = {
            "model_name": model_name,
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_config": retry_config,
            "logger": logger,
            "client_settings": client_settings or get_default_client_settings(),
            "api_config": _build_api_config(),
            "headers": default_headers,
        }
        self._client = UiPathHttpxClient(**gateway)
        # The frozen base URL already is the full passthrough endpoint, so
        # requests post to "".
        self._async_client = UiPathHttpxAsyncClient(**gateway)

    def _build_body(
        self, state: str | Mapping[str, Any] | list[Any], questions: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("At least one question is required.")
        return {"state": state, "model": self.model_name, "questions": dict(questions)}

    @staticmethod
    def _parse(response: Response) -> dict[str, Any]:
        response.raise_for_status()
        return response.json()

    def system_one(
        self, state: str | Mapping[str, Any] | list[Any], questions: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Ask Jev typed questions about ``state``.

        Args:
            state: The input to classify (text, or a JSON object/array).
            questions: Question id to question definition, e.g.
                ``{"department": {"type": "choice", "instructions": "...",
                "criteria": {"billing": "Payment issues"}}}``.

        Returns:
            The decoded response: ``{"model", "answers", "usage"}``.

        Raises:
            ValueError: If ``questions`` is empty.
            UiPathAPIError: If the request fails.
        """
        body = self._build_body(state, questions)
        return self._parse(self._client.post("", json=body))

    async def asystem_one(
        self, state: str | Mapping[str, Any] | list[Any], questions: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Async version of :meth:`system_one`."""
        body = self._build_body(state, questions)
        return self._parse(await self._async_client.post("", json=body))

    def close(self) -> None:
        """Close the underlying sync HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        self._client.close()
        await self._async_client.aclose()
