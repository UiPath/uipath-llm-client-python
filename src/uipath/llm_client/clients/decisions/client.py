import logging
from collections.abc import Mapping, Sequence
from typing import Any

from httpx import Response

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.llm_client.settings.constants import ApiType, RoutingMode, VendorType
from uipath.llm_client.utils.retry import RetryConfig

# Raw vendor passthrough: .../raw/vendor/openai/model/{model}/decisions (POST /v1/decisions).


def _build_api_config() -> UiPathAPIConfig:
    return UiPathAPIConfig(
        api_type=ApiType.DECISIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.OPENAI,
        freeze_base_url=True,
    )


class UiPathDecisionsClient:
    """Client for OpenAI's Decisions API (``POST /v1/decisions``).

    Calls the UiPath LLM Gateway raw vendor decisions passthrough for vendor
    ``openai`` using ``client_settings`` (or the default settings). It uses the
    UiPath httpx clients, so retries (including 429/529), logging and UiPath
    exception mapping behave like the other vendor clients.

    Args:
        model_name: The Decisions model, e.g. ``gpt-6-luna``. Required.
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
        model_name: str,
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
        self, input: str | Sequence[Any], questions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("At least one question is required.")
        return {
            "model": self.model_name,
            "input": input if isinstance(input, str) else list(input),
            "questions": [dict(question) for question in questions],
        }

    @staticmethod
    def _parse(response: Response) -> dict[str, Any]:
        response.raise_for_status()
        return response.json()

    def create(
        self, input: str | Sequence[Any], questions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Ask typed questions about ``input``.

        Args:
            input: The shared evidence: a text, or user messages whose content
                combines ``input_text`` and ``input_image`` parts.
            questions: The questions, each with a ``type`` (``choice``, ``score``
                or ``predicate``), a unique ``name`` and ``instructions``, plus
                ``choices`` (choice) or ``levels`` (score).

        Returns:
            The decoded response, with one entry per question in ``answers``.

        Raises:
            ValueError: If ``questions`` is empty.
            UiPathAPIError: If the request fails.
        """
        body = self._build_body(input, questions)
        return self._parse(self._client.post("", json=body))

    async def acreate(
        self, input: str | Sequence[Any], questions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Async version of :meth:`create`."""
        body = self._build_body(input, questions)
        return self._parse(await self._async_client.post("", json=body))

    def close(self) -> None:
        """Close the underlying sync HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        self._client.close()
        await self._async_client.aclose()
