"""Tests for the TypeSafe Jev client."""

import json
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from uipath.llm_client.clients.typesafe import UiPathJevClient
from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.utils.exceptions import UiPathAPIError

MODULE = "uipath.llm_client.clients.typesafe.client"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {"billing": "Payment issues", "technical": None},
    }
}
ANSWER = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.95, "technical": 0.05},
        }
    },
    "usage": {"input_tokens": 12, "output_tokens": 0},
}

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def mock_transport(requests: list[httpx.Request]) -> Iterator[Callable[[int], None]]:
    """Route both UiPath httpx clients through a MockTransport."""
    state: dict[str, int] = {"status": 200}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"detail": "bad"})
        return httpx.Response(200, json=ANSWER)

    def make_sync(**kwargs: Any) -> UiPathHttpxClient:
        return UiPathHttpxClient(**kwargs, transport=httpx.MockTransport(handler))

    def make_async(**kwargs: Any) -> UiPathHttpxAsyncClient:
        return UiPathHttpxAsyncClient(**kwargs, transport=httpx.MockTransport(handler))

    def set_status(status: int) -> None:
        state["status"] = status

    with (
        patch(f"{MODULE}.UiPathHttpxClient", side_effect=make_sync),
        patch(f"{MODULE}.UiPathHttpxAsyncClient", side_effect=make_async),
    ):
        yield set_status


@pytest.fixture
def gateway_settings(llmgw_env_vars: dict[str, str]) -> LLMGatewaySettings:
    with patch.dict("os.environ", llmgw_env_vars):
        return LLMGatewaySettings()


class TestLLMGatewayAccess:
    def test_calls_gateway_raw_vendor_endpoint(
        self,
        gateway_settings: LLMGatewaySettings,
        mock_transport: Callable[[int], None],
        requests: list[httpx.Request],
    ) -> None:
        client = UiPathJevClient(client_settings=gateway_settings)

        result = client.system_one("Payment failed", QUESTIONS)

        assert result == ANSWER
        (request,) = requests
        assert str(request.url) == (
            "https://cloud.uipath.com/test-org-id/test-tenant-id/"
            "llmgateway_/api/raw/vendor/typesafe/model/jev-latest/completions"
        )
        assert request.headers["X-UiPath-LlmGateway-RequestingProduct"] == "test-product"
        assert json.loads(request.content) == {
            "state": "Payment failed",
            "model": "jev-latest",
            "questions": QUESTIONS,
        }

    def test_model_name_in_url_and_body(
        self,
        gateway_settings: LLMGatewaySettings,
        mock_transport: Callable[[int], None],
        requests: list[httpx.Request],
    ) -> None:
        client = UiPathJevClient(model_name="jev-preview", client_settings=gateway_settings)
        client.system_one("text", QUESTIONS)

        (request,) = requests
        assert str(request.url).endswith("/raw/vendor/typesafe/model/jev-preview/completions")
        assert json.loads(request.content)["model"] == "jev-preview"

    async def test_async_call(
        self,
        gateway_settings: LLMGatewaySettings,
        mock_transport: Callable[[int], None],
        requests: list[httpx.Request],
    ) -> None:
        client = UiPathJevClient(client_settings=gateway_settings)

        result = await client.asystem_one("text", QUESTIONS)

        assert result == ANSWER
        assert len(requests) == 1

    def test_http_error_raises_uipath_error(
        self, gateway_settings: LLMGatewaySettings, mock_transport: Callable[[int], None]
    ) -> None:
        mock_transport(422)
        client = UiPathJevClient(client_settings=gateway_settings, max_retries=0)

        with pytest.raises(UiPathAPIError):
            client.system_one("text", QUESTIONS)

    def test_empty_questions_rejected(
        self, gateway_settings: LLMGatewaySettings, mock_transport: Callable[[int], None]
    ) -> None:
        client = UiPathJevClient(client_settings=gateway_settings)

        with pytest.raises(ValueError):
            client.system_one("text", {})

    def test_uses_default_settings_when_none_given(
        self, mock_transport: Callable[[int], None], llmgw_env_vars: dict[str, str]
    ) -> None:
        with patch.dict("os.environ", llmgw_env_vars):
            settings = LLMGatewaySettings()
        with patch(f"{MODULE}.get_default_client_settings", return_value=settings) as factory:
            client = UiPathJevClient()

        factory.assert_called_once_with()
        assert client.model_name == "jev-latest"
