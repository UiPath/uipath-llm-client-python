"""Tests for the OpenAI Decisions client."""

import json
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from uipath.llm_client.clients.decisions import UiPathDecisionsClient
from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings import LLMGatewaySettings, PlatformSettings
from uipath.llm_client.utils.exceptions import UiPathAPIError

MODULE = "uipath.llm_client.clients.decisions.client"
MODEL = "gpt-6-luna"

QUESTIONS = [
    {
        "type": "choice",
        "name": "department",
        "instructions": "Which department should handle this complaint?",
        "choices": [
            {"value": "billing", "description": "Payments, invoices, and refunds."},
            {"value": "technical"},
        ],
    },
    {
        "type": "predicate",
        "name": "is_urgent",
        "instructions": "The customer needs an answer today.",
    },
]
ANSWER = {
    "answers": [
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "confidence": 0.93,
            "probabilities": [
                {"value": "billing", "probability": 0.93},
                {"value": "technical", "probability": 0.07},
            ],
        },
        {"type": "predicate", "name": "is_urgent", "probability": 0.82},
    ]
}


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
            return httpx.Response(state["status"], json={"error": {"message": "bad"}})
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


class TestDecisionsClient:
    def test_calls_gateway_raw_vendor_decisions_endpoint(
        self,
        gateway_settings: LLMGatewaySettings,
        mock_transport: Callable[[int], None],
        requests: list[httpx.Request],
    ) -> None:
        client = UiPathDecisionsClient(model_name=MODEL, client_settings=gateway_settings)

        result = client.create("I was charged twice for my order.", QUESTIONS)

        assert result == ANSWER
        (request,) = requests
        assert str(request.url) == (
            "https://cloud.uipath.com/test-org-id/test-tenant-id/"
            "llmgateway_/api/raw/vendor/openai/model/gpt-6-luna/decisions"
        )
        assert "X-UiPath-LlmGateway-ApiFlavor" not in request.headers
        assert json.loads(request.content) == {
            "model": "gpt-6-luna",
            "input": "I was charged twice for my order.",
            "questions": QUESTIONS,
        }

    def test_calls_agenthub_raw_vendor_decisions_endpoint(
        self,
        platform_env_vars: dict[str, str],
        mock_platform_auth: None,
        mock_transport: Callable[[int], None],
        requests: list[httpx.Request],
    ) -> None:
        with patch.dict("os.environ", {**platform_env_vars, "UIPATH_LLM_SERVICE": "agenthub"}):
            settings = PlatformSettings()
            client = UiPathDecisionsClient(model_name=MODEL, client_settings=settings)
            client.create("text", QUESTIONS)

        (request,) = requests
        assert str(request.url) == (
            "https://cloud.uipath.com/org/tenant/"
            "agenthub_/llm/raw/vendor/openai/model/gpt-6-luna/decisions"
        )

    def test_sends_messages_with_images_as_a_list(
        self,
        gateway_settings: LLMGatewaySettings,
        mock_transport: Callable[[int], None],
        requests: list[httpx.Request],
    ) -> None:
        messages = (
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Is this a receipt?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AA"},
                ],
            },
        )
        client = UiPathDecisionsClient(model_name=MODEL, client_settings=gateway_settings)

        client.create(messages, QUESTIONS)

        (request,) = requests
        assert json.loads(request.content)["input"] == list(messages)

    async def test_async_call(
        self,
        gateway_settings: LLMGatewaySettings,
        mock_transport: Callable[[int], None],
        requests: list[httpx.Request],
    ) -> None:
        client = UiPathDecisionsClient(model_name=MODEL, client_settings=gateway_settings)

        result = await client.acreate("text", QUESTIONS)

        assert result == ANSWER
        assert len(requests) == 1

    def test_http_error_raises_uipath_error(
        self, gateway_settings: LLMGatewaySettings, mock_transport: Callable[[int], None]
    ) -> None:
        mock_transport(400)
        client = UiPathDecisionsClient(
            model_name=MODEL, client_settings=gateway_settings, max_retries=0
        )

        with pytest.raises(UiPathAPIError):
            client.create("text", QUESTIONS)

    def test_model_name_is_required(self, gateway_settings: LLMGatewaySettings) -> None:
        with pytest.raises(TypeError, match="model_name"):
            UiPathDecisionsClient(client_settings=gateway_settings)  # type: ignore[call-arg]

    def test_empty_questions_rejected(
        self, gateway_settings: LLMGatewaySettings, mock_transport: Callable[[int], None]
    ) -> None:
        client = UiPathDecisionsClient(model_name=MODEL, client_settings=gateway_settings)

        with pytest.raises(ValueError):
            client.create("text", [])

    def test_uses_default_settings_when_none_given(
        self, mock_transport: Callable[[int], None], llmgw_env_vars: dict[str, str]
    ) -> None:
        with patch.dict("os.environ", llmgw_env_vars):
            settings = LLMGatewaySettings()
        with patch(f"{MODULE}.get_default_client_settings", return_value=settings) as factory:
            client = UiPathDecisionsClient(model_name=MODEL)

        factory.assert_called_once_with()
        assert client.model_name == MODEL
