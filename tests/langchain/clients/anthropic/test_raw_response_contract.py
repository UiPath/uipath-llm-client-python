"""Unit tests for the langchain-anthropic >= 1.7.0 raw-response contract.

langchain-anthropic 1.7.0 changed `_create`/`_acreate` to return a raw-response
wrapper and calls `.parse()` on it in `_generate`/`_agenerate`/`_stream`/`_astream`.
These tests mock the HTTP transport and verify that the UiPath overrides (which
route through the UiPath-constructed SDK clients) satisfy that contract for
invoke, streaming, tool calling, structured output, and the `betas` payload
branch — i.e. no `AttributeError: ... 'parse'` and responses parse into
`AIMessage`s.
"""

import json
from typing import Any

import httpx
import pytest
from anthropic import Anthropic, AsyncAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic

from uipath.llm_client.settings import UiPathBaseSettings

MODEL_NAME = "claude-sonnet-4-6"

TEXT_MESSAGE: dict[str, Any] = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": MODEL_NAME,
    "content": [{"type": "text", "text": "Hello!"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _tool_use_message(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        **TEXT_MESSAGE,
        "content": [{"type": "tool_use", "id": "toolu_test", "name": name, "input": args}],
        "stop_reason": "tool_use",
    }


_STREAM_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            **TEXT_MESSAGE,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo!"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 5},
    },
    {"type": "message_stop"},
]

SSE_BODY = "".join(
    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in _STREAM_EVENTS
)


class _MockedTransport:
    """httpx.MockTransport handler that records requests and serves canned responses.

    Streaming requests (``"stream": true`` in the body) get an SSE stream; other
    requests get ``message`` as JSON. When the request declares tools, the
    response is a ``tool_use`` block invoking the first tool with ``tool_args``.
    """

    def __init__(self, tool_args: dict[str, Any] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.tool_args = tool_args or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
            )
        if body.get("tools"):
            message = _tool_use_message(body["tools"][0]["name"], self.tool_args)
        else:
            message = TEXT_MESSAGE
        return httpx.Response(
            200,
            json=message,
            headers={"X-LangSmith-Gateway-Metadata": json.dumps({"provider": "anthropic"})},
        )


def _mocked_chat(
    client_settings: UiPathBaseSettings,
    transport_handler: _MockedTransport,
    **model_kwargs: Any,
) -> UiPathChatAnthropic:
    """Build a UiPathChatAnthropic whose SDK clients run over a mock transport."""
    params: dict[str, Any] = {
        "model": MODEL_NAME,
        "settings": client_settings,
        "model_details": {},
        "max_tokens": 1024,
        **model_kwargs,
    }
    chat = UiPathChatAnthropic(**params)
    transport = httpx.MockTransport(transport_handler)
    # Shadow the cached properties with SDK clients over the mock transport,
    # preserving the shape _create/_acreate rely on.
    object.__setattr__(
        chat,
        "_anthropic_client",
        Anthropic(
            api_key="PLACEHOLDER",
            base_url="https://mock.uipath.test",
            max_retries=0,
            http_client=httpx.Client(transport=transport),
        ),
    )
    object.__setattr__(
        chat,
        "_async_anthropic_client",
        AsyncAnthropic(
            api_key="PLACEHOLDER",
            base_url="https://mock.uipath.test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        ),
    )
    return chat


@pytest.fixture
def transport_handler() -> _MockedTransport:
    return _MockedTransport(tool_args={"city": "Paris"})


class Weather(BaseModel):
    """Weather report for a city."""

    city: str


class TestRawResponseContract:
    def test_invoke(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        message = chat.invoke("Say hello")
        assert isinstance(message, AIMessage)
        assert message.text == "Hello!"
        assert transport_handler.requests[-1].url.path == "/v1/messages"
        assert "beta" not in str(transport_handler.requests[-1].url.query)

    async def test_ainvoke(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        message = await chat.ainvoke("Say hello")
        assert isinstance(message, AIMessage)
        assert message.text == "Hello!"

    def test_stream(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        chunks = list(chat.stream("Say hello"))
        assert chunks
        assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
        assert "".join(chunk.text for chunk in chunks) == "Hello!"

    async def test_astream(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        chunks = [chunk async for chunk in chat.astream("Say hello")]
        assert chunks
        assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
        assert "".join(chunk.text for chunk in chunks) == "Hello!"

    def test_tool_calling(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        message = chat.bind_tools([Weather]).invoke("Weather in Paris?")
        assert isinstance(message, AIMessage)
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["name"] == "Weather"
        assert message.tool_calls[0]["args"] == {"city": "Paris"}

    async def test_tool_calling_async(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        message = await chat.bind_tools([Weather]).ainvoke("Weather in Paris?")
        assert isinstance(message, AIMessage)
        assert message.tool_calls[0]["name"] == "Weather"
        assert message.tool_calls[0]["args"] == {"city": "Paris"}

    def test_structured_output(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        result = chat.with_structured_output(Weather).invoke("Weather in Paris?")
        assert isinstance(result, Weather)
        assert result.city == "Paris"

    async def test_structured_output_async(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler)
        result = await chat.with_structured_output(Weather).ainvoke("Weather in Paris?")
        assert isinstance(result, Weather)
        assert result.city == "Paris"


class TestBetasPayloadBranch:
    """When the payload carries `betas`, requests must route through beta.messages."""

    def test_invoke_routes_to_beta_endpoint(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(
            client_settings, transport_handler, betas=["token-efficient-tools-2025-02-19"]
        )
        message = chat.invoke("Say hello")
        assert isinstance(message, AIMessage)
        assert message.text == "Hello!"
        request = transport_handler.requests[-1]
        assert request.url.path == "/v1/messages"
        assert "beta=true" in str(request.url.query)
        assert request.headers.get("anthropic-beta") == "token-efficient-tools-2025-02-19"

    async def test_ainvoke_routes_to_beta_endpoint(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(
            client_settings, transport_handler, betas=["token-efficient-tools-2025-02-19"]
        )
        message = await chat.ainvoke("Say hello")
        assert isinstance(message, AIMessage)
        request = transport_handler.requests[-1]
        assert "beta=true" in str(request.url.query)
        assert request.headers.get("anthropic-beta") == "token-efficient-tools-2025-02-19"

    def test_stream_routes_to_beta_endpoint(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(
            client_settings, transport_handler, betas=["token-efficient-tools-2025-02-19"]
        )
        chunks = list(chat.stream("Say hello"))
        assert "".join(chunk.text for chunk in chunks) == "Hello!"
        assert "beta=true" in str(transport_handler.requests[-1].url.query)


class TestGatewayMetadataExtraction:
    """langchain-anthropic reads gateway metadata headers off the raw response.

    `_add_gateway_metadata` only runs when the API key marks a LangSmith gateway
    (`lsv2_` prefix); use one to verify the raw wrapper still exposes headers.
    """

    def test_gateway_metadata_lands_in_generation_info(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler, anthropic_api_key="lsv2_test")
        result = chat.generate([[HumanMessage("Say hello")]])
        generation_info = result.generations[0][0].generation_info
        assert generation_info is not None
        assert generation_info["lc_gateway_metadata"] == {"provider": "anthropic"}

    async def test_gateway_metadata_lands_in_generation_info_async(
        self, client_settings: UiPathBaseSettings, transport_handler: _MockedTransport
    ) -> None:
        chat = _mocked_chat(client_settings, transport_handler, anthropic_api_key="lsv2_test")
        result = await chat.agenerate([[HumanMessage("Say hello")]])
        generation_info = result.generations[0][0].generation_info
        assert generation_info is not None
        assert generation_info["lc_gateway_metadata"] == {"provider": "anthropic"}
