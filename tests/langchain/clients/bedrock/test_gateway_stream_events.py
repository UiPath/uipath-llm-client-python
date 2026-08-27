"""Tests for gateway-injected events in the Bedrock Converse response stream.

Under an LLM Gateway configuration that bills per call, the gateway appends an
extra event-stream frame (``:event-type: costMetadata``) after the terminal AWS
frame. AWS does not define that event and ``langchain_aws`` raises
``ValueError`` on any event it does not recognize, so an otherwise successful
streamed run used to die after the model had already produced text.

The stream is driven from raw event-stream bytes through ``httpx.MockTransport``
so the whole path under test is the production one: ``WrappedBotoClient``
decodes the frames and ``ChatBedrockConverse._stream`` parses them.
"""

import binascii
import json
import os
import struct
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from uipath_langchain_client.clients.bedrock.chat_models import UiPathChatBedrockConverse
from uipath_langchain_client.clients.bedrock.utils import WrappedBotoClient

from uipath.llm_client.httpx_client import UiPathHttpxClient
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}

MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
COST = 0.026846

_STRING_HEADER_TYPE = 7


def _encode_frame(event_type: str, payload: dict[str, Any]) -> bytes:
    """Encode one `vnd.amazon.eventstream` frame the way AWS wires it."""
    body = json.dumps(payload).encode()
    headers = b""
    for name, value in ((":event-type", event_type), (":message-type", "event")):
        encoded_name, encoded_value = name.encode(), value.encode()
        headers += (
            bytes([len(encoded_name)])
            + encoded_name
            + bytes([_STRING_HEADER_TYPE])
            + struct.pack(">H", len(encoded_value))
            + encoded_value
        )
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    prelude += struct.pack(">I", binascii.crc32(prelude))
    message = prelude + headers + body
    return message + struct.pack(">I", binascii.crc32(message))


# An ordinary Converse stream, with the gateway's cost frame appended after the
# terminal `metadata` frame (where the gateway actually injects it).
CONVERSE_EVENTS: list[tuple[str, dict[str, Any]]] = [
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hello"}}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": " world"}}),
    ("contentBlockStop", {"contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    (
        "metadata",
        {
            "usage": {"inputTokens": 10, "outputTokens": 3, "totalTokens": 13},
            "metrics": {"latencyMs": 421},
        },
    ),
    ("costMetadata", {"associated_dollar_cost": COST}),
]


def _stream_bytes(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return b"".join(_encode_frame(event_type, payload) for event_type, payload in events)


@pytest.fixture(autouse=True)
def clear_singletons():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


def _make_chat(events: list[tuple[str, dict[str, Any]]]) -> UiPathChatBedrockConverse:
    """Build a converse client whose gateway transport replays `events`."""
    with patch.dict(os.environ, LLMGW_ENV, clear=True):
        chat = UiPathChatBedrockConverse(model=MODEL, settings=LLMGatewaySettings())
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=_stream_bytes(events),
            headers={"content-type": "application/vnd.amazon.eventstream"},
        )
    )
    sync_client = UiPathHttpxClient(
        base_url="https://cloud.uipath.com/gateway",
        model_name=MODEL,
        transport=transport,
    )
    object.__setattr__(chat, "uipath_sync_client", sync_client)
    chat.client = WrappedBotoClient(sync_client)
    return chat


def _assert_stream_ok(chunks: list[Any]) -> None:
    """The ordinary events must be unchanged and the cost must be surfaced."""
    assert "".join(chunk.text for chunk in chunks) == "Hello world"
    metadata: dict[str, Any] = {}
    usage = None
    for chunk in chunks:
        metadata.update(chunk.response_metadata)
        usage = chunk.usage_metadata or usage
    assert metadata["stopReason"] == "end_turn"
    assert metadata["metrics"] == {"latencyMs": 421}
    assert metadata["associated_dollar_cost"] == COST
    assert usage is not None
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 3


def test_stream_surfaces_gateway_cost_metadata() -> None:
    _assert_stream_ok(list(_make_chat(CONVERSE_EVENTS).stream("hi")))


@pytest.mark.asyncio
async def test_astream_surfaces_gateway_cost_metadata() -> None:
    chat = _make_chat(CONVERSE_EVENTS)
    chunks = [chunk async for chunk in chat.astream("hi")]
    _assert_stream_ok(chunks)


def test_stream_tolerates_unknown_gateway_event() -> None:
    """A gateway-only event we have never seen must not break the stream."""
    events = CONVERSE_EVENTS[:-1] + [("someFutureGatewayEvent", {"whatever": 1})]
    chunks = list(_make_chat(events).stream("hi"))
    assert "".join(chunk.text for chunk in chunks) == "Hello world"


def test_stream_without_gateway_events_is_unchanged() -> None:
    """Baseline: a pure AWS stream keeps flowing exactly as before."""
    chunks = list(_make_chat(CONVERSE_EVENTS[:-1]).stream("hi"))
    assert "".join(chunk.text for chunk in chunks) == "Hello world"
    metadata: dict[str, Any] = {}
    for chunk in chunks:
        metadata.update(chunk.response_metadata)
    assert metadata["stopReason"] == "end_turn"
    assert "associated_dollar_cost" not in metadata


def test_cost_metadata_before_terminal_metadata_frame_is_still_surfaced() -> None:
    """Order is not part of the contract, so accept the cost frame anywhere."""
    events = [CONVERSE_EVENTS[-1]] + CONVERSE_EVENTS[:-1]
    chunks = list(_make_chat(events).stream("hi"))
    _assert_stream_ok(chunks)
