"""Gateway dollar-cost capture on the Bedrock chat models, through their real SDK decoders.

Frames arrive one per chunk, the shape in which the gateway's trailing frame shows up.
"""

import os
from unittest.mock import patch

import httpx
import pytest
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)

from tests.aws_event_stream import bedrock_chunk, event_frame, event_stream_response
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.utils.dollar_cost import INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER

CONVERSE_STREAM_FRAMES = [
    event_frame("messageStart", {"role": "assistant"}),
    event_frame("contentBlockDelta", {"delta": {"text": "Hello"}, "contentBlockIndex": 0}),
    event_frame("contentBlockStop", {"contentBlockIndex": 0}),
    event_frame("messageStop", {"stopReason": "end_turn"}),
    event_frame(
        "metadata",
        {
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        },
    ),
]
COST_FRAME = event_frame("costMetadata", {"associated_dollar_cost": 0.002145})


@pytest.fixture
def llmgw_settings():
    env = {
        "LLMGW_URL": "https://cloud.uipath.com",
        "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
        "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
        "LLMGW_REQUESTING_PRODUCT": "test-product",
        "LLMGW_REQUESTING_FEATURE": "test-feature",
        "LLMGW_ACCESS_TOKEN": "test-access-token",
    }
    with patch.dict(os.environ, env, clear=True):
        return LLMGatewaySettings()


def _make_converse_chat(
    settings: LLMGatewaySettings, frames: list[bytes]
) -> UiPathChatBedrockConverse:
    chat = UiPathChatBedrockConverse(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        settings=settings,
        model_details={},
        default_headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"},
    )
    # The wrapped boto client holds this same httpx client.
    chat.uipath_sync_client._transport = httpx.MockTransport(  # type: ignore[attr-defined]
        lambda request: event_stream_response(*frames)
    )
    chat.uipath_sync_client._mounts = {}  # type: ignore[attr-defined]
    return chat


def test_converse_chat_stream_surfaces_gateway_cost(llmgw_settings) -> None:
    chat = _make_converse_chat(llmgw_settings, [*CONVERSE_STREAM_FRAMES, COST_FRAME])
    chunks = list(chat.stream("Hello"))
    priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
    assert len(priced) == 1
    assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145
    assert "".join(c.text for c in chunks) == "Hello"


async def test_converse_chat_astream_surfaces_gateway_cost(llmgw_settings) -> None:
    """langchain-core bridges the sync stream per-next() under copied contexts; the cost must survive."""
    chat = _make_converse_chat(llmgw_settings, [*CONVERSE_STREAM_FRAMES, COST_FRAME])
    chunks = [c async for c in chat.astream("Hello")]
    priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
    assert len(priced) == 1
    assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145


def test_converse_chat_stream_omits_cost_when_not_priced(llmgw_settings) -> None:
    chat = _make_converse_chat(llmgw_settings, CONVERSE_STREAM_FRAMES)
    chunks = list(chat.stream("Hello"))
    assert all("associated_dollar_cost" not in c.response_metadata for c in chunks)


# UiPathChatAnthropicBedrock: the anthropic SDK's own event-stream decoder drops unknown
# events, so the cost must come from the raw bytes.


ANTHROPIC_STREAM_FRAMES = [
    bedrock_chunk(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    ),
    bedrock_chunk(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    ),
    bedrock_chunk(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
    ),
    bedrock_chunk({"type": "content_block_stop", "index": 0}),
    bedrock_chunk(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    ),
    bedrock_chunk(
        {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 1,
                "outputTokenCount": 1,
                "invocationLatency": 1,
                "firstByteLatency": 1,
            },
        }
    ),
]


def _make_anthropic_bedrock_chat(
    settings: LLMGatewaySettings, frames: list[bytes]
) -> UiPathChatAnthropicBedrock:
    chat = UiPathChatAnthropicBedrock(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        settings=settings,
        model_details={},
        default_headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"},
    )
    transport = httpx.MockTransport(lambda request: event_stream_response(*frames))
    chat.uipath_sync_client._transport = transport  # type: ignore[attr-defined]
    chat.uipath_sync_client._mounts = {}  # type: ignore[attr-defined]
    chat.uipath_async_client._transport = transport  # type: ignore[attr-defined]
    chat.uipath_async_client._mounts = {}  # type: ignore[attr-defined]
    return chat


def test_anthropic_bedrock_stream_surfaces_gateway_cost(llmgw_settings) -> None:
    chat = _make_anthropic_bedrock_chat(llmgw_settings, [*ANTHROPIC_STREAM_FRAMES, COST_FRAME])
    chunks = list(chat.stream("Hello"))
    assert "".join(c.text for c in chunks) == "Hello"
    priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
    assert len(priced) == 1
    assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145


async def test_anthropic_bedrock_astream_surfaces_gateway_cost(llmgw_settings) -> None:
    chat = _make_anthropic_bedrock_chat(llmgw_settings, [*ANTHROPIC_STREAM_FRAMES, COST_FRAME])
    chunks = [c async for c in chat.astream("Hello")]
    assert "".join(c.text for c in chunks) == "Hello"
    priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
    assert len(priced) == 1
    assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145


# UiPathChatBedrock: langchain-aws stops at the vendor's stop event without reading to
# EOF, so the cost frame must be drained on close.


def _make_invoke_chat(settings: LLMGatewaySettings, frames: list[bytes]) -> UiPathChatBedrock:
    chat = UiPathChatBedrock(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        settings=settings,
        model_details={},
        default_headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"},
    )
    chat.uipath_sync_client._transport = httpx.MockTransport(  # type: ignore[attr-defined]
        lambda request: event_stream_response(*frames)
    )
    chat.uipath_sync_client._mounts = {}  # type: ignore[attr-defined]
    return chat


def test_invoke_chat_stream_surfaces_gateway_cost(llmgw_settings) -> None:
    chat = _make_invoke_chat(llmgw_settings, [*ANTHROPIC_STREAM_FRAMES, COST_FRAME])
    chunks = list(chat.stream("Hello"))
    assert "".join(c.text for c in chunks) == "Hello"
    priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
    assert len(priced) == 1
    assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145


async def test_invoke_chat_astream_surfaces_gateway_cost(llmgw_settings) -> None:
    chat = _make_invoke_chat(llmgw_settings, [*ANTHROPIC_STREAM_FRAMES, COST_FRAME])
    chunks = [c async for c in chat.astream("Hello")]
    priced = [c for c in chunks if "associated_dollar_cost" in c.response_metadata]
    assert len(priced) == 1
    assert priced[0].response_metadata["associated_dollar_cost"] == 0.002145


def test_invoke_chat_stream_omits_cost_when_not_priced(llmgw_settings) -> None:
    chat = _make_invoke_chat(llmgw_settings, ANTHROPIC_STREAM_FRAMES)
    chunks = list(chat.stream("Hello"))
    assert all("associated_dollar_cost" not in c.response_metadata for c in chunks)
