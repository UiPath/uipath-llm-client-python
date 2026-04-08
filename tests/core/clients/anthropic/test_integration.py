# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
"""Integration tests for the UiPath Anthropic SDK wrapper clients.

Tests completions across Bedrock and Vertex AI providers, including tool use,
structured output, extended thinking, streaming, and async. Uses VCR cassettes.

NOTE: These tests require pre-recorded VCR cassettes.
Run with --vcr-record=all to record new cassettes against a live LLMGateway.
"""

import json

import pytest
from pydantic import BaseModel

from uipath.llm_client.clients.anthropic import (
    UiPathAnthropicBedrock,
    UiPathAnthropicVertex,
    UiPathAsyncAnthropicBedrock,
    UiPathAsyncAnthropicVertex,
)
from uipath.llm_client.settings import UiPathBaseSettings

# Skip all tests — the native SDK wrappers need live cassette recording first.
pytestmark = pytest.mark.skip(
    reason="Requires pre-recorded VCR cassettes (run against live gateway)"
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def bedrock_haiku(client_settings: UiPathBaseSettings) -> UiPathAnthropicBedrock:
    return UiPathAnthropicBedrock(
        model_name="anthropic.claude-haiku-4-5-20251001-v1:0",
        client_settings=client_settings,
    )


@pytest.fixture
def bedrock_sonnet(client_settings: UiPathBaseSettings) -> UiPathAnthropicBedrock:
    return UiPathAnthropicBedrock(
        model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
        client_settings=client_settings,
    )


@pytest.fixture
def bedrock_opus(client_settings: UiPathBaseSettings) -> UiPathAnthropicBedrock:
    return UiPathAnthropicBedrock(
        model_name="anthropic.claude-opus-4-5-20251101-v1:0",
        client_settings=client_settings,
    )


@pytest.fixture
def vertex_sonnet(client_settings: UiPathBaseSettings) -> UiPathAnthropicVertex:
    return UiPathAnthropicVertex(
        model_name="claude-sonnet-4-5@20250929",
        client_settings=client_settings,
    )


@pytest.fixture
def vertex_haiku(client_settings: UiPathBaseSettings) -> UiPathAnthropicVertex:
    return UiPathAnthropicVertex(
        model_name="claude-haiku-4-5@20251001",
        client_settings=client_settings,
    )


@pytest.fixture
def async_bedrock_sonnet(client_settings: UiPathBaseSettings) -> UiPathAsyncAnthropicBedrock:
    return UiPathAsyncAnthropicBedrock(
        model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
        client_settings=client_settings,
    )


@pytest.fixture
def async_vertex_sonnet(client_settings: UiPathBaseSettings) -> UiPathAsyncAnthropicVertex:
    return UiPathAsyncAnthropicVertex(
        model_name="claude-sonnet-4-5@20250929",
        client_settings=client_settings,
    )


# ============================================================================
# Structured output models
# ============================================================================


class MathAnswer(BaseModel):
    answer: int
    explanation: str


# ============================================================================
# Tool definitions
# ============================================================================

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
        },
        "required": ["city"],
    },
}


# ============================================================================
# Bedrock completions
# ============================================================================


class TestBedrockCompletions:
    @pytest.mark.vcr()
    def test_haiku_completion(self, bedrock_haiku: UiPathAnthropicBedrock):
        response = bedrock_haiku.messages.create(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text
        assert response.stop_reason == "end_turn"

    @pytest.mark.vcr()
    def test_sonnet_completion(self, bedrock_sonnet: UiPathAnthropicBedrock):
        response = bedrock_sonnet.messages.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text

    @pytest.mark.vcr()
    def test_opus_completion(self, bedrock_opus: UiPathAnthropicBedrock):
        response = bedrock_opus.messages.create(
            model="anthropic.claude-opus-4-5-20251101-v1:0",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text

    @pytest.mark.vcr()
    def test_completion_with_params(self, bedrock_sonnet: UiPathAnthropicBedrock):
        response = bedrock_sonnet.messages.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": "Say hi."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"


# ============================================================================
# Vertex AI completions
# ============================================================================


class TestVertexCompletions:
    @pytest.mark.vcr()
    def test_sonnet_completion(self, vertex_sonnet: UiPathAnthropicVertex):
        response = vertex_sonnet.messages.create(
            model="claude-sonnet-4-5@20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text

    @pytest.mark.vcr()
    def test_haiku_completion(self, vertex_haiku: UiPathAnthropicVertex):
        response = vertex_haiku.messages.create(
            model="claude-haiku-4-5@20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text


# ============================================================================
# Tool calling
# ============================================================================


class TestAnthropicToolCalling:
    @pytest.mark.vcr()
    def test_tool_calling(self, bedrock_sonnet: UiPathAnthropicBedrock):
        response = bedrock_sonnet.messages.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=200,
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[WEATHER_TOOL],
        )
        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        assert len(tool_use_blocks) >= 1
        tool_block = tool_use_blocks[0]
        assert tool_block.name == "get_weather"
        assert "city" in tool_block.input


# ============================================================================
# Structured output
# ============================================================================


class TestAnthropicStructuredOutput:
    @pytest.mark.vcr()
    def test_json_output(self, bedrock_sonnet: UiPathAnthropicBedrock):
        response = bedrock_sonnet.messages.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": (
                        'What is 15 + 27? Respond ONLY with JSON: {"answer": <int>, "explanation": "<str>"}'
                    ),
                },
            ],
        )
        assert response.content
        assert response.content[0].type == "text"
        parsed = json.loads(response.content[0].text)
        result = MathAnswer.model_validate(parsed)
        assert result.answer == 42


# ============================================================================
# Extended thinking
# ============================================================================


class TestAnthropicExtendedThinking:
    @pytest.mark.vcr()
    def test_thinking_enabled(self, bedrock_sonnet: UiPathAnthropicBedrock):
        response = bedrock_sonnet.messages.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=2048,
            messages=[{"role": "user", "content": "What is 15 + 27?"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        assert response.content
        thinking_blocks = [block for block in response.content if block.type == "thinking"]
        assert len(thinking_blocks) >= 1
        assert thinking_blocks[0].thinking


# ============================================================================
# Streaming
# ============================================================================


class TestAnthropicStreaming:
    @pytest.mark.vcr()
    def test_streaming(self, bedrock_sonnet: UiPathAnthropicBedrock):
        collected_text = ""
        with bedrock_sonnet.messages.stream(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        ) as stream:
            for text in stream.text_stream:
                collected_text += text
        assert len(collected_text) > 0


# ============================================================================
# Async tests
# ============================================================================


class TestAsyncAnthropic:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_completion(self, async_bedrock_sonnet: UiPathAsyncAnthropicBedrock):
        response = await async_bedrock_sonnet.messages.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_vertex_completion(self, async_vertex_sonnet: UiPathAsyncAnthropicVertex):
        response = await async_vertex_sonnet.messages.create(
            model="claude-sonnet-4-5@20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.content
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text
