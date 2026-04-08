# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Integration tests for UiPath OpenAI SDK wrapper clients.

Tests completions, structured output, tool use, streaming, reasoning,
embeddings, and async operations across UiPathOpenAI, UiPathAzureOpenAI,
and their async variants. Uses VCR cassettes.

NOTE: These tests require pre-recorded VCR cassettes.
Run with --vcr-record=all to record new cassettes against a live LLMGateway.
"""

import json

import pytest
from pydantic import BaseModel

from uipath.llm_client.clients.openai import (
    UiPathAsyncOpenAI,
    UiPathAzureOpenAI,
    UiPathOpenAI,
)
from uipath.llm_client.settings import UiPathBaseSettings

# Skip all tests in this module — the native SDK wrappers need live cassette
# recording first (the OpenAI SDK's event-hook URL rewriting requires host
# resolution through the actual gateway).
pytestmark = pytest.mark.skip(
    reason="Requires pre-recorded VCR cassettes (run against live gateway)"
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def openai_gpt4o(client_settings: UiPathBaseSettings) -> UiPathOpenAI:
    return UiPathOpenAI(model_name="gpt-4o-2024-11-20", client_settings=client_settings)


@pytest.fixture
def openai_gpt52(client_settings: UiPathBaseSettings) -> UiPathOpenAI:
    return UiPathOpenAI(model_name="gpt-5.2-2025-12-11", client_settings=client_settings)


@pytest.fixture
def openai_gpt54(client_settings: UiPathBaseSettings) -> UiPathOpenAI:
    return UiPathOpenAI(model_name="gpt-5.4-2026-03-05", client_settings=client_settings)


@pytest.fixture
def azure_openai_gpt4o(client_settings: UiPathBaseSettings) -> UiPathAzureOpenAI:
    return UiPathAzureOpenAI(model_name="gpt-4o-2024-11-20", client_settings=client_settings)


@pytest.fixture
def embedding_client(client_settings: UiPathBaseSettings) -> UiPathAzureOpenAI:
    return UiPathAzureOpenAI(model_name="text-embedding-3-large", client_settings=client_settings)


@pytest.fixture
def async_openai_gpt4o(client_settings: UiPathBaseSettings) -> UiPathAsyncOpenAI:
    return UiPathAsyncOpenAI(model_name="gpt-4o-2024-11-20", client_settings=client_settings)


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
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    },
}


# ============================================================================
# OpenAI completions (chat-completions)
# ============================================================================


class TestOpenAICompletions:
    @pytest.mark.vcr()
    def test_basic_completion(self, openai_gpt4o: UiPathOpenAI):
        response = openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.vcr()
    def test_gpt52_completion(self, openai_gpt52: UiPathOpenAI):
        response = openai_gpt52.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.vcr()
    def test_gpt54_completion(self, openai_gpt54: UiPathOpenAI):
        response = openai_gpt54.chat.completions.create(
            model="gpt-5.4-2026-03-05",
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.vcr()
    def test_completion_with_params(self, openai_gpt4o: UiPathOpenAI):
        response = openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
            temperature=0.0,
        )
        assert response.choices[0].message.content


# ============================================================================
# Azure OpenAI completions
# ============================================================================


class TestAzureOpenAICompletions:
    @pytest.mark.vcr()
    def test_basic_completion(self, azure_openai_gpt4o: UiPathAzureOpenAI):
        response = azure_openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"


# ============================================================================
# OpenAI Responses API
# ============================================================================


class TestOpenAIResponses:
    @pytest.mark.vcr()
    def test_responses_api(self, openai_gpt52: UiPathOpenAI):
        response = openai_gpt52.responses.create(
            model="gpt-5.2-2025-12-11",
            input="Say hello in one word.",
        )
        assert response.output


# ============================================================================
# OpenAI structured output
# ============================================================================


class TestOpenAIStructuredOutput:
    @pytest.mark.vcr()
    def test_json_object(self, openai_gpt4o: UiPathOpenAI):
        response = openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": 'What is 15 + 27? Respond with JSON: {"answer": <int>, "explanation": "<str>"}',
                },
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert parsed["answer"] == 42

    @pytest.mark.vcr()
    def test_pydantic_structured_output(self, openai_gpt4o: UiPathOpenAI):
        response = openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": 'What is 15 + 27? Respond with JSON: {"answer": <int>, "explanation": "<str>"}',
                },
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        assert content
        parsed = MathAnswer.model_validate_json(content)
        assert parsed.answer == 42


# ============================================================================
# OpenAI tool calling
# ============================================================================


class TestOpenAIToolCalling:
    @pytest.mark.vcr()
    def test_tool_calling(self, openai_gpt4o: UiPathOpenAI):
        response = openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
        )
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert "city" in args


# ============================================================================
# OpenAI streaming
# ============================================================================


class TestOpenAIStreaming:
    @pytest.mark.vcr()
    def test_streaming(self, openai_gpt4o: UiPathOpenAI):
        stream = openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        result = "".join(chunks)
        assert len(result) > 0


# ============================================================================
# OpenAI reasoning
# ============================================================================


class TestOpenAIReasoning:
    @pytest.mark.vcr()
    def test_reasoning_effort(self, openai_gpt52: UiPathOpenAI):
        response = openai_gpt52.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            reasoning_effort="medium",
        )
        assert response.choices[0].message.content


# ============================================================================
# OpenAI embeddings
# ============================================================================


class TestOpenAIEmbeddings:
    @pytest.mark.vcr()
    def test_single_embedding(self, embedding_client: UiPathAzureOpenAI):
        response = embedding_client.embeddings.create(
            model="text-embedding-3-large",
            input="Hello world",
        )
        assert len(response.data) == 1
        assert len(response.data[0].embedding) > 0

    @pytest.mark.vcr()
    def test_batch_embeddings(self, embedding_client: UiPathAzureOpenAI):
        response = embedding_client.embeddings.create(
            model="text-embedding-3-large",
            input=["Hello", "World"],
        )
        assert len(response.data) == 2


# ============================================================================
# Async tests
# ============================================================================


class TestAsyncOpenAI:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_completion(self, async_openai_gpt4o: UiPathAsyncOpenAI):
        response = await async_openai_gpt4o.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"
