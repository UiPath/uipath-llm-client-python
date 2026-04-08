# pyright: reportAttributeAccessIssue=false
"""Integration tests for the UiPathLiteLLM client.

Tests completions across multiple providers (OpenAI, Gemini, Bedrock, Vertex AI),
including structured output, tool use, and streaming. Uses VCR cassettes.
"""

import json

import pytest
from pydantic import BaseModel

from uipath.llm_client.clients.litellm import UiPathLiteLLM
from uipath.llm_client.settings import UiPathBaseSettings
from uipath.llm_client.settings.constants import ApiFlavor

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def openai_gpt4o_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(model_name="gpt-4o-2024-11-20", client_settings=client_settings)


@pytest.fixture
def openai_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(model_name="gpt-5.2-2025-12-11", client_settings=client_settings)


@pytest.fixture
def openai_gpt54_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(model_name="gpt-5.4-2026-03-05", client_settings=client_settings)


@pytest.fixture
def openai_responses_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(
        model_name="gpt-5.2-2025-12-11",
        client_settings=client_settings,
        api_flavor=ApiFlavor.RESPONSES,
    )


@pytest.fixture
def gemini_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(model_name="gemini-2.5-flash", client_settings=client_settings)


@pytest.fixture
def gemini_pro_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(model_name="gemini-2.5-pro", client_settings=client_settings)


@pytest.fixture
def gemini3_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(model_name="gemini-3-flash-preview", client_settings=client_settings)


@pytest.fixture
def bedrock_invoke_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(
        model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
        client_settings=client_settings,
    )


@pytest.fixture
def bedrock_converse_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(
        model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
        client_settings=client_settings,
        api_flavor=ApiFlavor.CONVERSE,
    )


@pytest.fixture
def bedrock_haiku_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(
        model_name="anthropic.claude-haiku-4-5-20251001-v1:0",
        client_settings=client_settings,
    )


@pytest.fixture
def bedrock_opus_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(
        model_name="anthropic.claude-opus-4-5-20251101-v1:0",
        client_settings=client_settings,
    )


@pytest.fixture
def vertex_claude_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(
        model_name="claude-sonnet-4-5@20250929",
        client_settings=client_settings,
    )


@pytest.fixture
def vertex_haiku_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(
        model_name="claude-haiku-4-5@20251001",
        client_settings=client_settings,
    )


@pytest.fixture
def embedding_client(client_settings: UiPathBaseSettings) -> UiPathLiteLLM:
    return UiPathLiteLLM(model_name="text-embedding-3-large", client_settings=client_settings)


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
# OpenAI completions (chat-completions flavor)
# ============================================================================


class TestOpenAICompletions:
    @pytest.mark.vcr()
    def test_gpt4o_completion(self, openai_gpt4o_client: UiPathLiteLLM):
        response = openai_gpt4o_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.vcr()
    def test_basic_completion(self, openai_client: UiPathLiteLLM):
        response = openai_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.vcr()
    def test_gpt54_completion(self, openai_gpt54_client: UiPathLiteLLM):
        response = openai_gpt54_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.vcr()
    def test_completion_with_params(self, openai_client: UiPathLiteLLM):
        response = openai_client.completion(
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
            temperature=0.0,
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_reasoning_effort(self, openai_client: UiPathLiteLLM):
        response = openai_client.completion(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            reasoning_effort="medium",
        )
        assert response.choices[0].message.content


class TestOpenAIResponses:
    @pytest.mark.vcr()
    def test_responses_api(self, openai_responses_client: UiPathLiteLLM):
        response = openai_responses_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content


# ============================================================================
# OpenAI structured output
# ============================================================================


class TestOpenAIStructuredOutput:
    @pytest.mark.vcr()
    def test_json_object(self, openai_client: UiPathLiteLLM):
        response = openai_client.completion(
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
    def test_pydantic_structured_output(self, openai_client: UiPathLiteLLM):
        response = openai_client.completion(
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
# OpenAI tool use
# ============================================================================


class TestOpenAIToolUse:
    @pytest.mark.vcr()
    def test_tool_calling(self, openai_client: UiPathLiteLLM):
        response = openai_client.completion(
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
# Gemini completions
# ============================================================================


class TestGeminiCompletions:
    @pytest.mark.vcr()
    def test_basic_completion(self, gemini_client: UiPathLiteLLM):
        response = gemini_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_gemini_pro_completion(self, gemini_pro_client: UiPathLiteLLM):
        response = gemini_pro_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_gemini3_completion(self, gemini3_client: UiPathLiteLLM):
        response = gemini3_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content


class TestGeminiToolUse:
    @pytest.mark.vcr()
    def test_tool_calling(self, gemini_client: UiPathLiteLLM):
        response = gemini_client.completion(
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[WEATHER_TOOL],
        )
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls and len(tool_calls) >= 1


# ============================================================================
# Bedrock completions (invoke + converse)
# ============================================================================


class TestBedrockInvokeCompletions:
    @pytest.mark.vcr()
    def test_basic_completion(self, bedrock_invoke_client: UiPathLiteLLM):
        response = bedrock_invoke_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=50,
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_haiku_completion(self, bedrock_haiku_client: UiPathLiteLLM):
        response = bedrock_haiku_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=50,
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_opus_completion(self, bedrock_opus_client: UiPathLiteLLM):
        response = bedrock_opus_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=50,
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_tool_calling(self, bedrock_invoke_client: UiPathLiteLLM):
        response = bedrock_invoke_client.completion(
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[WEATHER_TOOL],
            max_tokens=200,
        )
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls and len(tool_calls) >= 1


class TestBedrockConverseCompletions:
    @pytest.mark.vcr()
    def test_basic_completion(self, bedrock_converse_client: UiPathLiteLLM):
        response = bedrock_converse_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=50,
        )
        assert response.choices[0].message.content


# ============================================================================
# Vertex AI Claude completions
# ============================================================================


class TestVertexClaudeCompletions:
    @pytest.mark.vcr()
    def test_basic_completion(self, vertex_claude_client: UiPathLiteLLM):
        response = vertex_claude_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=50,
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_vertex_haiku_completion(self, vertex_haiku_client: UiPathLiteLLM):
        response = vertex_haiku_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=50,
        )
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_tool_calling(self, vertex_claude_client: UiPathLiteLLM):
        response = vertex_claude_client.completion(
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[WEATHER_TOOL],
            max_tokens=200,
        )
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls and len(tool_calls) >= 1


# ============================================================================
# Embeddings
# ============================================================================


class TestEmbeddings:
    @pytest.mark.vcr()
    def test_single_embedding(self, embedding_client: UiPathLiteLLM):
        response = embedding_client.embedding(input="Hello world")
        assert len(response.data) == 1
        assert len(response.data[0]["embedding"]) > 0

    @pytest.mark.vcr()
    def test_batch_embeddings(self, embedding_client: UiPathLiteLLM):
        response = embedding_client.embedding(input=["Hello world", "Goodbye world"])
        assert len(response.data) == 2


# ============================================================================
# Streaming tests
# ============================================================================


class TestStreaming:
    @pytest.mark.vcr()
    def test_openai_streaming(self, openai_client: UiPathLiteLLM):
        response = openai_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            stream=True,
        )
        chunks = []
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert len(chunks) > 0

    @pytest.mark.vcr()
    def test_gemini_streaming(self, gemini_client: UiPathLiteLLM):
        response = gemini_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            stream=True,
        )
        chunks = []
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert len(chunks) > 0

    @pytest.mark.vcr()
    def test_gemini_streaming_tool_calling(self, gemini_client: UiPathLiteLLM):
        response = gemini_client.completion(
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[WEATHER_TOOL],
            stream=True,
        )
        tool_calls_found = False
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                tool_calls_found = True
        assert tool_calls_found

    @pytest.mark.vcr()
    def test_vertex_claude_streaming(self, vertex_claude_client: UiPathLiteLLM):
        response = vertex_claude_client.completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=50,
            stream=True,
        )
        chunks = []
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert len(chunks) > 0


# ============================================================================
# Async tests
# ============================================================================


class TestAsyncCompletions:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_openai_completion(self, openai_client: UiPathLiteLLM):
        response = await openai_client.acompletion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_gemini_completion(self, gemini_client: UiPathLiteLLM):
        response = await gemini_client.acompletion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.choices[0].message.content


class TestAsyncEmbeddings:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_embedding(self, embedding_client: UiPathLiteLLM):
        response = await embedding_client.aembedding(input="Hello world")
        assert len(response.data) == 1
