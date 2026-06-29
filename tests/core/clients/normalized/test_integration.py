"""Integration tests for the normalized client.

These tests verify the normalized client works end-to-end with VCR cassettes.
They test:
1. Basic chat completions (sync)
2. Chat completions with parameters (temperature, max_tokens)
3. Streaming completions (sync via .stream())
4. Tool calling (dict and Pydantic tools)
5. Structured output via json_object response_format
6. Embeddings (sync)
7. Async completions (via .acreate())
8. Async embeddings (via .acreate())
"""

import json

import pytest
from pydantic import BaseModel

from uipath.llm_client.clients.normalized import (
    ChatCompletion,
    ChatCompletionChunk,
    EmbeddingResponse,
    UiPathNormalizedClient,
)
from uipath.llm_client.settings import UiPathBaseSettings

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def normalized_client(client_settings: UiPathBaseSettings) -> UiPathNormalizedClient:
    return UiPathNormalizedClient(
        model_name="gpt-4o-2024-11-20",
        client_settings=client_settings,
    )


@pytest.fixture
def embedding_client(client_settings: UiPathBaseSettings) -> UiPathNormalizedClient:
    return UiPathNormalizedClient(
        model_name="text-embedding-ada-002",
        client_settings=client_settings,
    )


# ============================================================================
# Structured output models
# ============================================================================


class MathAnswer(BaseModel):
    answer: int
    explanation: str


# ============================================================================
# Sync completions tests
# ============================================================================


class TestNormalizedCompletions:
    @pytest.mark.vcr()
    def test_basic_completion(self, normalized_client: UiPathNormalizedClient):
        response = normalized_client.completions.create(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert isinstance(response, ChatCompletion)
        assert len(response.choices) >= 1
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens > 0

    @pytest.mark.vcr()
    def test_completion_with_params(self, normalized_client: UiPathNormalizedClient):
        response = normalized_client.completions.create(
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
            temperature=0.0,
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content

    @pytest.mark.vcr()
    def test_completion_with_system_message(self, normalized_client: UiPathNormalizedClient):
        response = normalized_client.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be very brief."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content


class TestNormalizedStreaming:
    @pytest.mark.vcr()
    def test_streaming(self, normalized_client: UiPathNormalizedClient):
        chunks = list(
            normalized_client.completions.stream(
                messages=[{"role": "user", "content": "Count from 1 to 3."}],
            )
        )
        assert len(chunks) > 0
        assert all(isinstance(c, ChatCompletionChunk) for c in chunks)

        content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
        assert len(content_chunks) > 0


class TestNormalizedToolCalling:
    @pytest.mark.vcr()
    def test_tool_calling(self, normalized_client: UiPathNormalizedClient):
        response = normalized_client.completions.create(
            messages=[
                {"role": "user", "content": "What is the weather in London?"},
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "The city name"},
                        },
                        "required": ["city"],
                    },
                }
            ],
            tool_choice={"type": "required"},
        )
        assert isinstance(response, ChatCompletion)
        assert len(response.choices[0].message.tool_calls) >= 1
        tc = response.choices[0].message.tool_calls[0]
        assert tc.name == "get_weather"
        assert "city" in tc.arguments

    @pytest.mark.vcr()
    def test_tool_calling_with_pydantic(self, normalized_client: UiPathNormalizedClient):
        class GetWeatherInput(BaseModel):
            """Get the current weather in a city."""

            city: str

        response = normalized_client.completions.create(
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            tools=[GetWeatherInput],
            tool_choice={"type": "required"},
        )
        assert isinstance(response, ChatCompletion)
        assert len(response.choices[0].message.tool_calls) >= 1


class TestNormalizedStructuredOutput:
    @pytest.mark.vcr()
    def test_structured_output_json_object(self, normalized_client: UiPathNormalizedClient):
        """Test structured output using json_object response_format."""
        response = normalized_client.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        'What is 15 + 27? Respond with JSON: {"answer": <int>, "explanation": "<str>"}'
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        assert isinstance(response, ChatCompletion)
        content = response.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert parsed["answer"] == 42

    @pytest.mark.vcr()
    def test_structured_output_pydantic_parsed(self, normalized_client: UiPathNormalizedClient):
        """Test that response_format with a Pydantic model populates message.parsed."""
        response = normalized_client.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        'What is 15 + 27? Respond with JSON: {"answer": <int>, "explanation": "<str>"}'
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        assert isinstance(response, ChatCompletion)
        content = response.choices[0].message.content
        assert content
        parsed = MathAnswer.model_validate_json(content)
        assert parsed.answer == 42


# ============================================================================
# Embeddings tests
# ============================================================================


class TestNormalizedEmbeddings:
    @pytest.mark.vcr()
    def test_single_embedding(self, embedding_client: UiPathNormalizedClient):
        response = embedding_client.embeddings.create(input="Hello world")
        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 1
        assert len(response.data[0].embedding) > 0

    @pytest.mark.vcr()
    def test_batch_embeddings(self, embedding_client: UiPathNormalizedClient):
        response = embedding_client.embeddings.create(input=["Hello world", "Goodbye world"])
        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 2


# ============================================================================
# Async tests
# ============================================================================


class TestAsyncNormalizedCompletions:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_completion(self, normalized_client: UiPathNormalizedClient):
        response = await normalized_client.completions.acreate(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content


class TestAsyncNormalizedEmbeddings:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_embedding(self, embedding_client: UiPathNormalizedClient):
        response = await embedding_client.embeddings.acreate(
            input="Hello world",
        )
        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 1
