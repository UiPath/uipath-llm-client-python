"""Integration tests for UiPathNormalizedClient.

Tests the provider-agnostic normalized client against three providers:
- OpenAI (gpt-5.2-2025-12-11)
- Google Gemini (gemini-3-flash-preview)
- Anthropic on AWS Bedrock (anthropic.claude-haiku-4-5-20251001-v1:0)

Each provider is tested for:
- Normal chat completion (sync and async)
- Structured output via JSON schema
- Tool calling
- Streaming (sync and async)

Embeddings are tested with text-embedding-3-large and gemini-embedding-001.
"""

import json
from typing import Any

import pytest
from pydantic import BaseModel

from uipath.llm_client.clients.normalized import UiPathNormalizedClient
from uipath.llm_client.settings import UiPathBaseSettings

COMPLETION_MODELS = [
    "gpt-5.2-2025-12-11",
    "gemini-3-flash-preview",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
]

EMBEDDING_MODELS = [
    "text-embedding-3-large",
    "gemini-embedding-001",
]


# --- Tool definitions: dict, Pydantic, and callable forms ---

WEATHER_TOOL_DICT: dict[str, Any] = {
    "name": "get_weather",
    "description": "Get the current weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and country, e.g. 'Paris, France'.",
            },
        },
        "required": ["location"],
    },
}


class GetWeather(BaseModel):
    """Get the current weather for a given location."""

    location: str


def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"Sunny in {location}"


# --- Structured output: dict and Pydantic forms ---

CAPITAL_SCHEMA_DICT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "CapitalAnswer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "capital": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["capital", "country"],
            "additionalProperties": False,
        },
    },
}


class CapitalAnswer(BaseModel):
    capital: str
    country: str


@pytest.fixture(params=COMPLETION_MODELS)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=EMBEDDING_MODELS)
def embedding_model_name(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


_DEBUG_HEADERS = {"X-UiPath-LLMGateway-AllowFull4xxResponse": "true"}


@pytest.fixture
def client(model_name: str, client_settings: UiPathBaseSettings) -> UiPathNormalizedClient:
    return UiPathNormalizedClient(
        model_name=model_name,
        client_settings=client_settings,
        default_headers=_DEBUG_HEADERS,
    )


@pytest.fixture
def embed_client(
    embedding_model_name: str, client_settings: UiPathBaseSettings
) -> UiPathNormalizedClient:
    return UiPathNormalizedClient(
        model_name=embedding_model_name,
        client_settings=client_settings,
        default_headers=_DEBUG_HEADERS,
    )


@pytest.mark.vcr
class TestNormalizedClientCompletions:
    def test_create(self, client: UiPathNormalizedClient) -> None:
        response = client.completions.create(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=500,
        )

        assert "choices" in response
        assert len(response["choices"]) > 0
        message = response["choices"][0]["message"]
        assert message["role"] == "assistant"
        # Gemini 3 thinking models may omit content when only thinking blocks are present
        assert message.get("content") or message.get("tool_calls")

    @pytest.mark.asyncio
    async def test_acreate(self, client: UiPathNormalizedClient) -> None:
        response = await client.completions.acreate(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=500,
        )

        assert "choices" in response
        assert len(response["choices"]) > 0
        message = response["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert message.get("content") or message.get("tool_calls")

    def test_structured_output_pydantic(self, client: UiPathNormalizedClient) -> None:
        response = client.completions.create(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            output_format=CapitalAnswer,
            max_tokens=100,
        )

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert content
        data = json.loads(content)
        assert "capital" in data
        assert "Paris" in data["capital"]

    @pytest.mark.asyncio
    async def test_astructured_output_dict(self, client: UiPathNormalizedClient) -> None:
        response = await client.completions.acreate(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            output_format=CAPITAL_SCHEMA_DICT,
            max_tokens=100,
        )

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert content
        data = json.loads(content)
        assert "capital" in data
        assert "Paris" in data["capital"]

    def test_tool_calling_dict(self, client: UiPathNormalizedClient) -> None:
        response = client.completions.create(
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[WEATHER_TOOL_DICT],
            tool_choice={"type": "auto"},
            max_tokens=200,
        )

        assert "choices" in response
        choice = response["choices"][0]
        tool_calls = choice["message"].get("tool_calls", [])
        assert tool_calls, f"Expected tool_calls in response, got: {choice!r}"
        assert tool_calls[0].get("name") == "get_weather"

    @pytest.mark.asyncio
    async def test_atool_calling_pydantic(self, client: UiPathNormalizedClient) -> None:
        response = await client.completions.acreate(
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[GetWeather],
            tool_choice={"type": "auto"},
            max_tokens=200,
        )

        assert "choices" in response
        choice = response["choices"][0]
        tool_calls = choice["message"].get("tool_calls", [])
        assert tool_calls, f"Expected tool_calls in response, got: {choice!r}"
        assert tool_calls[0].get("name") == "GetWeather"

    def test_tool_calling_callable(self, client: UiPathNormalizedClient) -> None:
        response = client.completions.create(
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[get_weather],
            tool_choice={"type": "auto"},
            max_tokens=200,
        )

        assert "choices" in response
        choice = response["choices"][0]
        tool_calls = choice["message"].get("tool_calls", [])
        assert tool_calls, f"Expected tool_calls in response, got: {choice!r}"
        assert tool_calls[0].get("name") == "get_weather"

    def test_stream(self, client: UiPathNormalizedClient) -> None:
        chunks = list(
            client.completions.stream(
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
                max_tokens=500,
            )
        )

        assert len(chunks) > 0
        all_content = "".join(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            or chunk.get("choices", [{}])[0].get("message", {}).get("content", "")
            for chunk in chunks
            if chunk.get("choices")
        )
        assert all_content

    @pytest.mark.asyncio
    async def test_astream(self, client: UiPathNormalizedClient) -> None:
        chunks = []
        async for chunk in client.completions.astream(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=500,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        all_content = "".join(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            or chunk.get("choices", [{}])[0].get("message", {}).get("content", "")
            for chunk in chunks
            if chunk.get("choices")
        )
        assert all_content


@pytest.mark.vcr
class TestNormalizedClientEmbeddings:
    def test_create(self, embed_client: UiPathNormalizedClient) -> None:
        response = embed_client.embeddings.create(input="Hello world")

        assert "data" in response
        assert len(response["data"]) > 0
        embedding = response["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_acreate(self, embed_client: UiPathNormalizedClient) -> None:
        response = await embed_client.embeddings.acreate(input="Hello world")

        assert "data" in response
        assert len(response["data"]) > 0
        embedding = response["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_create_batch(self, embed_client: UiPathNormalizedClient) -> None:
        texts = ["Hello world", "Goodbye world", "How are you?"]
        response = embed_client.embeddings.create(input=texts)

        assert "data" in response
        assert len(response["data"]) == len(texts)
        for item in response["data"]:
            assert "embedding" in item
            assert isinstance(item["embedding"], list)
            assert len(item["embedding"]) > 0
