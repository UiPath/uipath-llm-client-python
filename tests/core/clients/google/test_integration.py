# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalIterable=false
# pyright: reportOperatorIssue=false
"""Integration tests for the UiPathGoogle (Google Gemini SDK wrapper) client.

Tests completions across multiple Gemini models, tool calling, structured
output, embeddings, and async operations. Uses VCR cassettes.

NOTE: These tests require pre-recorded VCR cassettes.
Run with --vcr-record=all to record new cassettes against a live LLMGateway.
"""

import json

import pytest
from google.genai import types

from uipath.llm_client.clients.google import UiPathGoogle
from uipath.llm_client.settings import UiPathBaseSettings

# Skip all tests — the native SDK wrappers need live cassette recording first.
pytestmark = pytest.mark.skip(
    reason="Requires pre-recorded VCR cassettes (run against live gateway)"
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def gemini_flash(client_settings: UiPathBaseSettings) -> UiPathGoogle:
    return UiPathGoogle(model_name="gemini-2.5-flash", client_settings=client_settings)


@pytest.fixture
def gemini_pro(client_settings: UiPathBaseSettings) -> UiPathGoogle:
    return UiPathGoogle(model_name="gemini-2.5-pro", client_settings=client_settings)


@pytest.fixture
def gemini3_flash(client_settings: UiPathBaseSettings) -> UiPathGoogle:
    return UiPathGoogle(model_name="gemini-3-flash-preview", client_settings=client_settings)


@pytest.fixture
def embedding_client(client_settings: UiPathBaseSettings) -> UiPathGoogle:
    return UiPathGoogle(model_name="gemini-embedding-001", client_settings=client_settings)


# ============================================================================
# Tool definitions
# ============================================================================

WEATHER_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get the current weather in a city",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "city": types.Schema(type="STRING", description="The city name"),
                    },
                    required=["city"],
                ),
            )
        ]
    )
]

# ============================================================================
# Completions
# ============================================================================


class TestGeminiCompletions:
    @pytest.mark.vcr()
    def test_flash_completion(self, gemini_flash: UiPathGoogle):
        response = gemini_flash.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello in one word.",
        )
        assert response.text

    @pytest.mark.vcr()
    def test_pro_completion(self, gemini_pro: UiPathGoogle):
        response = gemini_pro.models.generate_content(
            model="gemini-2.5-pro",
            contents="Say hello in one word.",
        )
        assert response.text

    @pytest.mark.vcr()
    def test_gemini3_flash_completion(self, gemini3_flash: UiPathGoogle):
        response = gemini3_flash.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Say hello in one word.",
        )
        assert response.text

    @pytest.mark.vcr()
    def test_completion_with_config(self, gemini_flash: UiPathGoogle):
        response = gemini_flash.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello in one word.",
            config={"max_output_tokens": 10, "temperature": 0.0},
        )
        assert response.text


# ============================================================================
# Tool calling
# ============================================================================


class TestGeminiToolCalling:
    @pytest.mark.vcr()
    def test_tool_calling(self, gemini_flash: UiPathGoogle):
        response = gemini_flash.models.generate_content(
            model="gemini-2.5-flash",
            contents="What is the weather in London?",
            config={"tools": WEATHER_TOOLS},
        )
        # The response should contain at least one candidate with function call parts
        assert response.candidates
        candidate = response.candidates[0]
        function_call_parts = [
            part for part in candidate.content.parts if part.function_call is not None
        ]
        assert len(function_call_parts) >= 1
        fc = function_call_parts[0].function_call
        assert fc.name == "get_weather"
        assert "city" in fc.args


# ============================================================================
# Structured output
# ============================================================================


class TestGeminiStructuredOutput:
    @pytest.mark.vcr()
    def test_json_output(self, gemini_flash: UiPathGoogle):
        response = gemini_flash.models.generate_content(
            model="gemini-2.5-flash",
            contents='What is 15 + 27? Respond with JSON: {"answer": <int>}',
            config={"response_mime_type": "application/json"},
        )
        assert response.text
        parsed = json.loads(response.text)
        assert parsed["answer"] == 42


# ============================================================================
# Embeddings
# ============================================================================


class TestGeminiEmbeddings:
    @pytest.mark.vcr()
    def test_single_embedding(self, embedding_client: UiPathGoogle):
        response = embedding_client.models.embed_content(
            model="gemini-embedding-001",
            contents="Hello world",
        )
        assert response.embeddings
        assert len(response.embeddings) >= 1
        assert len(response.embeddings[0].values) > 0

    @pytest.mark.vcr()
    def test_batch_embeddings(self, embedding_client: UiPathGoogle):
        response = embedding_client.models.embed_content(
            model="gemini-embedding-001",
            contents=["Hello", "World"],
        )
        assert response.embeddings
        assert len(response.embeddings) >= 2


# ============================================================================
# Async tests
# ============================================================================


class TestAsyncGemini:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_completion(self, gemini_flash: UiPathGoogle):
        response = await gemini_flash.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello in one word.",
        )
        assert response.text
