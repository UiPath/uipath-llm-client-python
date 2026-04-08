"""Tests for the normalized client module.

This module tests:
1. UiPathNormalizedClient initialization and client creation
2. Completions.create (sync, non-streaming)
3. Completions.stream (sync, streaming)
4. Completions.acreate (async, non-streaming)
5. Tool calling (tool definition building, tool_choice resolution)
6. Structured output (Pydantic, TypedDict, dict schemas)
7. Embeddings.create and Embeddings.acreate
8. Response type parsing (ChatCompletion, ChatCompletionChunk, EmbeddingResponse)
"""

import json
from collections.abc import Generator
from typing import TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from uipath.llm_client.clients.normalized import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Delta,
    EmbeddingData,
    EmbeddingResponse,
    Message,
    ToolCall,
    ToolCallChunk,
    UiPathNormalizedClient,
    Usage,
)
from uipath.llm_client.clients.normalized.completions import (
    Completions,
    _aiter_sse,
    _build_request,
    _build_response_format,
    _build_tool_definition,
    _iter_sse,
    _parse_response,
    _parse_stream_chunk,
    _parse_structured_output,
    _parse_tool_call,
    _parse_tool_call_chunk,
    _resolve_tool_choice,
)
from uipath.llm_client.clients.normalized.embeddings import _parse_embedding_response
from uipath.llm_client.settings.utils import SingletonMeta

# ============================================================================
# Fixtures
# ============================================================================

_CLIENT_MODULE = "uipath.llm_client.clients.normalized.client"


@pytest.fixture(autouse=True)
def clear_singleton_instances():
    """Clear singleton instances before each test to ensure isolation."""
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.build_base_url.return_value = "https://gateway.uipath.com/llm/v1"
    settings.build_auth_headers.return_value = {"Authorization": "Bearer test-token"}
    settings.build_auth_pipeline.return_value = None
    return settings


@pytest.fixture
def mock_sync_client():
    client = MagicMock()
    return client


@pytest.fixture
def mock_async_client():
    client = AsyncMock()
    return client


# ============================================================================
# Response parsing helpers
# ============================================================================

SAMPLE_COMPLETION_RESPONSE = {
    "id": "chatcmpl-123",
    "created": 1234567890,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}

SAMPLE_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-456",
    "created": 1234567890,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "name": "get_weather",
                        "arguments": {"city": "London"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 20,
        "total_tokens": 35,
    },
}

SAMPLE_STREAM_CHUNKS = [
    {
        "id": "chatcmpl-789",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-789",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": {"content": " world!"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-789",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        },
    },
]

SAMPLE_EMBEDDING_RESPONSE = {
    "data": [
        {"embedding": [0.1, 0.2, 0.3], "index": 0},
        {"embedding": [0.4, 0.5, 0.6], "index": 1},
    ],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}


# ============================================================================
# Test: Response type parsing
# ============================================================================


class TestParseResponse:
    def test_basic_completion(self):
        result = _parse_response(SAMPLE_COMPLETION_RESPONSE)
        assert isinstance(result, ChatCompletion)
        assert result.id == "chatcmpl-123"
        assert result.model == "gpt-4o"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello! How can I help you?"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 8
        assert result.usage.total_tokens == 18

    def test_tool_call_response(self):
        result = _parse_response(SAMPLE_TOOL_CALL_RESPONSE)
        assert len(result.choices[0].message.tool_calls) == 1
        tc = result.choices[0].message.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "London"}

    def test_empty_response(self):
        result = _parse_response({"choices": [], "usage": {}})
        assert len(result.choices) == 0
        assert result.usage.prompt_tokens == 0

    def test_tool_call_with_string_arguments(self):
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "func",
                                "arguments": '{"key": "value"}',
                            }
                        ]
                    }
                }
            ],
            "usage": {},
        }
        result = _parse_response(data)
        tc = result.choices[0].message.tool_calls[0]
        assert tc.arguments == {"key": "value"}

    def test_tool_call_with_invalid_json_arguments(self):
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "func",
                                "arguments": "not json",
                            }
                        ]
                    }
                }
            ],
            "usage": {},
        }
        result = _parse_response(data)
        tc = result.choices[0].message.tool_calls[0]
        assert tc.arguments == {}


class TestParseStreamChunk:
    def test_content_chunk(self):
        result = _parse_stream_chunk(SAMPLE_STREAM_CHUNKS[0])
        assert isinstance(result, ChatCompletionChunk)
        assert result.id == "chatcmpl-789"
        assert len(result.choices) == 1
        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].delta.role == "assistant"

    def test_chunk_with_usage(self):
        result = _parse_stream_chunk(SAMPLE_STREAM_CHUNKS[2])
        assert result.usage is not None
        assert result.usage.prompt_tokens == 5
        assert result.choices[0].finish_reason == "stop"

    def test_chunk_without_usage(self):
        result = _parse_stream_chunk(SAMPLE_STREAM_CHUNKS[0])
        assert result.usage is None

    def test_stream_tool_call_chunk(self):
        data = {
            "id": "chatcmpl-tc",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "get_weather",
                                "arguments": '{"city":',
                                "index": 0,
                            }
                        ]
                    }
                }
            ],
        }
        result = _parse_stream_chunk(data)
        assert len(result.choices[0].delta.tool_calls) == 1
        tc = result.choices[0].delta.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == '{"city":'

    def test_stream_tool_call_with_function_format(self):
        data = {
            "id": "chatcmpl-tc",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                                "index": 0,
                            }
                        ]
                    }
                }
            ],
        }
        result = _parse_stream_chunk(data)
        tc = result.choices[0].delta.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == '{"city": "Paris"}'


class TestParseEmbeddingResponse:
    def test_basic_embedding(self):
        result = _parse_embedding_response(SAMPLE_EMBEDDING_RESPONSE)
        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 2
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[1].embedding == [0.4, 0.5, 0.6]
        assert result.model == "text-embedding-ada-002"
        assert result.usage.prompt_tokens == 5

    def test_empty_embedding(self):
        result = _parse_embedding_response({"data": [], "usage": {}})
        assert len(result.data) == 0


# ============================================================================
# Test: Structured output
# ============================================================================


class TestBuildResponseFormat:
    def test_pydantic_model(self):
        class MyModel(BaseModel):
            name: str
            age: int

        result = _build_response_format(MyModel)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "MyModel"
        assert result["json_schema"]["strict"] is True
        assert "properties" in result["json_schema"]["schema"]

    def test_typed_dict(self):
        class MyDict(TypedDict):
            name: str
            score: float

        result = _build_response_format(MyDict)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "MyDict"
        assert result["json_schema"]["strict"] is True
        schema = result["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "score" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["score"]["type"] == "number"

    def test_dict_schema(self):
        schema = {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        }
        result = _build_response_format(schema)
        assert result["type"] == "json_schema"
        assert result["json_schema"] == schema

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported response_format"):
            _build_response_format("not a type")  # type: ignore[arg-type]


class TestParseStructuredOutput:
    def test_parse_pydantic(self):
        class Answer(BaseModel):
            text: str
            score: float

        content = '{"text": "hello", "score": 0.9}'
        result = _parse_structured_output(content, Answer)
        assert isinstance(result, Answer)
        assert result.text == "hello"
        assert result.score == 0.9

    def test_parse_dict(self):
        content = '{"key": "value"}'
        result = _parse_structured_output(content, {"type": "object"})
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        result = _parse_structured_output("not json", str)
        assert result is None

    def test_response_with_structured_output(self):
        class Answer(BaseModel):
            text: str

        data = {
            "choices": [
                {
                    "message": {
                        "content": '{"text": "hello"}',
                    }
                }
            ],
            "usage": {},
        }
        result = _parse_response(data, response_format=Answer)
        assert result.choices[0].message.parsed is not None
        assert isinstance(result.choices[0].message.parsed, Answer)
        assert result.choices[0].message.parsed.text == "hello"

    def test_response_without_structured_output(self):
        data = {
            "choices": [
                {
                    "message": {
                        "content": "plain text",
                    }
                }
            ],
            "usage": {},
        }
        result = _parse_response(data)
        assert result.choices[0].message.parsed is None


# ============================================================================
# Test: Tool definition building
# ============================================================================


class TestBuildToolDefinition:
    def test_dict_passthrough(self):
        tool = {"name": "my_tool", "description": "does stuff", "parameters": {}}
        result = _build_tool_definition(tool)
        assert result is tool

    def test_pydantic_model(self):
        class WeatherInput(BaseModel):
            """Get weather for a city."""

            city: str
            units: str = "celsius"

        result = _build_tool_definition(WeatherInput)
        assert result["name"] == "WeatherInput"
        assert result["description"] == "Get weather for a city."
        assert "properties" in result["parameters"]
        assert "city" in result["parameters"]["properties"]

    def test_callable(self):
        def get_weather(city: str, units: str = "celsius") -> str:
            """Get weather for a city."""
            return f"Weather in {city}"

        result = _build_tool_definition(get_weather)
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather for a city."
        assert "city" in result["parameters"]["properties"]
        assert "city" in result["parameters"]["required"]
        assert "units" not in result["parameters"]["required"]

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported tool type"):
            _build_tool_definition(42)  # type: ignore[arg-type]


class TestToolChoiceResolution:
    def test_auto(self):
        result = _resolve_tool_choice("auto", [])
        assert result == "auto"

    def test_required(self):
        result = _resolve_tool_choice("required", [])
        assert result == "required"

    def test_none(self):
        result = _resolve_tool_choice("none", [])
        assert result == "none"

    def test_specific_tool(self):
        tools = [{"name": "get_weather"}, {"name": "search"}]
        result = _resolve_tool_choice("get_weather", tools)
        assert result == {"type": "tool", "name": "get_weather"}

    def test_unknown_becomes_auto(self):
        result = _resolve_tool_choice("unknown_tool", [{"name": "other"}])
        assert result == "auto"

    def test_dict_passthrough(self):
        choice = {"type": "required"}
        result = _resolve_tool_choice(choice, [])
        assert result is choice


# ============================================================================
# Test: Tool call parsing
# ============================================================================


class TestParseToolCall:
    def test_basic(self):
        tc = _parse_tool_call({"id": "call_1", "name": "func", "arguments": {"x": 1}})
        assert tc.id == "call_1"
        assert tc.name == "func"
        assert tc.arguments == {"x": 1}

    def test_string_arguments(self):
        tc = _parse_tool_call({"id": "call_1", "name": "func", "arguments": '{"x": 1}'})
        assert tc.arguments == {"x": 1}

    def test_invalid_string_arguments(self):
        tc = _parse_tool_call({"id": "call_1", "name": "func", "arguments": "not json"})
        assert tc.arguments == {}


class TestParseToolCallChunk:
    def test_flat_format(self):
        tc = _parse_tool_call_chunk(
            {"id": "call_1", "name": "func", "arguments": '{"x":', "index": 0}
        )
        assert tc.name == "func"
        assert tc.arguments == '{"x":'

    def test_function_format(self):
        tc = _parse_tool_call_chunk(
            {
                "id": "call_1",
                "function": {"name": "func", "arguments": '{"x": 1}'},
                "index": 0,
            }
        )
        assert tc.name == "func"
        assert tc.arguments == '{"x": 1}'

    def test_dict_arguments_converted(self):
        tc = _parse_tool_call_chunk(
            {"id": "call_1", "name": "func", "arguments": {"x": 1}, "index": 0}
        )
        assert tc.arguments == '{"x": 1}'


# ============================================================================
# Test: Client initialization
# ============================================================================


class TestUiPathNormalizedClientInit:
    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_default_settings(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        client = UiPathNormalizedClient(model_name="gpt-4o")
        assert client._model_name == "gpt-4o"
        mock_get_settings.assert_called_once()

    def test_custom_settings(self):
        settings = MagicMock()
        settings.build_auth_pipeline.return_value = None

        client = UiPathNormalizedClient(model_name="gpt-4o", client_settings=settings)
        assert client._client_settings is settings

    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_has_completions_namespace(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        client = UiPathNormalizedClient(model_name="gpt-4o")
        assert hasattr(client, "completions")
        assert isinstance(client.completions, Completions)

    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_has_embeddings_namespace(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        client = UiPathNormalizedClient(model_name="gpt-4o")
        from uipath.llm_client.clients.normalized.embeddings import Embeddings

        assert hasattr(client, "embeddings")
        assert isinstance(client.embeddings, Embeddings)

    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_completions_api_config(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        client = UiPathNormalizedClient(model_name="gpt-4o")
        assert client._completions_api_config.api_type == "completions"
        assert client._completions_api_config.routing_mode == "normalized"
        assert client._completions_api_config.freeze_base_url is True

    @patch(f"{_CLIENT_MODULE}.get_default_client_settings")
    def test_embeddings_api_config(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.build_auth_pipeline.return_value = None
        mock_get_settings.return_value = mock_settings

        client = UiPathNormalizedClient(model_name="gpt-4o")
        assert client._embeddings_api_config.api_type == "embeddings"
        assert client._embeddings_api_config.routing_mode == "normalized"
        assert client._embeddings_api_config.freeze_base_url is True


# ============================================================================
# Test: Completions.create (sync, non-streaming)
# ============================================================================


class TestCompletionsCreate:
    def test_basic_create(self, mock_sync_client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_COMPLETION_RESPONSE
        mock_sync_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        result = completions.create(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content == "Hello! How can I help you?"
        mock_sync_client.request.assert_called_once()
        call_kwargs = mock_sync_client.request.call_args
        body = call_kwargs.kwargs["json"]
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    def test_create_with_params(self, mock_sync_client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_COMPLETION_RESPONSE
        mock_sync_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            stop=["END"],
            n=2,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )

        body = mock_sync_client.request.call_args.kwargs["json"]
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.9
        assert body["stop"] == ["END"]
        assert body["n"] == 2
        assert body["presence_penalty"] == 0.1
        assert body["frequency_penalty"] == 0.2

    def test_create_omits_none_params(self, mock_sync_client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_COMPLETION_RESPONSE
        mock_sync_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        completions.create(
            messages=[{"role": "user", "content": "Hello"}],
        )

        body = mock_sync_client.request.call_args.kwargs["json"]
        assert "max_tokens" not in body
        assert "temperature" not in body
        assert "stop" not in body

    def test_create_with_tools(self, mock_sync_client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_TOOL_CALL_RESPONSE
        mock_sync_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        result = completions.create(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            tool_choice="auto",
        )

        body = mock_sync_client.request.call_args.kwargs["json"]
        assert "tools" in body
        assert body["tool_choice"] == "auto"
        assert len(result.choices[0].message.tool_calls) == 1

    def test_create_with_response_format(self, mock_sync_client):
        class MyOutput(BaseModel):
            answer: str

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"answer": "42"}'}}],
            "usage": {},
        }
        mock_sync_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        result = completions.create(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            response_format=MyOutput,
        )

        body = mock_sync_client.request.call_args.kwargs["json"]
        assert "response_format" in body
        assert body["response_format"]["type"] == "json_schema"
        assert result.choices[0].message.parsed is not None
        assert result.choices[0].message.parsed.answer == "42"

    def test_create_with_kwargs(self, mock_sync_client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_COMPLETION_RESPONSE
        mock_sync_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            reasoning={"effort": "high"},
        )

        body = mock_sync_client.request.call_args.kwargs["json"]
        assert body["reasoning"] == {"effort": "high"}


# ============================================================================
# Test: Completions.stream (sync, streaming)
# ============================================================================


class TestCompletionsStream:
    def test_stream_yields_chunks(self, mock_sync_client):
        sse_lines = [f"data: {json.dumps(chunk)}" for chunk in SAMPLE_STREAM_CHUNKS]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(sse_lines)
        mock_sync_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_sync_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        chunks = list(
            completions.stream(
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

        assert len(chunks) == 3
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world!"
        assert chunks[2].choices[0].finish_reason == "stop"

    def test_stream_skips_invalid_json(self, mock_sync_client):
        lines = [
            "data: {invalid json",
            f"data: {json.dumps(SAMPLE_STREAM_CHUNKS[0])}",
            "",  # empty line
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)
        mock_sync_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_sync_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        chunks = list(
            completions.stream(
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

        assert len(chunks) == 1

    def test_stream_skips_empty_id(self, mock_sync_client):
        lines = [
            f"data: {json.dumps({'id': '', 'choices': []})}",
            f"data: {json.dumps(SAMPLE_STREAM_CHUNKS[0])}",
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)
        mock_sync_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_sync_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        chunks = list(
            completions.stream(
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

        assert len(chunks) == 1

    def test_stream_sets_stream_flag(self, mock_sync_client):
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter([])
        mock_sync_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_sync_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        client_obj = MagicMock()
        client_obj._sync_client = mock_sync_client

        completions = Completions(client_obj)
        list(
            completions.stream(
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

        call_kwargs = mock_sync_client.stream.call_args
        body = call_kwargs.kwargs["json"]
        assert body["stream"] is True


# ============================================================================
# Test: Completions.acreate (async, non-streaming)
# ============================================================================


class TestAsyncCompletionsCreate:
    @pytest.mark.asyncio
    async def test_basic_acreate(self, mock_async_client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_COMPLETION_RESPONSE
        mock_async_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._async_client = mock_async_client

        completions = Completions(client_obj)
        result = await completions.acreate(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content == "Hello! How can I help you?"


# ============================================================================
# Test: Embeddings
# ============================================================================


class TestEmbeddingsCreate:
    def test_basic_create(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_EMBEDDING_RESPONSE

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._embedding_sync_client = mock_client

        from uipath.llm_client.clients.normalized.embeddings import Embeddings

        embeddings = Embeddings(client_obj)
        result = embeddings.create(input=["Hello world", "Goodbye"])

        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 2
        assert result.data[0].embedding == [0.1, 0.2, 0.3]

        body = mock_client.request.call_args.kwargs["json"]
        assert body["input"] == ["Hello world", "Goodbye"]

    def test_string_input_wrapped(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_EMBEDDING_RESPONSE

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._embedding_sync_client = mock_client

        from uipath.llm_client.clients.normalized.embeddings import Embeddings

        embeddings = Embeddings(client_obj)
        embeddings.create(input="Hello world")

        body = mock_client.request.call_args.kwargs["json"]
        assert body["input"] == ["Hello world"]


class TestAsyncEmbeddingsCreate:
    @pytest.mark.asyncio
    async def test_basic_acreate(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_EMBEDDING_RESPONSE

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._embedding_async_client = mock_client

        from uipath.llm_client.clients.normalized.embeddings import Embeddings

        embeddings = Embeddings(client_obj)
        result = await embeddings.acreate(input=["Hello world"])

        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 2


# ============================================================================
# Test: Type models
# ============================================================================


class TestTypeModels:
    def test_usage_defaults(self):
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_tool_call(self):
        tc = ToolCall(id="call_1", name="func", arguments={"x": 1})
        assert tc.id == "call_1"
        assert tc.name == "func"
        assert tc.arguments == {"x": 1}

    def test_tool_call_chunk(self):
        tc = ToolCallChunk(id="call_1", name="func", arguments='{"x":', index=0)
        assert tc.arguments == '{"x":'

    def test_message_defaults(self):
        msg = Message()
        assert msg.role == "assistant"
        assert msg.content == ""
        assert msg.tool_calls == []
        assert msg.parsed is None

    def test_delta_defaults(self):
        delta = Delta()
        assert delta.role is None
        assert delta.content == ""
        assert delta.tool_calls == []

    def test_choice_defaults(self):
        choice = Choice()
        assert choice.index == 0
        assert choice.finish_reason is None

    def test_chat_completion_defaults(self):
        cc = ChatCompletion()
        assert cc.id == ""
        assert cc.choices == []
        assert cc.usage.prompt_tokens == 0

    def test_embedding_data(self):
        ed = EmbeddingData(embedding=[0.1, 0.2], index=0)
        assert ed.embedding == [0.1, 0.2]

    def test_embedding_response(self):
        er = EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.1], index=0)],
            model="test-model",
        )
        assert len(er.data) == 1
        assert er.model == "test-model"


# ============================================================================
# Test: Request body building
# ============================================================================


class TestBuildRequest:
    def test_minimal_request(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert body == {"messages": [{"role": "user", "content": "Hi"}]}

    def test_stream_flag(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        assert body["stream"] is True

    def test_all_optional_params(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
            stop=["END"],
            n=3,
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.7
        assert body["stop"] == ["END"]
        assert body["n"] == 3
        assert body["top_p"] == 0.9
        assert body["presence_penalty"] == 0.5
        assert body["frequency_penalty"] == 0.3

    def test_with_tools(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"name": "func", "description": "d", "parameters": {}}],
            tool_choice="auto",
        )
        assert len(body["tools"]) == 1
        assert body["tool_choice"] == "auto"

    def test_with_response_format(self):
        class MyModel(BaseModel):
            x: int

        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            response_format=MyModel,
        )
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "MyModel"

    def test_kwargs_merged(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            custom_param="value",
        )
        assert body["custom_param"] == "value"

    def test_openai_specific_params(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            seed=42,
            logit_bias={"123": -100},
            logprobs=True,
            top_logprobs=5,
            parallel_tool_calls=False,
            reasoning_effort="high",
            reasoning={"effort": "high"},
        )
        assert body["seed"] == 42
        assert body["logit_bias"] == {"123": -100}
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5
        assert body["parallel_tool_calls"] is False
        assert body["reasoning_effort"] == "high"
        assert body["reasoning"] == {"effort": "high"}

    def test_anthropic_specific_params(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            top_k=40,
            thinking={"type": "enabled", "budget_tokens": 1000},
        )
        assert body["top_k"] == 40
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1000}

    def test_google_specific_params(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            thinking_level="high",
            thinking_budget=2000,
            include_thoughts=True,
            safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
        )
        assert body["thinking_level"] == "high"
        assert body["thinking_budget"] == 2000
        assert body["include_thoughts"] is True
        assert len(body["safety_settings"]) == 1

    def test_shared_params(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            verbosity="low",
        )
        assert body["verbosity"] == "low"

    def test_removed_infra_params_go_through_kwargs(self):
        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            user="user-123",
            service_tier="auto",
            metadata={"request_id": "abc"},
        )
        assert body["user"] == "user-123"
        assert body["service_tier"] == "auto"
        assert body["metadata"] == {"request_id": "abc"}

    def test_pydantic_messages(self):
        class ChatMessage(BaseModel):
            role: str
            content: str

        body = _build_request(
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    def test_mixed_dict_and_pydantic_messages(self):
        class ChatMessage(BaseModel):
            role: str
            content: str

        body = _build_request(
            messages=[
                {"role": "system", "content": "Be brief."},
                ChatMessage(role="user", content="Hi"),
            ],
        )
        assert body["messages"] == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]

    def test_pydantic_message_with_none_fields_excluded(self):
        class ChatMessage(BaseModel):
            role: str
            content: str
            name: str | None = None

        body = _build_request(
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    def test_pydantic_tool_in_tools_list(self):
        class GetWeather(BaseModel):
            """Get weather for a city."""

            city: str

        body = _build_request(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[GetWeather],
            tool_choice="auto",
        )
        assert body["tools"][0]["name"] == "GetWeather"
        assert "city" in body["tools"][0]["parameters"]["properties"]


# ============================================================================
# Test: Real-world response shapes (from captured API payloads)
# ============================================================================


class TestRealWorldResponses:
    """Tests using actual response shapes observed from the normalized API."""

    def test_gpt4o_basic(self):
        data = {
            "id": "chatcmpl-DQdh09fdBuc8LPCkDqhJKgrQy3IN8",
            "model": "gpt-4o-2024-11-20",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "Hello.", "role": "assistant"},
                }
            ],
            "created": 1775241562,
            "usage": {
                "completion_tokens": 3,
                "prompt_tokens": 14,
                "total_tokens": 17,
                "cache_read_input_tokens": 0,
                "thoughts_tokens": 0,
            },
        }
        result = _parse_response(data)
        assert result.object == "chat.completion"
        assert result.model == "gpt-4o-2024-11-20"
        assert result.choices[0].message.content == "Hello."
        assert result.usage.thoughts_tokens == 0

    def test_gemini_with_avg_logprobs_and_signature(self):
        data = {
            "id": "gemini-123",
            "model": "gemini-2.5-flash",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "avg_logprobs": -0.123,
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "signature": "abc123signature",
                        "tool_calls": [
                            {"id": "call_1", "name": "get_weather", "arguments": {"city": "London"}}
                        ],
                    },
                }
            ],
            "created": 1775241600,
            "usage": {
                "completion_tokens": 5,
                "prompt_tokens": 31,
                "total_tokens": 130,
                "cache_read_input_tokens": 0,
                "thoughts_tokens": 94,
                "request_processing_tier": "ON_DEMAND",
            },
        }
        result = _parse_response(data)
        assert result.choices[0].avg_logprobs == -0.123
        assert result.choices[0].message.signature == "abc123signature"
        assert result.choices[0].message.tool_calls[0].name == "get_weather"
        assert result.usage.thoughts_tokens == 94
        assert result.usage.request_processing_tier == "ON_DEMAND"

    def test_anthropic_with_thinking(self):
        data = {
            "id": "anthropic-456",
            "model": "claude-haiku-4-5",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "15 + 27 = 42",
                        "role": "assistant",
                        "signature": "ErACsignature",
                        "thinking": "This is a straightforward arithmetic problem.\n15 + 27 = 42",
                    },
                }
            ],
            "created": 1775241700,
            "usage": {
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "completion_tokens": 10,
                "prompt_tokens": 14,
                "total_tokens": 24,
            },
        }
        result = _parse_response(data)
        assert (
            result.choices[0].message.thinking
            == "This is a straightforward arithmetic problem.\n15 + 27 = 42"
        )
        assert result.choices[0].message.signature == "ErACsignature"
        assert result.choices[0].message.content == "15 + 27 = 42"
        assert result.usage.cache_creation_input_tokens == 0

    def test_gpt5_with_reasoning_usage(self):
        data = {
            "id": "chatcmpl-gpt5",
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "100", "role": "assistant"},
                }
            ],
            "created": 1775241800,
            "usage": {
                "completion_tokens": 2,
                "prompt_tokens": 20,
                "total_tokens": 22,
                "cache_read_input_tokens": 0,
                "thoughts_tokens": 50,
                "request_processing_tier": "ON_DEMAND",
            },
        }
        result = _parse_response(data)
        assert result.usage.thoughts_tokens == 50
        assert result.usage.request_processing_tier == "ON_DEMAND"

    def test_embedding_response_real_shape(self):
        """Embeddings only return prompt_tokens and total_tokens."""
        data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
        from uipath.llm_client.clients.normalized.embeddings import _parse_embedding_response

        result = _parse_embedding_response(data)
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[0].index == 0  # auto-assigned
        assert result.usage.prompt_tokens == 2
        assert result.usage.completion_tokens == 0  # default

    def test_tool_call_arguments_always_dict(self):
        """Normalized API always returns arguments as dict, not string."""
        data = {
            "id": "tc-test",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "get_weather",
                                "arguments": {"city": "London"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        result = _parse_response(data)
        tc = result.choices[0].message.tool_calls[0]
        assert isinstance(tc.arguments, dict)
        assert tc.arguments == {"city": "London"}


# ============================================================================
# Test: SSE parsing helpers
# ============================================================================


def _make_generator(lines: list[str]) -> Generator[str, None, None]:
    """Helper to create a generator from a list of strings."""
    yield from lines


class TestIterSSE:
    def test_parses_multiple_data_lines(self):
        lines = [
            f"data: {json.dumps({'id': 'chunk-1', 'choices': [{'delta': {'content': 'Hi'}}]})}",
            f"data: {json.dumps({'id': 'chunk-2', 'choices': [{'delta': {'content': ' there'}}]})}",
        ]
        results = list(_iter_sse(_make_generator(lines)))
        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[1]["id"] == "chunk-2"

    def test_handles_done_marker(self):
        lines = [
            f"data: {json.dumps({'id': 'chunk-1', 'choices': [{'delta': {'content': 'Hi'}}]})}",
            "data: [DONE]",
        ]
        results = list(_iter_sse(_make_generator(lines)))
        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"

    def test_skips_invalid_json(self):
        lines = [
            "data: not-valid-json",
            f"data: {json.dumps({'id': 'chunk-1', 'choices': []})}",
        ]
        results = list(_iter_sse(_make_generator(lines)))
        assert len(results) == 1

    def test_skips_empty_id(self):
        lines = [
            f"data: {json.dumps({'id': '', 'choices': []})}",
            f"data: {json.dumps({'id': 'chunk-1', 'choices': []})}",
        ]
        results = list(_iter_sse(_make_generator(lines)))
        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"

    def test_handles_lines_without_data_prefix(self):
        lines = [
            json.dumps({"id": "chunk-1", "choices": [{"delta": {"content": "Hi"}}]}),
        ]
        results = list(_iter_sse(_make_generator(lines)))
        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"


class TestAiterSSE:
    @pytest.mark.asyncio
    async def test_parses_multiple_data_lines(self):
        async def async_lines():
            yield f"data: {json.dumps({'id': 'chunk-1', 'choices': [{'delta': {'content': 'Hi'}}]})}"
            yield f"data: {json.dumps({'id': 'chunk-2', 'choices': [{'delta': {'content': ' there'}}]})}"

        results = []
        async for data in _aiter_sse(async_lines()):
            results.append(data)
        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[1]["id"] == "chunk-2"

    @pytest.mark.asyncio
    async def test_handles_done_marker(self):
        async def async_lines():
            yield f"data: {json.dumps({'id': 'chunk-1', 'choices': []})}"
            yield "data: [DONE]"

        results = []
        async for data in _aiter_sse(async_lines()):
            results.append(data)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_skips_invalid_json(self):
        async def async_lines():
            yield "data: {bad json"
            yield f"data: {json.dumps({'id': 'chunk-1', 'choices': []})}"

        results = []
        async for data in _aiter_sse(async_lines()):
            results.append(data)
        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"


# ============================================================================
# Test: Embeddings.acreate (async)
# ============================================================================


class TestEmbeddingsAcreateExtended:
    @pytest.mark.asyncio
    async def test_acreate_string_input_wrapped(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_EMBEDDING_RESPONSE

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._embedding_async_client = mock_client

        from uipath.llm_client.clients.normalized.embeddings import Embeddings

        embeddings = Embeddings(client_obj)
        result = await embeddings.acreate(input="single string")

        assert isinstance(result, EmbeddingResponse)
        body = mock_client.request.call_args.kwargs["json"]
        assert body["input"] == ["single string"]

    @pytest.mark.asyncio
    async def test_acreate_with_kwargs(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_EMBEDDING_RESPONSE

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response

        client_obj = MagicMock()
        client_obj._embedding_async_client = mock_client

        from uipath.llm_client.clients.normalized.embeddings import Embeddings

        embeddings = Embeddings(client_obj)
        await embeddings.acreate(input=["hello"], dimensions=256)

        body = mock_client.request.call_args.kwargs["json"]
        assert body["dimensions"] == 256
