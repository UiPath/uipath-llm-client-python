"""Response types for the UiPath Normalized API."""

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallChunk(BaseModel):
    id: str = ""
    name: str = ""
    arguments: str = ""
    index: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    thoughts_tokens: int = 0
    request_processing_tier: str | None = None


class Message(BaseModel):
    role: str = "assistant"
    content: str | None = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    signature: str | None = None
    thinking: str | None = None
    # Structured output (populated client-side when output_format is used)
    parsed: Any = None


class Delta(BaseModel):
    role: str | None = None
    content: str | None = ""
    tool_calls: list[ToolCallChunk] = Field(default_factory=list)


class Choice(BaseModel):
    index: int = 0
    message: Message = Field(default_factory=Message)
    finish_reason: str | None = None
    avg_logprobs: float | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: str | None = None
    avg_logprobs: float | None = None


class ChatCompletion(BaseModel):
    id: str = ""
    object: str = ""
    created: int = 0
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class ChatCompletionChunk(BaseModel):
    id: str = ""
    object: str = ""
    created: int | str = 0
    model: str = ""
    choices: list[StreamChoice] = Field(default_factory=list)
    usage: Usage | None = None


class EmbeddingData(BaseModel):
    embedding: list[float] = Field(default_factory=list)
    index: int = 0


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
