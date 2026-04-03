"""Completions endpoint for the UiPath Normalized API."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from uipath.llm_client.clients.normalized.types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Delta,
    Message,
    StreamChoice,
    ToolCall,
    ToolCallChunk,
    Usage,
)

try:
    from typing import is_typeddict
except ImportError:
    from typing_extensions import is_typeddict

# ---------------------------------------------------------------------------
# Public input types
# ---------------------------------------------------------------------------

ResponseFormatType = Union[type[BaseModel], type, dict[str, Any]]
"""Response format: Pydantic model, TypedDict, or raw dict (e.g. {"type": "json_object"})."""

ToolType = Union[dict[str, Any], type[BaseModel], Callable[..., Any]]
"""Tool definition: dict (raw schema), Pydantic model, or callable."""

ToolChoiceType = Union[str, dict[str, Any]]
"""Tool choice: "auto", "required", "none", a tool name, or a dict."""

MessageType = Union[dict[str, Any], BaseModel]
"""A single message: dict with role/content or a Pydantic model with those fields."""


def _normalize_messages(messages: Sequence[MessageType]) -> list[dict[str, Any]]:
    """Convert a sequence of messages (dicts or pydantic models) to dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        elif isinstance(msg, BaseModel):
            result.append(msg.model_dump(exclude_none=True))
        else:
            result.append(dict(msg))  # type: ignore[arg-type]
    return result


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _json_schema_from_type(tp: type) -> dict[str, Any]:
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        return {"type": "array", "items": _json_schema_from_type(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object"}
    simple = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return {"type": simple.get(tp, "object")}


def _build_response_format(
    response_format: ResponseFormatType, strict: bool | None = None
) -> dict[str, Any]:
    if isinstance(response_format, dict):
        if "type" in response_format:
            return response_format
        return {"type": "json_schema", "json_schema": response_format}

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        js: dict[str, Any] = {
            "name": response_format.__name__,
            "schema": response_format.model_json_schema(),
        }
        if strict is not False:
            js["strict"] = True
        return {"type": "json_schema", "json_schema": js}

    if isinstance(response_format, type) and is_typeddict(response_format):
        hints = get_type_hints(response_format)
        properties = {name: _json_schema_from_type(tp) for name, tp in hints.items()}
        js = {
            "name": response_format.__name__,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
        }
        if strict is not False:
            js["strict"] = True
        return {"type": "json_schema", "json_schema": js}

    if isinstance(response_format, type):
        js = {
            "name": response_format.__name__,
            "schema": _json_schema_from_type(response_format),
        }
        if strict is True:
            js["strict"] = True
        return {"type": "json_schema", "json_schema": js}

    raise TypeError(f"Unsupported response_format type: {type(response_format)}")


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


def _build_tool_definition(tool: ToolType) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool

    if isinstance(tool, type) and issubclass(tool, BaseModel):
        schema = tool.model_json_schema()
        schema.pop("title", None)
        return {"name": tool.__name__, "description": tool.__doc__ or "", "parameters": schema}

    if callable(tool):
        import inspect

        sig = inspect.signature(tool)
        hints = get_type_hints(tool)
        properties = {name: _json_schema_from_type(hints.get(name, str)) for name in sig.parameters}
        required = [
            name for name, p in sig.parameters.items() if p.default is inspect.Parameter.empty
        ]
        return {
            "name": tool.__name__,
            "description": tool.__doc__ or "",
            "parameters": {"type": "object", "properties": properties, "required": required},
        }

    raise TypeError(f"Unsupported tool type: {type(tool)}")


def _resolve_tool_choice(
    tool_choice: ToolChoiceType, tools: list[dict[str, Any]]
) -> dict[str, Any] | str:
    if isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice in ("auto", "required", "none"):
        return tool_choice
    tool_names = [t.get("name", "") for t in tools]
    if tool_choice in tool_names:
        return {"type": "tool", "name": tool_choice}
    return "auto"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    arguments = raw.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return ToolCall(id=raw.get("id", ""), name=raw.get("name", ""), arguments=arguments)


def _parse_tool_call_chunk(raw: dict[str, Any]) -> ToolCallChunk:
    if "function" in raw:
        name = raw["function"].get("name", "")
        args = raw["function"].get("arguments", "")
    else:
        name = raw.get("name", "")
        args = raw.get("arguments", "")
    if isinstance(args, dict):
        args = json.dumps(args) if args else ""
    return ToolCallChunk(id=raw.get("id", ""), name=name, arguments=args, index=raw.get("index", 0))


def _parse_structured_output(content: str, response_format: ResponseFormatType) -> Any:
    try:
        parsed_json = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return response_format.model_validate(parsed_json)
    return parsed_json


def _parse_response(
    data: dict[str, Any], response_format: ResponseFormatType | None = None
) -> ChatCompletion:
    usage = Usage(**data.get("usage", {}))
    choices: list[Choice] = []
    for choice_data in data.get("choices", []):
        msg_data = choice_data.get("message", {})
        tool_calls = [_parse_tool_call(tc) for tc in msg_data.get("tool_calls", [])]
        content = msg_data.get("content", "")
        parsed = (
            _parse_structured_output(content, response_format)
            if response_format and content
            else None
        )
        message = Message(
            role=msg_data.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls,
            signature=msg_data.get("signature"),
            thinking=msg_data.get("thinking"),
            parsed=parsed,
        )
        choices.append(
            Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                avg_logprobs=choice_data.get("avg_logprobs"),
            )
        )
    return ChatCompletion(
        id=data.get("id", ""),
        object=data.get("object", ""),
        created=data.get("created", 0),
        model=data.get("model", ""),
        choices=choices,
        usage=usage,
    )


def _parse_stream_chunk(data: dict[str, Any]) -> ChatCompletionChunk:
    usage = Usage(**data["usage"]) if data.get("usage") else None
    choices: list[StreamChoice] = []
    for choice_data in data.get("choices", []):
        delta_data = choice_data.get("delta", choice_data.get("message", {}))
        tool_calls = [_parse_tool_call_chunk(tc) for tc in delta_data.get("tool_calls", [])]
        delta = Delta(
            role=delta_data.get("role"),
            content=delta_data.get("content", ""),
            tool_calls=tool_calls,
        )
        choices.append(
            StreamChoice(
                index=choice_data.get("index", 0),
                delta=delta,
                finish_reason=choice_data.get("finish_reason"),
                avg_logprobs=choice_data.get("avg_logprobs"),
            )
        )
    return ChatCompletionChunk(
        id=data.get("id", ""),
        object=data.get("object", ""),
        created=data.get("created", 0),
        model=data.get("model", ""),
        choices=choices,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _build_request(
    *,
    messages: Sequence[MessageType],
    stream: bool = False,
    tools: Sequence[ToolType] | None = None,
    tool_choice: ToolChoiceType | None = None,
    response_format: ResponseFormatType | None = None,
    strict: bool | None = None,
    # Common
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop: list[str] | str | None = None,
    n: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    # OpenAI
    logit_bias: dict[str, int] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    parallel_tool_calls: bool | None = None,
    # OpenAI reasoning (o1/o3/gpt-5)
    reasoning_effort: str | None = None,
    reasoning: dict[str, Any] | None = None,
    # Anthropic
    thinking: dict[str, Any] | None = None,
    # Google
    thinking_level: str | None = None,
    thinking_budget: int | None = None,
    include_thoughts: bool | None = None,
    safety_settings: list[dict[str, Any]] | None = None,
    # Shared
    verbosity: str | None = None,
    # Aliases (resolve to canonical names above)
    stop_sequences: list[str] | None = None,
    max_output_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    candidate_count: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the request body for a chat completion."""
    # Resolve aliases
    max_tokens = max_tokens or max_output_tokens or max_completion_tokens
    stop = stop or stop_sequences
    n = n or candidate_count

    body: dict[str, Any] = {"messages": _normalize_messages(messages)}

    if stream:
        body["stream"] = True

    optional: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop": stop,
        "n": n,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "parallel_tool_calls": parallel_tool_calls,
        "reasoning_effort": reasoning_effort,
        "reasoning": reasoning,
        "thinking": thinking,
        "thinking_level": thinking_level,
        "thinking_budget": thinking_budget,
        "include_thoughts": include_thoughts,
        "safety_settings": safety_settings,
        "verbosity": verbosity,
    }
    body.update({k: v for k, v in optional.items() if v is not None})

    if tools is not None:
        body["tools"] = [_build_tool_definition(t) for t in tools]
        if tool_choice is not None:
            body["tool_choice"] = _resolve_tool_choice(tool_choice, body["tools"])

    if response_format is not None:
        body["response_format"] = _build_response_format(response_format, strict=strict)

    body.update(kwargs)
    return body


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _iter_sse(lines: Generator[str, None, None]) -> Generator[dict[str, Any], None, None]:
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in data and not data["id"]:
            continue
        yield data


async def _aiter_sse(lines: AsyncGenerator[str, None]) -> AsyncGenerator[dict[str, Any], None]:
    async for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in data and not data["id"]:
            continue
        yield data


# ---------------------------------------------------------------------------
# Completions namespace
# ---------------------------------------------------------------------------


class Completions:
    """``client.completions`` — create, acreate, stream, astream."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(
        self,
        *,
        messages: Sequence[MessageType],
        tools: Sequence[ToolType] | None = None,
        tool_choice: ToolChoiceType | None = None,
        response_format: ResponseFormatType | None = None,
        strict: bool | None = None,
        # Common
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        # OpenAI
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        # Anthropic
        thinking: dict[str, Any] | None = None,
        # Google
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        include_thoughts: bool | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        # Shared
        verbosity: str | None = None,
        # Aliases
        stop_sequences: list[str] | None = None,
        max_output_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        candidate_count: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion (sync)."""
        body = _build_request(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            strict=strict,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            reasoning=reasoning,
            thinking=thinking,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
            safety_settings=safety_settings,
            verbosity=verbosity,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            candidate_count=candidate_count,
            **kwargs,
        )
        response = self._client._sync_client.request("POST", "/", json=body)
        response.raise_for_status()
        return _parse_response(response.json(), response_format=response_format)

    async def acreate(
        self,
        *,
        messages: Sequence[MessageType],
        tools: Sequence[ToolType] | None = None,
        tool_choice: ToolChoiceType | None = None,
        response_format: ResponseFormatType | None = None,
        strict: bool | None = None,
        # Common
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        # OpenAI
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        # Anthropic
        thinking: dict[str, Any] | None = None,
        # Google
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        include_thoughts: bool | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        # Shared
        verbosity: str | None = None,
        # Aliases
        stop_sequences: list[str] | None = None,
        max_output_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        candidate_count: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion (async)."""
        body = _build_request(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            strict=strict,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            reasoning=reasoning,
            thinking=thinking,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
            safety_settings=safety_settings,
            verbosity=verbosity,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            candidate_count=candidate_count,
            **kwargs,
        )
        response = await self._client._async_client.request("POST", "/", json=body)
        response.raise_for_status()
        return _parse_response(response.json(), response_format=response_format)

    def stream(
        self,
        *,
        messages: Sequence[MessageType],
        tools: Sequence[ToolType] | None = None,
        tool_choice: ToolChoiceType | None = None,
        response_format: ResponseFormatType | None = None,
        strict: bool | None = None,
        # Common
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        # OpenAI
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        # Anthropic
        thinking: dict[str, Any] | None = None,
        # Google
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        include_thoughts: bool | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        # Shared
        verbosity: str | None = None,
        # Aliases
        stop_sequences: list[str] | None = None,
        max_output_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        candidate_count: int | None = None,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Stream chat completion chunks (sync)."""
        body = _build_request(
            messages=messages,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            strict=strict,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            reasoning=reasoning,
            thinking=thinking,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
            safety_settings=safety_settings,
            verbosity=verbosity,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            candidate_count=candidate_count,
            **kwargs,
        )
        with self._client._sync_client.stream("POST", "/", json=body) as response:
            response.raise_for_status()
            for data in _iter_sse(response.iter_lines()):
                yield _parse_stream_chunk(data)

    async def astream(
        self,
        *,
        messages: Sequence[MessageType],
        tools: Sequence[ToolType] | None = None,
        tool_choice: ToolChoiceType | None = None,
        response_format: ResponseFormatType | None = None,
        strict: bool | None = None,
        # Common
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        n: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        # OpenAI
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        # Anthropic
        thinking: dict[str, Any] | None = None,
        # Google
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        include_thoughts: bool | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        # Shared
        verbosity: str | None = None,
        # Aliases
        stop_sequences: list[str] | None = None,
        max_output_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        candidate_count: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream chat completion chunks (async)."""
        body = _build_request(
            messages=messages,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            strict=strict,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            reasoning=reasoning,
            thinking=thinking,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
            safety_settings=safety_settings,
            verbosity=verbosity,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            candidate_count=candidate_count,
            **kwargs,
        )
        async with self._client._async_client.stream("POST", "/", json=body) as response:
            response.raise_for_status()
            async for data in _aiter_sse(response.aiter_lines()):
                yield _parse_stream_chunk(data)
