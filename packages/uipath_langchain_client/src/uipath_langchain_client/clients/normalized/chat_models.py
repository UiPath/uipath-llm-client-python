"""
Normalized Chat Model for UiPath LangChain Client

This module provides a provider-agnostic chat model that uses UiPath's normalized API.
The normalized API provides a consistent interface across all LLM providers (OpenAI,
Google, Anthropic, etc.), making it easy to switch providers without code changes.

The normalized API supports:
- Standard chat completions with messages
- Tool/function calling with automatic format conversion
- Streaming responses (sync and async)
- Extended thinking/reasoning parameters for supported models

Example:
    >>> from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
    >>> from uipath_langchain_client.settings import get_default_client_settings
    >>>
    >>> settings = get_default_client_settings()
    >>> chat = UiPathChat(
    ...     model="gpt-4o-2024-11-20",
    ...     settings=settings,
    ... )
    >>> response = chat.invoke("Hello!")
"""

import json
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from functools import partial
from typing import Any, Literal, Union, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.base import (
    LanguageModelInput,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    InputTokenDetails,
    OutputTokenDetails,
    ToolCallChunk,
    UsageMetadata,
)
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import AliasChoices, BaseModel, Field

from uipath.llm_client.utils.model_family import is_anthropic_model_name
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.settings import ApiType, RoutingMode, UiPathAPIConfig

_DictOrPydanticClass = Union[dict[str, Any], type[BaseModel], type]
_DictOrPydantic = Union[dict[str, Any], BaseModel]


def _oai_structured_outputs_parser(ai_msg: AIMessage, schema: type[BaseModel]) -> BaseModel:
    if not ai_msg.content:
        raise ValueError("Expected non-empty content from model.")
    content = ai_msg.content
    if isinstance(content, list):
        # Extract the first text block from content parts
        content = next((c for c in content if isinstance(c, str)), str(content[0]))
    parsed = json.loads(content)
    return schema.model_validate(parsed)


def _build_normalized_response_format(
    schema: _DictOrPydanticClass, strict: bool | None = None
) -> dict[str, Any]:
    """Build response_format for the normalized API from a schema."""
    if isinstance(schema, dict):
        return {"type": "json_schema", "json_schema": schema}

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        rf: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": json_schema,
            },
        }
        if strict is not None:
            rf["json_schema"]["strict"] = strict
        return rf

    # TypedDict or other type — convert via openai tool schema
    tool_schema = convert_to_openai_tool(schema)
    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": tool_schema["function"]["name"],
            "schema": tool_schema["function"]["parameters"],
        },
    }
    if strict is not None:
        rf["json_schema"]["strict"] = strict
    return rf


class UiPathChat(UiPathBaseChatModel):
    """LangChain chat model using UiPath's normalized (provider-agnostic) API.

    This model provides a consistent interface across all LLM providers supported
    by UiPath AgentHub and LLM Gateway. It automatically handles message format
    conversion, tool calling, and streaming for any supported provider.

    Attributes:
        model_name: The model identifier (e.g., "gpt-4o-2024-11-20", "gemini-2.5-flash").
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0.0 to 2.0).
        stop: Stop sequences to end generation.
        n: Number of completions to generate.
        top_p: Nucleus sampling probability mass.
        presence_penalty: Penalty for repeated tokens (-2.0 to 2.0).
        frequency_penalty: Penalty based on token frequency (-2.0 to 2.0).

    Extended Thinking (model-specific):
        reasoning: OpenAI o1/o3 reasoning config {"effort": "low"|"medium"|"high"}.
        reasoning_effort: OpenAI reasoning effort level.
        thinking: Anthropic Claude thinking config {"type": "enabled", "budget_tokens": N}.
        thinking_level: Gemini thinking level.
        thinking_budget: Gemini thinking token budget.
        include_thoughts: Whether to include thinking in Gemini responses.

    Example:
        >>> chat = UiPathChat(
        ...     model="gpt-4o-2024-11-20",
        ...     settings=settings,
        ...     temperature=0.7,
        ...     max_tokens=1000,
        ... )
        >>> response = chat.invoke("Explain machine learning.")
    """

    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.NORMALIZED,
        freeze_base_url=True,
    )

    # Common
    max_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max_tokens", "max_output_tokens", "max_completion_tokens"),
    )
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | str | None = Field(
        default=None,
        validation_alias=AliasChoices("stop", "stop_sequences"),
    )
    n: int | None = Field(
        default=None,
        validation_alias=AliasChoices("n", "candidate_count"),
    )
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    disabled_params: dict[str, Any] | None = None

    # OpenAI
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None

    # Anthropic
    thinking: dict[str, Any] | None = None

    # Google
    thinking_level: str | None = None
    thinking_budget: int | None = None
    include_thoughts: bool | None = None
    safety_settings: list[dict[str, Any]] | None = None

    # Shared
    verbosity: str | None = None

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "uipath-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for the normalized API request."""
        candidates: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            # OpenAI
            "logit_bias": self.logit_bias,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "parallel_tool_calls": self.parallel_tool_calls,
            "reasoning_effort": self.reasoning_effort,
            "reasoning": self.reasoning,
            # Anthropic
            "thinking": self.thinking,
            # Google
            "thinking_level": self.thinking_level,
            "thinking_budget": self.thinking_budget,
            "include_thoughts": self.include_thoughts,
            "safety_settings": self.safety_settings,
            # Shared
            "verbosity": self.verbosity,
        }

        set_fields = self.model_fields_set
        return {
            **{k: v for k, v in candidates.items() if k in set_fields},
            **self.model_kwargs,
        }

    def _get_usage_metadata(self, json_data: dict[str, Any]) -> UsageMetadata:
        return UsageMetadata(
            input_tokens=json_data.get("prompt_tokens", 0),
            output_tokens=json_data.get("completion_tokens", 0),
            total_tokens=json_data.get("total_tokens", 0),
            input_token_details=InputTokenDetails(
                audio=json_data.get("audio_tokens", 0),
                cache_read=json_data.get("cache_read_input_tokens", 0),
                cache_creation=json_data.get("cache_creation_input_tokens", 0),
            ),
            output_token_details=OutputTokenDetails(
                audio=json_data.get("output_audio_tokens", 0),
                reasoning=json_data.get("thoughts_tokens", 0),
            ),
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model with automatic tool choice detection."""
        formatted_tools = [convert_to_openai_function(t, strict=strict) for t in tools]
        tool_names = [tool["name"] for tool in formatted_tools]

        if tool_choice is None:
            tool_choice = "auto"
        elif tool_choice in ["required", "any"]:
            tool_choice = "required"
        elif tool_choice in tool_names:
            pass
        else:
            tool_choice = "auto"

        if tool_choice in ["required", "auto"]:
            tool_choice_object: dict[str, Any] = {
                "type": tool_choice,
            }
        else:
            tool_choice_object = {
                "type": "tool",
                "name": tool_choice,
            }

        bind_kwargs: dict[str, Any] = {
            "tools": formatted_tools,
            "tool_choice": tool_choice_object,
            **kwargs,
        }
        if parallel_tool_calls is not None:
            bind_kwargs["parallel_tool_calls"] = parallel_tool_calls

        return super().bind(**bind_kwargs)

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a Pydantic class, TypedDict, JSON Schema dict,
                or OpenAI function schema.
            method: Either "json_schema" (uses response_format) or "function_calling"
                (uses tool calling to force the schema).
            include_raw: If True, returns dict with 'raw', 'parsed', and 'parsing_error'.
            strict: If True, model output is guaranteed to match the schema exactly.
            **kwargs: Additional arguments passed to bind().

        Returns:
            A Runnable that parses the model output into the given schema.
        """
        if schema is None:
            raise ValueError("schema must be specified.")

        is_pydantic = isinstance(schema, type) and is_basemodel_subclass(schema)

        if method == "function_calling":
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice="any",
                strict=strict,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling", "strict": strict},
                    "schema": schema,
                },
                **kwargs,
            )
            if is_pydantic:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
                **kwargs,
            )
            if is_pydantic:
                from langchain_core.output_parsers import PydanticOutputParser

                output_parser = PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
            else:
                output_parser = JsonOutputParser()
        elif method == "json_schema":
            response_format = _build_normalized_response_format(schema, strict=strict)
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format={
                    "kwargs": {"method": method, "strict": strict},
                    "schema": convert_to_openai_tool(schema),
                },
                **kwargs,
            )
            if is_pydantic:
                output_parser = RunnableLambda(
                    partial(_oai_structured_outputs_parser, schema=cast(type, schema))
                ).with_types(output_type=cast(type, schema))
            else:
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method: '{method}'. "
                "Expected 'function_calling', 'json_mode', or 'json_schema'."
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=lambda x: output_parser.invoke(x["raw"]),
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(
                parsed=lambda _: None,
            )
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnablePassthrough.assign(raw=llm) | parser_with_fallback  # type: ignore[return-value]
        return llm | output_parser  # type: ignore[return-value]

    def _preprocess_request(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Convert LangChain messages to normalized API request format."""
        converted_messages = convert_to_openai_messages(messages)
        for message, converted_message in zip(messages, converted_messages):
            if isinstance(message, AIMessage):
                if isinstance(converted_message["content"], list):
                    converted_message["content"] = [
                        item for item in converted_message["content"] if item["type"] != "tool_call"
                    ]
                    if len(converted_message["content"]) == 0:
                        converted_message["content"] = ""
                if (
                    self.model_name
                    and is_anthropic_model_name(self.model_name)
                    and not converted_message["content"]
                ):
                    converted_message["content"] = "tool_call"
                if "tool_calls" in converted_message:
                    converted_message["tool_calls"] = [
                        {
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "arguments": (
                                tool_call["function"]["arguments"]
                                if isinstance(tool_call["function"]["arguments"], dict)
                                else (
                                    json.loads(tool_call["function"]["arguments"])
                                    if tool_call["function"]["arguments"]
                                    else {}
                                )
                            ),
                        }
                        for tool_call in converted_message["tool_calls"]
                    ]
                if "signature" in message.additional_kwargs:  # required for Gemini models
                    converted_message["signature"] = message.additional_kwargs["signature"]
            elif converted_message["role"] == "tool":
                converted_message["content"] = {
                    "result": converted_message["content"],
                    "call_id": converted_message.pop("tool_call_id"),
                }

        request_body = {
            "messages": converted_messages,
            **self._default_params,
            **kwargs,
        }
        if stop is not None:
            request_body["stop"] = stop

        return request_body

    def _postprocess_response(self, response: dict[str, Any]) -> ChatResult:
        """Convert normalized API response to LangChain ChatResult format."""
        generations = []
        llm_output = {
            "id": response.get("id"),
            "created": response.get("created"),
            "model_name": response.get("model"),
        }
        usage = response.get("usage", {})
        usage_metadata = self._get_usage_metadata(usage)
        for choice in response["choices"]:
            generation_info = {
                "finish_reason": choice.get("finish_reason", ""),
            }
            message = choice["message"]
            generation = ChatGeneration(
                message=AIMessage(
                    content=message.get("content", ""),
                    tool_calls=[
                        {
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "args": tool_call["arguments"],
                        }
                        for tool_call in message.get("tool_calls", [])
                    ],
                    additional_kwargs={},
                    response_metadata={},
                    usage_metadata=usage_metadata,
                ),
                generation_info=generation_info,
            )
            if "signature" in message:  # required for Gemini models
                generation.message.additional_kwargs["signature"] = message["signature"]
            generations.append(generation)
        return ChatResult(
            generations=generations,
            llm_output=llm_output,
        )

    def _uipath_generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request_body = self._preprocess_request(messages, stop=stop, **kwargs)
        response = self.uipath_request(request_body=request_body, raise_status_error=True)
        return self._postprocess_response(response.json())

    async def _uipath_agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request_body = self._preprocess_request(messages, stop=stop, **kwargs)
        response = await self.uipath_arequest(request_body=request_body, raise_status_error=True)
        return self._postprocess_response(response.json())

    def _generate_chunk(self, json_data: dict[str, Any]) -> ChatGenerationChunk:
        generation_info = {
            "id": json_data.get("id"),
            "created": json_data.get("created", ""),
            "model_name": json_data.get("model", ""),
        }
        content = ""
        usage_metadata = None
        tool_call_chunks = []
        if usage := json_data.get("usage", {}):
            usage_metadata = self._get_usage_metadata(usage)
        if choices := json_data.get("choices", []):
            if "finish_reason" in choices[0]:
                generation_info["finish_reason"] = choices[0]["finish_reason"]

            if "delta" in choices[0]:
                content = choices[0]["delta"].get("content", "")
                tool_calls = choices[0]["delta"].get("tool_calls", [])
            elif "message" in choices[0]:
                content = choices[0]["message"].get("content", "")
                tool_calls = choices[0]["message"].get("tool_calls", [])
            else:
                content = choices[0].get("content", "")
                tool_calls = choices[0].get("tool_calls", [])

            for tool_call in tool_calls:
                if "function" in tool_call:
                    name = tool_call["function"].get("name", "")
                    args = tool_call["function"].get("arguments", "")
                else:
                    name = tool_call.get("name", "")
                    args = tool_call.get("arguments", "")
                if args == {}:
                    args = ""
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_call_chunks.append(
                    ToolCallChunk(
                        id=tool_call.get("id", ""),
                        name=name,
                        args=args,
                        index=tool_call.get("index", 0),
                    )
                )

        return ChatGenerationChunk(
            text=content or "",
            generation_info=generation_info,
            message=AIMessageChunk(
                content=content or "",
                usage_metadata=usage_metadata,
                tool_call_chunks=tool_call_chunks,
            ),
        )

    def _uipath_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Generator[ChatGenerationChunk, None, None]:
        request_body = self._preprocess_request(messages, stop=stop, **kwargs)
        request_body["stream"] = True
        for chunk in self.uipath_stream(
            request_body=request_body, stream_type="lines", raise_status_error=True
        ):
            chunk = str(chunk)
            if chunk.startswith("data:"):
                chunk = chunk[len("data:") :].strip()
            try:
                json_data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "id" in json_data and not json_data["id"]:
                continue
            yield self._generate_chunk(json_data)

    async def _uipath_astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        request_body = self._preprocess_request(messages, stop=stop, **kwargs)
        request_body["stream"] = True
        async for chunk in self.uipath_astream(
            request_body=request_body, stream_type="lines", raise_status_error=True
        ):
            chunk = str(chunk)
            if chunk.startswith("data:"):
                chunk = chunk[len("data:") :].strip()
            try:
                json_data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "id" in json_data and not json_data["id"]:
                continue
            yield self._generate_chunk(json_data)
