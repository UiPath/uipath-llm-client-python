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
    >>> from uipath_langchain_client.normalized.chat_models import UiPathNormalizedChatModel
    >>> from uipath_langchain_client.settings import get_default_client_settings
    >>>
    >>> settings = get_default_client_settings()
    >>> chat = UiPathNormalizedChatModel(
    ...     model="gpt-4o-2024-11-20",
    ...     client_settings=settings,
    ... )
    >>> response = chat.invoke("Hello!")
"""

import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.base import (
    LanguageModelInput,
)
from langchain_core.language_models.chat_models import BaseChatModel
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
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)
from pydantic import Field
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig


class UiPathNormalizedChatModel(UiPathBaseLLMClient, BaseChatModel):
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
        >>> chat = UiPathNormalizedChatModel(
        ...     model="gpt-4o-2024-11-20",
        ...     client_settings=settings,
        ...     temperature=0.7,
        ...     max_tokens=1000,
        ... )
        >>> response = chat.invoke("Explain machine learning.")
    """

    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="normalized",
        freeze_base_url=True,
    )

    # Standard LLM parameters
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")

    n: int | None = None  # Number of completions to generate
    top_p: float | None = None  # Nucleus sampling probability mass
    presence_penalty: float | None = None  # Penalty for repeated tokens
    frequency_penalty: float | None = None  # Frequency-based repetition penalty
    verbosity: str | None = None  # Response verbosity: "low", "medium", or "high"

    model_kwargs: dict[str, Any] = Field(
        default_factory=dict
    )  # Additional model-specific parameters
    disabled_params: dict[str, Any] | None = None  # Parameters to exclude from requests

    # OpenAI o1/o3 reasoning parameters
    reasoning: dict[str, Any] | None = None  # {"effort": "low"|"medium"|"high", "summary": ...}
    reasoning_effort: str | None = None  # "minimal", "low", "medium", or "high"

    # Anthropic Claude extended thinking parameters
    thinking: dict[str, Any] | None = None  # {"type": "enabled"|"disabled", "budget_tokens": N}

    # Google Gemini thinking parameters
    thinking_level: str | None = None  # Thinking depth level
    thinking_budget: int | None = None  # Token budget for thinking
    include_thoughts: bool | None = None  # Include thinking in response

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "UiPath-Normalized"

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        exclude_if_none = {
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
            "stop": self.stop or None,  # Also exclude empty list for this
            "n": self.n,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            "reasoning": self.reasoning,
            "reasoning_effort": self.reasoning_effort,
            "thinking": self.thinking,
            "thinking_level": self.thinking_level,
            "thinking_budget": self.thinking_budget,
            "include_thoughts": self.include_thoughts,
        }

        return {
            "model": self.model_name,
            **{k: v for k, v in exclude_if_none.items() if v is not None},
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
                audio=json_data.get("audio_tokens", 0),
                reasoning=json_data.get("thoughts_tokens", 0),
            ),
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        strict: bool | None = None,
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
            tool_choice_object = {
                "type": tool_choice,
            }
        else:
            tool_choice_object = {
                "type": "tool",
                "name": tool_choice,
            }

        return super().bind(
            tools=formatted_tools,
            tool_choice=tool_choice_object,
            **kwargs,
        )

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
                    and "claude" in self.model_name.lower()
                    and not converted_message["content"]
                ):
                    converted_message["content"] = "tool_call"
                if "tool_calls" in converted_message:
                    converted_message["tool_calls"] = [
                        {
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
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

    def _generate(
        self,
        messages: list[BaseMessage],
        *args: Any,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request_body = self._preprocess_request(messages, **kwargs)
        response = self.uipath_request(request_body=request_body)
        return self._postprocess_response(response.json())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        *args: Any,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request_body = self._preprocess_request(messages, **kwargs)
        response = await self.uipath_arequest(request_body=request_body)
        return self._postprocess_response(response.json())

    def _generate_chunk(
        self, original_message: str, json_data: dict[str, Any]
    ) -> ChatGenerationChunk:
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
            text=original_message,
            generation_info=generation_info,
            message=AIMessageChunk(
                content=content,
                usage_metadata=usage_metadata,
                tool_call_chunks=tool_call_chunks,
            ),
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        *args: Any,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request_body = self._preprocess_request(messages, **kwargs)
        for chunk in self.uipath_stream(request_body=request_body, stream_type="lines"):
            chunk = str(chunk)
            if chunk.startswith("data:"):
                chunk = chunk.split("data:")[1].strip()
            try:
                json_data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "id" in json_data and not json_data["id"]:
                continue
            yield self._generate_chunk(chunk, json_data)

    async def _astream(
        self,
        messages: list[BaseMessage],
        *args: Any,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        request_body = self._preprocess_request(messages, **kwargs)
        async for chunk in self.uipath_astream(request_body=request_body, stream_type="lines"):
            chunk = str(chunk)
            if chunk.startswith("data:"):
                chunk = chunk.split("data:")[1].strip()
            try:
                json_data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "id" in json_data and not json_data["id"]:
                continue
            yield self._generate_chunk(chunk, json_data)
