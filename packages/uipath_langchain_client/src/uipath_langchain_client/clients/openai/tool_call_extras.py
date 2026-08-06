"""Preserve provider-specific tool-call fields in OpenAI-compatible chats."""

from collections.abc import Mapping
from typing import Any, cast

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult

_OPENAI_TOOL_CALL_EXTRAS_KEY = "__openai_tool_call_extras__"
_STANDARD_TOOL_CALL_FIELDS = frozenset({"id", "type", "function", "index"})


def _get_tool_call_extras(
    tool_call: Mapping[str, Any], fallback_key: str | None = None
) -> tuple[str | None, dict[str, Any]]:
    """Extract provider extensions without retaining replaceable call fields."""
    extras = {
        key: value for key, value in tool_call.items() if key not in _STANDARD_TOOL_CALL_FIELDS
    }
    if not extras:
        return None, {}
    return tool_call.get("id") or fallback_key, extras


def _store_tool_call_extras(
    message: AIMessage | AIMessageChunk,
    tool_calls: list[Mapping[str, Any]],
) -> None:
    stored_extras = message.additional_kwargs.get(_OPENAI_TOOL_CALL_EXTRAS_KEY)
    extras_by_id = dict(stored_extras) if isinstance(stored_extras, Mapping) else {}
    for index, tool_call in enumerate(tool_calls):
        fallback_key = (
            f"__index_{tool_call['index']}" if "index" in tool_call else f"__index_{index}"
        )
        key, extras = _get_tool_call_extras(tool_call, fallback_key)
        if key and extras:
            extras_by_id[key] = extras
    if extras_by_id:
        message.additional_kwargs[_OPENAI_TOOL_CALL_EXTRAS_KEY] = extras_by_id


class OpenAIToolCallExtrasMixin:
    """Keep provider-specific tool-call fields across LangChain conversion."""

    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(exclude={"choices": {"__all__": {"message": {"parsed"}}}})
        )
        result = cast(Any, super())._create_chat_result(response, generation_info)

        for choice, generation in zip(
            response_dict.get("choices") or [], result.generations, strict=False
        ):
            raw_tool_calls = choice.get("message", {}).get("tool_calls") or []
            if raw_tool_calls and isinstance(generation.message, AIMessage):
                _store_tool_call_extras(generation.message, raw_tool_calls)

        return cast(ChatResult, result)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        generation = cast(Any, super())._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return cast(ChatGenerationChunk | None, generation)

        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        if choices and choices[0].get("delta"):
            raw_tool_calls = choices[0]["delta"].get("tool_calls") or []
            if raw_tool_calls:
                _store_tool_call_extras(generation.message, raw_tool_calls)

        return cast(ChatGenerationChunk, generation)

    def _get_generation_chunk_from_completion(
        self, completion: openai.BaseModel
    ) -> ChatGenerationChunk:
        generation = cast(Any, super())._get_generation_chunk_from_completion(completion)
        # This final summary chunk follows the actual tool-call deltas. Repeating
        # string-valued extras here would make LangChain concatenate the signature.
        generation.message.additional_kwargs.pop(_OPENAI_TOOL_CALL_EXTRAS_KEY, None)
        return cast(ChatGenerationChunk, generation)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = cast(Any, self)._convert_input(input_).to_messages()
        payload = cast(Any, super())._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return cast(dict[str, Any], payload)

        for message, payload_message in zip(messages, payload_messages, strict=False):
            if not isinstance(message, AIMessage) or not isinstance(payload_message, dict):
                continue

            extras_by_id = message.additional_kwargs.get(_OPENAI_TOOL_CALL_EXTRAS_KEY)
            tool_calls = payload_message.get("tool_calls")
            if not isinstance(extras_by_id, dict) or not isinstance(tool_calls, list):
                continue

            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                extras = extras_by_id.get(tool_call.get("id")) or extras_by_id.get(
                    f"__index_{index}"
                )
                if isinstance(extras, Mapping):
                    tool_call.update(
                        {
                            key: value
                            for key, value in extras.items()
                            if key not in _STANDARD_TOOL_CALL_FIELDS
                        }
                    )

        return cast(dict[str, Any], payload)
