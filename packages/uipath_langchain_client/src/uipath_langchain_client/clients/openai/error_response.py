"""Surface a chat completions error body as a typed UiPath error."""

from typing import Any, cast

import openai
from langchain_core.outputs import ChatResult

from uipath.llm_client.utils.exceptions import UiPathError, UiPathLLMErrorCode


def _error_message(error: Any) -> str:
    """The endpoint's own message for an ``error`` field, as text."""
    if isinstance(error, dict):
        message = cast(dict[str, Any], error).get("message")
        if isinstance(message, str) and message:
            return message
    return str(error)


class OpenAIErrorResponseMixin:
    """Raise a typed error for a chat completions body that is an error object.

    Some OpenAI-compatible endpoints (seen on bring-your-own model endpoints)
    answer a call with HTTP 200 and a body such as ``{"error": "..."}`` instead
    of a completion. langchain-openai's ``_create_chat_result`` raises a bare
    ``ValueError(response["error"])`` for it. That error carries no HTTP
    response, so ``wrap_provider_errors`` could only turn it into an
    uncategorised ``UiPathError`` holding the endpoint's message, and callers
    could not tell it from a client-side bug. This raises a ``UiPathError`` with
    :attr:`UiPathLLMErrorCode.PROVIDER_ERROR_RESPONSE` instead, so callers can
    attribute the failure to the model endpoint.
    """

    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        error = (
            response.get("error")
            if isinstance(response, dict)
            else getattr(response, "error", None)
        )
        if error:
            raise UiPathError(
                _error_message(error),
                error_code=UiPathLLMErrorCode.PROVIDER_ERROR_RESPONSE,
            )
        return cast(ChatResult, cast(Any, super())._create_chat_result(response, generation_info))
