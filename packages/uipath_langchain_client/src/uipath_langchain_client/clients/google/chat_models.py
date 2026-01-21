from collections.abc import AsyncIterator, Iterator
from typing import Self

from httpx import Response
from pydantic import Field, SecretStr, model_validator
from uipath_langchain_client.base_client import UiPathBaseLLMClient
from uipath_langchain_client.settings import UiPathAPIConfig

try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    from google.genai.client import Client
    from google.genai.types import HttpOptions
except ImportError as e:
    raise ImportError(
        "The 'google' extra is required to use UiPathChatGoogleGenerativeAI. "
        "Install it with: uv add uipath-langchain-client[google]"
    ) from e


def _wrap_iter_lines(original: Iterator[str]) -> Iterator[str]:
    """Wrap iter_lines to extract individual JSON objects from streaming responses.

    The LLM Gateway wraps streaming JSON responses in an array like [{...}, {...}].
    This extracts each complete JSON object and yields them individually.
    Handles multiple JSON objects on a single line (e.g., {...},{...}).

    We prefix output with 'data: ' so the SDK's _iter_response_stream bypasses its
    broken brace counting (which doesn't handle braces inside strings) and yields
    our JSON objects directly.

    Temporal Fix until it's fixed in the main package.
    """
    buffer = ""
    balance = 0
    in_string = False
    escape_next = False

    for line in original:
        # Handle data: prefix (SSE format)
        if line.startswith("data:"):
            line = line[5:].lstrip()

        for char in line:
            # Handle escape sequences in strings
            if escape_next:
                buffer += char
                escape_next = False
                continue

            if char == "\\" and in_string:
                buffer += char
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                buffer += char
                continue

            # Only track braces outside of strings
            if not in_string:
                if char == "{":
                    balance += 1
                    buffer += char
                elif char == "}":
                    buffer += char
                    balance -= 1
                    if balance == 0 and buffer:
                        # Complete JSON object found - yield with 'data: ' prefix
                        # so SDK bypasses its broken brace counting
                        yield "data: " + buffer
                        buffer = ""
                elif balance == 0:
                    # Skip characters outside JSON objects (array brackets, commas, whitespace)
                    continue
                else:
                    buffer += char
            else:
                buffer += char

    # Yield any remaining buffer (handles incomplete streams)
    if buffer:
        yield "data: " + buffer


async def _wrap_aiter_lines(original: AsyncIterator[str]) -> AsyncIterator[str]:
    """Async version of _wrap_iter_lines.

    Extracts individual JSON objects from streaming responses.
    Handles multiple JSON objects on a single line (e.g., {...},{...}).

    We prefix output with 'data: ' so the SDK's _iter_response_stream bypasses its
    broken brace counting (which doesn't handle braces inside strings) and yields
    our JSON objects directly.
    """
    buffer = ""
    balance = 0
    in_string = False
    escape_next = False

    async for line in original:
        # Handle data: prefix (SSE format)
        if line.startswith("data:"):
            line = line[5:].lstrip()

        for char in line:
            # Handle escape sequences in strings
            if escape_next:
                buffer += char
                escape_next = False
                continue

            if char == "\\" and in_string:
                buffer += char
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                buffer += char
                continue

            # Only track braces outside of strings
            if not in_string:
                if char == "{":
                    balance += 1
                    buffer += char
                elif char == "}":
                    buffer += char
                    balance -= 1
                    if balance == 0 and buffer:
                        # Complete JSON object found - yield with 'data: ' prefix
                        # so SDK bypasses its broken brace counting
                        yield "data: " + buffer
                        buffer = ""
                elif balance == 0:
                    # Skip characters outside JSON objects (array brackets, commas, whitespace)
                    continue
                else:
                    buffer += char
            else:
                buffer += char

    # Yield any remaining buffer (handles incomplete streams)
    if buffer:
        yield "data: " + buffer


class UiPathChatGoogleGenerativeAI(UiPathBaseLLMClient, ChatGoogleGenerativeAI):
    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="vertexai",
        api_flavor="generate-content",
        api_version="v1beta1",
        freeze_base_url=True,
    )

    # Override fields to avoid errors when instantiating the class
    model: str = Field(default="", alias="model_name")
    google_api_key: SecretStr | None = Field(default=SecretStr("PLACEHOLDER"))

    @model_validator(mode="after")
    def setup_uipath_client(self) -> Self:
        def fix_streaming_response(response: Response):
            """Monkey-patch iter_lines to strip JSON array brackets."""
            original_iter_lines = response.iter_lines
            response.iter_lines = lambda: _wrap_iter_lines(original_iter_lines())

        async def fix_streaming_response_async(response: Response):
            """Monkey-patch aiter_lines to strip JSON array brackets."""
            original_aiter_lines = response.aiter_lines
            response.aiter_lines = lambda: _wrap_aiter_lines(original_aiter_lines())

        self.uipath_sync_client.event_hooks["response"].append(fix_streaming_response)
        self.uipath_async_client.event_hooks["response"].append(fix_streaming_response_async)

        # TODO: in exactly 2 weeks, we need to uncomment this part of the code because it will work, 5 february 2026 is the date.
        # def fix_url_for_streaming(request: Request):
        #     if request.headers.get("X-UiPath-Streaming-Enabled") == "true":
        #         request.url = URL(request.url).copy_add_param("alt", "sse")

        # async def fix_url_for_streaming_async(request: Request):
        #     if request.headers.get("X-UiPath-Streaming-Enabled") == "true":
        #         request.url = URL(request.url).copy_add_param("alt", "sse")

        # self.uipath_sync_client.event_hooks["request"].append(fix_url_for_streaming)
        # self.uipath_async_client.event_hooks["request"].append(fix_url_for_streaming_async)

        self.client = Client(
            vertexai=True,
            api_key="PLACEHOLDER",
            http_options=HttpOptions(
                base_url=str(self.uipath_sync_client.base_url),
                headers=dict(self.uipath_sync_client.headers),
                timeout=None,  # handled by the UiPath client
                retry_options=None,  # handled by the UiPath client
                httpx_client=self.uipath_sync_client,
                httpx_async_client=self.uipath_async_client,
            ),
        )
        return self
