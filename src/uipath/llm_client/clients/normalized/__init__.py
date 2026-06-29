"""UiPath Normalized Client - Provider-agnostic LLM client.

No optional dependencies required. Works with the base uipath-llm-client package.
"""

from uipath.llm_client.clients.normalized.client import UiPathNormalizedClient
from uipath.llm_client.clients.normalized.completions import (
    MessageType,
    ResponseFormatType,
    ToolChoiceType,
    ToolType,
)
from uipath.llm_client.clients.normalized.types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Delta,
    EmbeddingData,
    EmbeddingResponse,
    Message,
    StreamChoice,
    ToolCall,
    ToolCallChunk,
    Usage,
)

__all__ = [
    "UiPathNormalizedClient",
    # Input types
    "MessageType",
    "ToolType",
    "ToolChoiceType",
    "ResponseFormatType",
    # Response types
    "ChatCompletion",
    "ChatCompletionChunk",
    "Choice",
    "Delta",
    "EmbeddingData",
    "EmbeddingResponse",
    "Message",
    "StreamChoice",
    "ToolCall",
    "ToolCallChunk",
    "Usage",
]
