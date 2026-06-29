"""UiPath Anthropic client wrappers (Bedrock, Vertex AI, Foundry).

Requires the ``anthropic`` optional extra::

    uv add uipath-llm-client[anthropic]
"""

from uipath.llm_client.clients.anthropic.client import (
    UiPathAnthropic,
    UiPathAnthropicBedrock,
    UiPathAnthropicFoundry,
    UiPathAnthropicVertex,
    UiPathAsyncAnthropic,
    UiPathAsyncAnthropicBedrock,
    UiPathAsyncAnthropicFoundry,
    UiPathAsyncAnthropicVertex,
)

__all__ = [
    "UiPathAnthropic",
    "UiPathAsyncAnthropic",
    "UiPathAnthropicBedrock",
    "UiPathAsyncAnthropicBedrock",
    "UiPathAnthropicVertex",
    "UiPathAsyncAnthropicVertex",
    "UiPathAnthropicFoundry",
    "UiPathAsyncAnthropicFoundry",
]
