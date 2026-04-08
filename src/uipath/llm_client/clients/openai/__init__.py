"""UiPath OpenAI/Azure OpenAI client wrappers.

Requires the ``openai`` optional extra::

    uv add uipath-llm-client[openai]
"""

from uipath.llm_client.clients.openai.client import (
    UiPathAsyncAzureOpenAI,
    UiPathAsyncOpenAI,
    UiPathAzureOpenAI,
    UiPathOpenAI,
)

__all__ = [
    "UiPathOpenAI",
    "UiPathAsyncOpenAI",
    "UiPathAzureOpenAI",
    "UiPathAsyncAzureOpenAI",
]
