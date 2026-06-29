"""UiPath LiteLLM Client — provider-agnostic LLM access via LiteLLM.

Requires the ``litellm`` optional extra::

    uv add uipath-llm-client[litellm]
"""

from uipath.llm_client.clients.litellm.client import UiPathLiteLLM

__all__ = [
    "UiPathLiteLLM",
]
