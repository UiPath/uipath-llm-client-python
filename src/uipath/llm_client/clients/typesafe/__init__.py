"""UiPath client for TypeSafe AI's Jev classification model.

Jev is not a chat model: it answers typed questions (``choice``, ``score``,
``noul``) about an input ``state`` with calibrated probabilities. See
https://docs.typesafe.ai/api for the request/response contract.
"""

from uipath.llm_client.clients.typesafe.client import (
    JEV_DEFAULT_MODEL,
    UiPathJevClient,
)

__all__ = [
    "JEV_DEFAULT_MODEL",
    "UiPathJevClient",
]
