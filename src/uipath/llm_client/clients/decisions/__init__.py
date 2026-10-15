"""UiPath client for OpenAI's Decisions API.

The Decisions API is not a chat model: it answers typed questions (``choice``,
``score``, ``predicate``) about an ``input`` (text, or text with images) with
probabilities. See https://developers.openai.com/api/docs/guides/decisions for the
request/response contract.
"""

from uipath.llm_client.clients.decisions.client import UiPathDecisionsClient

__all__ = [
    "UiPathDecisionsClient",
]
