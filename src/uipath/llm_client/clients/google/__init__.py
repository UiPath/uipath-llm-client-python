"""UiPath Google Gemini client wrapper.

Requires the ``google`` optional extra::

    uv add uipath-llm-client[google]
"""

from uipath.llm_client.clients.google.client import UiPathGoogle

__all__ = [
    "UiPathGoogle",
]
