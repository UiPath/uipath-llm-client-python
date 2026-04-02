"""Shared utilities for UiPath LangChain provider clients."""

from httpx import URL, Request

from uipath_langchain_client.settings import ApiFlavor


def fix_url_and_api_flavor_header(base_url: str, request: Request) -> None:
    """Detect API flavor from URL suffix and rewrite the URL to the base gateway URL.

    Inspects the outgoing request URL to determine whether it targets the
    OpenAI *responses* or *chat completions* endpoint and sets the
    ``X-UiPath-LlmGateway-ApiFlavor`` header accordingly.  The request URL
    is then collapsed back to *base_url* so that the gateway receives a
    clean path.

    Args:
        base_url: The UiPath gateway base URL to rewrite the request to.
        request: The outgoing httpx request (mutated in place).
    """
    url_suffix = str(request.url).split(base_url)[-1]
    if "responses" in url_suffix:
        request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.RESPONSES.value
    else:
        request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.CHAT_COMPLETIONS.value
    request.url = URL(base_url)
