"""Shared utilities for UiPath LangChain provider clients."""

from httpx import URL, Request

from uipath_langchain_client.settings import ApiFlavor


def fix_url_and_api_flavor_header(
    base_url: str, request: Request, *, api_flavor: str | None = None
) -> None:
    """Set the API flavor header and rewrite the URL to the base gateway URL.

    When *api_flavor* is provided (e.g. from the discovery endpoint), it is
    used directly — the model only supports that specific flavor.  Otherwise
    the flavor is inferred from the outgoing URL suffix (``/responses`` vs
    ``/chat/completions``).

    Args:
        base_url: The UiPath gateway base URL to rewrite the request to.
        request: The outgoing httpx request (mutated in place).
        api_flavor: Locked API flavor from discovery. When set, overrides
            dynamic detection from the URL path.
    """
    if api_flavor is not None:
        request.headers["X-UiPath-LlmGateway-ApiFlavor"] = api_flavor
    else:
        url_suffix = str(request.url).split(base_url)[-1]
        if "responses" in url_suffix:
            request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.RESPONSES.value
        else:
            request.headers["X-UiPath-LlmGateway-ApiFlavor"] = ApiFlavor.CHAT_COMPLETIONS.value
    request.url = URL(base_url)
