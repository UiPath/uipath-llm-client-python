import contextvars
from collections.abc import Mapping, Sequence

from httpx import Headers

from uipath.llm_client.settings.base import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode

_CAPTURED_RESPONSE_HEADERS: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "_captured_response_headers", default=None
)

_DYNAMIC_REQUEST_HEADERS: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "_dynamic_request_headers", default=None
)


def get_captured_response_headers() -> dict[str, str]:
    """Get response headers captured from the most recent request in this context.

    Returns an empty dict if no headers have been captured or if called
    outside a capture scope.
    """
    return dict(_CAPTURED_RESPONSE_HEADERS.get() or {})


def set_captured_response_headers(headers: dict[str, str]) -> contextvars.Token[dict[str, str] | None]:
    """Set captured response headers for the current context."""
    return _CAPTURED_RESPONSE_HEADERS.set(headers)


def get_dynamic_request_headers() -> dict[str, str]:
    """Get dynamic headers to be injected into the next outgoing request.

    Returns an empty dict if no dynamic headers have been set in this context.
    """
    return dict(_DYNAMIC_REQUEST_HEADERS.get() or {})


def set_dynamic_request_headers(headers: dict[str, str]) -> contextvars.Token[dict[str, str] | None]:
    """Set headers to be injected into the next outgoing request."""
    return _DYNAMIC_REQUEST_HEADERS.set(headers)


def extract_matching_headers(
    response_headers: Headers,
    prefixes: Sequence[str],
) -> dict[str, str]:
    """Extract response headers whose names match any of the given prefixes (case-insensitive)."""
    result: dict[str, str] = {}
    for name, value in response_headers.items():
        name_lower = name.lower()
        for prefix in prefixes:
            if name_lower.startswith(prefix.lower()):
                result[name] = value
                break
    return result


def build_routing_headers(
    *,
    model_name: str | None = None,
    byo_connection_id: str | None = None,
    api_config: UiPathAPIConfig | None = None,
) -> Mapping[str, str]:
    """Build UiPath LLM Gateway routing headers based on configuration.

    Args:
        api_config: UiPath API configuration.
        model_name: LLM model name (required for normalized API).
        byo_connection_id: Bring Your Own connection ID.

    Returns:
        Headers mapping for routing requests through the gateway.
    """
    headers: dict[str, str] = {}
    if api_config is not None:
        if api_config.routing_mode == RoutingMode.NORMALIZED and model_name is not None:
            headers["X-UiPath-LlmGateway-NormalizedApi-ModelName"] = model_name
        elif api_config.routing_mode == RoutingMode.PASSTHROUGH and api_config.api_type == ApiType.COMPLETIONS:
            if api_config.api_flavor is not None:
                headers["X-UiPath-LlmGateway-ApiFlavor"] = api_config.api_flavor
            if api_config.api_version is not None:
                headers["X-UiPath-LlmGateway-ApiVersion"] = api_config.api_version
    if byo_connection_id is not None:
        headers["X-UiPath-LlmGateway-ByoIsConnectionId"] = byo_connection_id
    return headers
