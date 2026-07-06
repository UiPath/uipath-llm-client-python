import contextvars
from collections.abc import Mapping, Sequence
from urllib.parse import quote

from httpx import Headers

from uipath.llm_client.settings.base import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode

# HTTP/1.1 carries header values as ISO-8859-1 (latin-1) octets. httpx, however,
# re-encodes ``str`` values as ASCII in ``Headers.update`` and crashes with
# ``UnicodeEncodeError`` on anything outside that range. We pre-encode values to
# latin-1 bytes (transmitted verbatim) and, for the rare value outside latin-1
# (e.g. CJK/emoji), fall back to ASCII percent-encoding which never crashes.
HTTP_HEADER_ENCODING = "latin-1"

UIPATH_DEFAULT_REQUEST_HEADERS: dict[str, str] = {
    "X-UiPath-LLMGateway-TimeoutSeconds": "895",  # server side timeout
    "X-UiPath-LLMGateway-AllowFull4xxResponse": "false",  # allow full 4xx responses (default is false) — kept false to avoid PII leakage in logs
}

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


def set_captured_response_headers(
    headers: dict[str, str],
) -> contextvars.Token[dict[str, str] | None]:
    """Set captured response headers for the current context."""
    return _CAPTURED_RESPONSE_HEADERS.set(headers)


def get_dynamic_request_headers() -> dict[str, str]:
    """Get dynamic headers to be injected into the next outgoing request.

    Returns an empty dict if no dynamic headers have been set in this context.
    """
    return dict(_DYNAMIC_REQUEST_HEADERS.get() or {})


def set_dynamic_request_headers(
    headers: dict[str, str],
) -> contextvars.Token[dict[str, str] | None]:
    """Set headers to be injected into the next outgoing request."""
    return _DYNAMIC_REQUEST_HEADERS.set(headers)


def encode_header_value(value: str) -> bytes:
    """Encode a header value to bytes so httpx can transmit it without crashing.

    httpx encodes ``str`` header values as ASCII and raises ``UnicodeEncodeError``
    on any non-ASCII character. HTTP/1.1 header values are carried as ISO-8859-1
    (latin-1) octets, so latin-1-representable values are encoded as latin-1 bytes
    (passed through by httpx verbatim). Values outside latin-1 (e.g. CJK or emoji)
    are percent-encoded to pure-ASCII bytes so the send path never crashes.

    Args:
        value: The raw header value, possibly containing non-ASCII characters.

    Returns:
        latin-1 ``bytes`` when the value is latin-1-representable, otherwise
        percent-encoded ASCII ``bytes``.
    """
    try:
        return value.encode(HTTP_HEADER_ENCODING)
    except UnicodeEncodeError:
        return quote(value).encode("ascii")


def encode_header_items(headers: Mapping[str, str]) -> list[tuple[bytes, bytes]]:
    """Encode header names and values to bytes for safe injection via ``Headers.update``.

    Returns ``(name, value)`` byte tuples — the form httpx accepts without
    re-encoding — applying :func:`encode_header_value` to each value.
    """
    return [
        (name.encode(HTTP_HEADER_ENCODING), encode_header_value(value))
        for name, value in headers.items()
    ]


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
        elif (
            api_config.routing_mode == RoutingMode.PASSTHROUGH
            and api_config.api_type == ApiType.COMPLETIONS
        ):
            if api_config.api_flavor is not None:
                headers["X-UiPath-LlmGateway-ApiFlavor"] = api_config.api_flavor
            if api_config.api_version is not None:
                headers["X-UiPath-LlmGateway-ApiVersion"] = api_config.api_version
    if byo_connection_id is not None:
        headers["X-UiPath-LlmGateway-ByoIsConnectionId"] = byo_connection_id
    return headers
