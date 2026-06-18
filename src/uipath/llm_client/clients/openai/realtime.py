"""Realtime (WebSocket) support for the UiPath OpenAI clients.

The OpenAI Realtime API speaks over a WebSocket rather than HTTP, so it does not
go through the httpx routing used for completions/embeddings. Instead,
``UiPathOpenAI`` / ``UiPathAsyncOpenAI`` expose ``client.realtime.connect()`` —
exactly like the stock OpenAI SDK — by swapping in the resource wrappers defined
here. On connect these wrappers:

- point the client's ``websocket_base_url`` at the gateway passthrough realtime
  path (``.../vendor/<vendor>/model/<model>``); the SDK appends ``/realtime``,
- set the S2S bearer token as ``api_key`` (the SDK sends it as
  ``Authorization: Bearer`` on the WebSocket upgrade),
- inject the ``X-UiPath-*`` routing headers on the upgrade request.

Completions/embeddings are unaffected: their auth comes from the httpx auth
pipeline, and the realtime URL is only built when ``.realtime`` is accessed.

Example:
    >>> import asyncio
    >>> from uipath.llm_client.clients.openai import UiPathAsyncOpenAI
    >>>
    >>> async def main():
    ...     client = UiPathAsyncOpenAI(model_name="gpt-realtime")
    ...     async with client.realtime.connect() as conn:
    ...         await conn.session.update(
    ...             session={"type": "realtime", "output_modalities": ["text"]}
    ...         )
    ...         await conn.conversation.item.create(
    ...             item={
    ...                 "type": "message",
    ...                 "role": "user",
    ...                 "content": [{"type": "input_text", "text": "Say hello!"}],
    ...             }
    ...         )
    ...         await conn.response.create()
    ...         async for event in conn:
    ...             if event.type == "response.output_text.delta":
    ...                 print(event.delta, end="")
    ...             elif event.type == "response.done":
    ...                 break
    >>> asyncio.run(main())
"""

import re

from typing_extensions import override

from uipath.llm_client.settings import UiPathAPIConfig, UiPathBaseSettings
from uipath.llm_client.settings.constants import RoutingMode
from uipath.llm_client.settings.llmgateway.auth import LLMGatewayS2SAuth

try:
    from openai import AsyncOpenAI, OpenAI
    from openai._types import Headers, Omit, Query, omit
    from openai.resources.realtime import AsyncRealtime, Realtime
    from openai.resources.realtime.realtime import (
        AsyncRealtimeConnectionManager,
        RealtimeConnectionManager,
    )
    from openai.types.websocket_connection_options import WebSocketConnectionOptions
except ImportError as e:
    raise ImportError(
        "The 'openai' extra is required for realtime support. "
        "Install it with: uv add uipath-llm-client[openai]"
    ) from e

# The gateway expects the native-OpenAI realtime vendor segment in the path.
DEFAULT_REALTIME_VENDOR = "nativeopenai"
# Passthrough api_type segment for the realtime endpoint (not a normalized ApiType).
REALTIME_API_TYPE = "realtime"


def _realtime_api_config(vendor_type: str) -> UiPathAPIConfig:
    return UiPathAPIConfig(
        api_type=REALTIME_API_TYPE,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=vendor_type,
    )


def build_realtime_ws_base_url(
    settings: UiPathBaseSettings,
    *,
    model_name: str,
    vendor_type: str = DEFAULT_REALTIME_VENDOR,
) -> str:
    """Build the ``websocket_base_url`` to hand to the OpenAI SDK.

    The SDK appends ``/realtime`` to ``websocket_base_url`` when connecting, so
    this strips the trailing ``/realtime`` produced by ``build_base_url`` and
    converts the scheme to ``wss``/``ws``.
    """
    url = settings.build_base_url(
        model_name=model_name, api_config=_realtime_api_config(vendor_type)
    )
    suffix = f"/{REALTIME_API_TYPE}"
    if url.endswith(suffix):
        url = url[: -len(suffix)]
    # Collapse accidental double slashes (e.g. a trailing slash in base_url),
    # leaving the scheme's own "//" intact.
    scheme, sep, rest = url.partition("://")
    if sep:
        url = f"{scheme}://{re.sub(r'/{2,}', '/', rest)}"
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def _prepare_connection(
    client: "OpenAI | AsyncOpenAI",
    settings: UiPathBaseSettings,
    *,
    model: str,
    vendor_type: str,
    extra_headers: Headers,
) -> Headers:
    """Configure ``client`` for a gateway realtime connection to ``model``.

    Sets ``websocket_base_url`` and the S2S bearer token (read straight from the
    gateway auth handler), and returns the ``X-UiPath-*`` routing headers merged
    with ``extra_headers`` for the WebSocket upgrade.
    """
    client.websocket_base_url = build_realtime_ws_base_url(
        settings, model_name=model, vendor_type=vendor_type
    )
    auth = settings.build_auth_pipeline()
    if isinstance(auth, LLMGatewayS2SAuth) and auth.access_token:
        client.api_key = auth.access_token
    merged: dict[str, object] = {
        **settings.build_auth_headers(
            model_name=model, api_config=_realtime_api_config(vendor_type)
        )
    }
    if extra_headers:
        merged.update(extra_headers)
    return merged  # type: ignore[return-value]


class _UiPathRealtime(Realtime):
    """``Realtime`` resource that routes ``connect()`` through the gateway."""

    def __init__(
        self, client: OpenAI, *, settings: UiPathBaseSettings, model: str, vendor_type: str
    ) -> None:
        super().__init__(client)
        self._uipath_settings = settings
        self._uipath_model = model
        self._uipath_vendor = vendor_type

    @override
    def connect(
        self,
        *,
        call_id: str | Omit = omit,
        model: str | Omit = omit,
        extra_query: Query = {},
        extra_headers: Headers = {},
        websocket_connection_options: WebSocketConnectionOptions = {},
    ) -> RealtimeConnectionManager:
        resolved_model: str = self._uipath_model if model is omit else model  # type: ignore[assignment]
        merged_headers = _prepare_connection(
            self._client,
            self._uipath_settings,
            model=resolved_model,
            vendor_type=self._uipath_vendor,
            extra_headers=extra_headers,
        )
        return super().connect(
            call_id=call_id,
            model=resolved_model,
            extra_query=extra_query,
            extra_headers=merged_headers,
            websocket_connection_options=websocket_connection_options,
        )


class _UiPathAsyncRealtime(AsyncRealtime):
    """``AsyncRealtime`` resource that routes ``connect()`` through the gateway."""

    def __init__(
        self, client: AsyncOpenAI, *, settings: UiPathBaseSettings, model: str, vendor_type: str
    ) -> None:
        super().__init__(client)
        self._uipath_settings = settings
        self._uipath_model = model
        self._uipath_vendor = vendor_type

    @override
    def connect(
        self,
        *,
        call_id: str | Omit = omit,
        model: str | Omit = omit,
        extra_query: Query = {},
        extra_headers: Headers = {},
        websocket_connection_options: WebSocketConnectionOptions = {},
    ) -> AsyncRealtimeConnectionManager:
        resolved_model: str = self._uipath_model if model is omit else model  # type: ignore[assignment]
        merged_headers = _prepare_connection(
            self._client,
            self._uipath_settings,
            model=resolved_model,
            vendor_type=self._uipath_vendor,
            extra_headers=extra_headers,
        )
        return super().connect(
            call_id=call_id,
            model=resolved_model,
            extra_query=extra_query,
            extra_headers=merged_headers,
            websocket_connection_options=websocket_connection_options,
        )
