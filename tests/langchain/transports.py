"""Route a chat model's UiPath httpx clients to a test transport."""

from typing import Any

import httpx


def route_transport(
    chat: Any,
    transport: httpx.BaseTransport,
    async_transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    # Vendor SDKs hold these very client objects, so swapping the transport is
    # enough. Mounts are cleared because they would bypass the swapped transport.
    chat.uipath_sync_client._transport = transport
    chat.uipath_sync_client._mounts = {}
    if async_transport is not None:
        chat.uipath_async_client._transport = async_transport
        chat.uipath_async_client._mounts = {}
