"""Shared utilities for building UiPath-configured httpx clients."""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath.llm_client.settings.base import UiPathAPIConfig, UiPathBaseSettings
from uipath.llm_client.utils.retry import RetryConfig


def build_httpx_client(
    *,
    model_name: str,
    byo_connection_id: str | None,
    client_settings: UiPathBaseSettings,
    timeout: float | None,
    max_retries: int | None,
    default_headers: Mapping[str, str] | None,
    captured_headers: Sequence[str],
    retry_config: RetryConfig | None,
    logger: logging.Logger | None,
    api_config: UiPathAPIConfig | None = None,
    event_hooks: dict[str, list[Callable[..., Any]]] | None = None,
) -> UiPathHttpxClient:
    """Build a sync UiPath httpx client with auth, routing headers, and retry.

    When *api_config* is provided the base URL and routing/auth headers are
    derived from *client_settings*.  When it is ``None`` (e.g. for OpenAI
    clients that resolve routing per-request via event hooks) those are
    omitted and only the auth pipeline, default headers and retry transport
    are configured.
    """
    headers: dict[str, str] = {**(default_headers or {})}
    kwargs: dict[str, Any] = {}

    if api_config is not None:
        headers.update(
            client_settings.build_auth_headers(model_name=model_name, api_config=api_config)
        )
        kwargs["base_url"] = client_settings.build_base_url(
            model_name=model_name, api_config=api_config
        )

    if event_hooks is not None:
        kwargs["event_hooks"] = event_hooks

    return UiPathHttpxClient(
        model_name=model_name,
        byo_connection_id=byo_connection_id,
        api_config=api_config,
        timeout=timeout,
        max_retries=max_retries,
        retry_config=retry_config,
        headers=headers,
        captured_headers=captured_headers,
        logger=logger,
        auth=client_settings.build_auth_pipeline(),
        **kwargs,
    )


def build_httpx_async_client(
    *,
    model_name: str,
    byo_connection_id: str | None,
    client_settings: UiPathBaseSettings,
    timeout: float | None,
    max_retries: int | None,
    default_headers: Mapping[str, str] | None,
    captured_headers: Sequence[str],
    retry_config: RetryConfig | None,
    logger: logging.Logger | None,
    api_config: UiPathAPIConfig | None = None,
    event_hooks: dict[str, list[Callable[..., Any]]] | None = None,
) -> UiPathHttpxAsyncClient:
    """Build an async UiPath httpx client with auth, routing headers, and retry.

    See :func:`build_httpx_client` for parameter details.
    """
    headers: dict[str, str] = {**(default_headers or {})}
    kwargs: dict[str, Any] = {}

    if api_config is not None:
        headers.update(
            client_settings.build_auth_headers(model_name=model_name, api_config=api_config)
        )
        kwargs["base_url"] = client_settings.build_base_url(
            model_name=model_name, api_config=api_config
        )

    if event_hooks is not None:
        kwargs["event_hooks"] = event_hooks

    return UiPathHttpxAsyncClient(
        model_name=model_name,
        byo_connection_id=byo_connection_id,
        api_config=api_config,
        timeout=timeout,
        max_retries=max_retries,
        retry_config=retry_config,
        headers=headers,
        captured_headers=captured_headers,
        logger=logger,
        auth=client_settings.build_auth_pipeline(),
        **kwargs,
    )
