"""UiPath Anthropic client wrappers for routing through UiPath LLM Gateway.

This module provides Anthropic client variants that route requests through
the UiPath LLM Gateway while preserving the full Anthropic SDK interface.

Example:
    >>> from uipath_llm_client.clients.anthropic import UiPathAnthropic
    >>>
    >>> client = UiPathAnthropic(model_name="claude-3-5-sonnet-20241022")
    >>> response = client.messages.create(
    ...     model="claude-3-5-sonnet-20241022",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=1024,
    ... )
"""

import logging
from typing import Any

from uipath_llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
from uipath_llm_client.settings import (
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath_llm_client.utils.retry import RetryConfig

try:
    from anthropic import (
        Anthropic,
        AnthropicBedrock,
        AnthropicFoundry,
        AnthropicVertex,
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )
except ImportError as e:
    raise ImportError(
        "The 'anthropic' extra is required to use UiPath Anthropic clients. "
        "Install it with: uv add uipath-llm-client[anthropic]"
    ) from e


def _build_api_config(vendor_type: str = "anthropic") -> UiPathAPIConfig:
    """Build standard API config for Anthropic clients."""
    return UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type=vendor_type,
        freeze_base_url=True,
    )


class UiPathAnthropic(Anthropic):
    """Anthropic client routed through UiPath LLM Gateway.

    Wraps the standard Anthropic client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name (e.g., "claude-3-5-sonnet-20241022").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to Anthropic client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config()
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            api_key="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )


class UiPathAsyncAnthropic(AsyncAnthropic):
    """Async Anthropic client routed through UiPath LLM Gateway.

    Wraps the standard AsyncAnthropic client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name (e.g., "claude-3-5-sonnet-20241022").
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to AsyncAnthropic client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config()
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            api_key="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )


class UiPathAnthropicBedrock(AnthropicBedrock):
    """Anthropic Bedrock client routed through UiPath LLM Gateway.

    Wraps the AnthropicBedrock client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name.
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to AnthropicBedrock client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config(vendor_type="awsbedrock")
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            aws_access_key="PLACEHOLDER",
            aws_secret_key="PLACEHOLDER",
            aws_region="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )


class UiPathAsyncAnthropicBedrock(AsyncAnthropicBedrock):
    """Async Anthropic Bedrock client routed through UiPath LLM Gateway.

    Wraps the AsyncAnthropicBedrock client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name.
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to AsyncAnthropicBedrock client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config(vendor_type="awsbedrock")
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            aws_access_key="PLACEHOLDER",
            aws_secret_key="PLACEHOLDER",
            aws_region="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )


class UiPathAnthropicVertex(AnthropicVertex):
    """Anthropic Vertex client routed through UiPath LLM Gateway.

    Wraps the AnthropicVertex client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name.
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to AnthropicVertex client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config(vendor_type="vertexai")
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            region="PLACEHOLDER",
            project_id="PLACEHOLDER",
            access_token="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )


class UiPathAsyncAnthropicVertex(AsyncAnthropicVertex):
    """Async Anthropic Vertex client routed through UiPath LLM Gateway.

    Wraps the AsyncAnthropicVertex client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name.
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to AsyncAnthropicVertex client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config(vendor_type="vertexai")
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            region="PLACEHOLDER",
            project_id="PLACEHOLDER",
            access_token="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )


class UiPathAnthropicFoundry(AnthropicFoundry):
    """Anthropic Foundry (Azure) client routed through UiPath LLM Gateway.

    Wraps the AnthropicFoundry client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name.
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to AnthropicFoundry client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config(vendor_type="azure")
        httpx_client = UiPathHttpxClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            api_key="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )


class UiPathAsyncAnthropicFoundry(AsyncAnthropicFoundry):
    """Async Anthropic Foundry (Azure) client routed through UiPath LLM Gateway.

    Wraps the AsyncAnthropicFoundry client to route requests through UiPath's
    LLM Gateway while preserving the full Anthropic SDK interface.

    Args:
        model_name: The Anthropic model name.
        byo_connection_id: Bring Your Own connection ID for custom deployments.
        client_settings: UiPath client settings. Defaults to environment-based settings.
        retry_config: Custom retry configuration.
        logger: Logger instance for request/response logging.
        **kwargs: Additional arguments passed to AsyncAnthropicFoundry client.
    """

    def __init__(
        self,
        *,
        model_name: str,
        byo_connection_id: str | None = None,
        client_settings: UiPathBaseSettings | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        client_settings = client_settings or get_default_client_settings()
        api_config = _build_api_config(vendor_type="azure")
        httpx_client = UiPathHttpxAsyncClient(
            model_name=model_name,
            byo_connection_id=byo_connection_id,
            api_config=api_config,
            timeout=kwargs.pop("timeout", None),
            max_retries=kwargs.pop("max_retries", None),
            retry_config=retry_config,
            base_url=client_settings.build_base_url(model_name=model_name, api_config=api_config),
            headers={
                **kwargs.pop("default_headers", {}),
                **client_settings.build_auth_headers(model_name=model_name, api_config=api_config),
            },
            logger=logger,
            auth=client_settings.build_auth_pipeline(),
        )
        super().__init__(
            api_key="PLACEHOLDER",
            timeout=None,
            max_retries=0,
            http_client=httpx_client,
            **kwargs,
        )
