#!/usr/bin/env python3
"""Smoke tests for uipath-llm-client package.

This script is executed by the CD pipeline to verify that the package
is correctly built and all essential imports work before publishing.

Run with: python tests/core/core_smoke_test.py
"""

import sys


def test_main_package_imports():
    """Test that the main package and its exports can be imported."""
    print("Testing main package imports...")

    from uipath_llm_client import (
        AgentHubSettings,
        LLMGatewaySettings,
        RetryConfig,
        UiPathHttpxAsyncClient,
        UiPathHttpxClient,
        __version__,
        get_default_client_settings,
    )

    # Verify version is a non-empty string
    assert isinstance(__version__, str), f"__version__ should be str, got {type(__version__)}"
    assert len(__version__) > 0, "__version__ should not be empty"
    print(f"  Package version: {__version__}")

    # Verify factory function is callable
    assert callable(get_default_client_settings), "get_default_client_settings should be callable"
    print("  Factory function is callable")

    # Verify settings classes are types
    assert isinstance(AgentHubSettings, type), "AgentHubSettings should be a class"
    assert isinstance(LLMGatewaySettings, type), "LLMGatewaySettings should be a class"
    print("  Settings classes are importable")

    # Verify HTTPX clients are types
    assert isinstance(UiPathHttpxClient, type), "UiPathHttpxClient should be a class"
    assert isinstance(UiPathHttpxAsyncClient, type), "UiPathHttpxAsyncClient should be a class"
    print("  HTTPX client classes are importable")

    # Verify RetryConfig is a type (TypedDict)
    assert RetryConfig is not None, "RetryConfig should be importable"
    print("  RetryConfig is importable")

    print("  Main package imports OK")


def test_exception_imports():
    """Test that all exception classes can be imported from main package."""
    print("Testing exception imports...")

    from uipath_llm_client import (
        UiPathAPIError,
        UiPathAuthenticationError,
        UiPathBadRequestError,
        UiPathConflictError,
        UiPathGatewayTimeoutError,
        UiPathInternalServerError,
        UiPathNotFoundError,
        UiPathPermissionDeniedError,
        UiPathRateLimitError,
        UiPathRequestTooLargeError,
        UiPathServiceUnavailableError,
        UiPathTooManyRequestsError,
        UiPathUnprocessableEntityError,
    )

    # Verify all exceptions are types
    exceptions = [
        UiPathAPIError,
        UiPathAuthenticationError,
        UiPathBadRequestError,
        UiPathConflictError,
        UiPathGatewayTimeoutError,
        UiPathInternalServerError,
        UiPathNotFoundError,
        UiPathPermissionDeniedError,
        UiPathRateLimitError,
        UiPathRequestTooLargeError,
        UiPathServiceUnavailableError,
        UiPathTooManyRequestsError,
        UiPathUnprocessableEntityError,
    ]

    for exc in exceptions:
        assert isinstance(exc, type), f"{exc.__name__} should be a class"

    print(f"  All {len(exceptions)} exception classes are importable")
    print("  Exception imports OK")


def test_settings_module_imports():
    """Test that the settings module exports can be imported directly."""
    print("Testing settings module imports...")

    from uipath_llm_client.settings import (
        UIPATH_LLM_BACKEND_ENV,
        AgentHubSettings,
        BackendType,
        LLMGatewaySettings,
        UiPathAPIConfig,
        UiPathBaseSettings,
        get_default_client_settings,
    )

    # Verify settings classes are types
    assert isinstance(AgentHubSettings, type), "AgentHubSettings should be a class"
    assert isinstance(LLMGatewaySettings, type), "LLMGatewaySettings should be a class"
    assert isinstance(UiPathAPIConfig, type), "UiPathAPIConfig should be a class"
    assert isinstance(UiPathBaseSettings, type), "UiPathBaseSettings should be a class"

    # Verify factory function is callable
    assert callable(get_default_client_settings), "get_default_client_settings should be callable"

    # Verify constants
    assert isinstance(UIPATH_LLM_BACKEND_ENV, str), "UIPATH_LLM_BACKEND_ENV should be a string"
    assert BackendType is not None, "BackendType should be importable"

    # Verify inheritance
    assert issubclass(AgentHubSettings, UiPathBaseSettings), (
        "AgentHubSettings should inherit from UiPathBaseSettings"
    )
    assert issubclass(LLMGatewaySettings, UiPathBaseSettings), (
        "LLMGatewaySettings should inherit from UiPathBaseSettings"
    )

    print("  Settings module imports OK")


def test_httpx_client_module_imports():
    """Test that the httpx_client module exports can be imported directly."""
    print("Testing httpx_client module imports...")

    from httpx import AsyncClient, Client

    from uipath_llm_client.httpx_client import (
        UiPathHttpxAsyncClient,
        UiPathHttpxClient,
        build_routing_headers,
    )

    # Verify classes are types
    assert isinstance(UiPathHttpxClient, type), "UiPathHttpxClient should be a class"
    assert isinstance(UiPathHttpxAsyncClient, type), "UiPathHttpxAsyncClient should be a class"

    # Verify function is callable
    assert callable(build_routing_headers), "build_routing_headers should be callable"

    # Verify inheritance from httpx
    assert issubclass(UiPathHttpxClient, Client), (
        "UiPathHttpxClient should inherit from httpx.Client"
    )
    assert issubclass(UiPathHttpxAsyncClient, AsyncClient), (
        "UiPathHttpxAsyncClient should inherit from httpx.AsyncClient"
    )

    print("  httpx_client module imports OK")


def test_exceptions_module_imports():
    """Test that the exceptions module exports can be imported directly."""
    print("Testing exceptions module imports...")

    from httpx import HTTPStatusError

    from uipath_llm_client.utils.exceptions import (
        UiPathAPIError,
        UiPathAuthenticationError,
        UiPathBadRequestError,
        UiPathConflictError,
        UiPathGatewayTimeoutError,
        UiPathInternalServerError,
        UiPathNotFoundError,
        UiPathPermissionDeniedError,
        UiPathRateLimitError,
        UiPathRequestTooLargeError,
        UiPathServiceUnavailableError,
        UiPathTooManyRequestsError,
        UiPathUnprocessableEntityError,
        patch_raise_for_status,
    )

    # Verify base exception inherits from HTTPStatusError
    assert issubclass(UiPathAPIError, HTTPStatusError), (
        "UiPathAPIError should inherit from HTTPStatusError"
    )

    # Verify all specific exceptions inherit from UiPathAPIError
    specific_exceptions = [
        UiPathAuthenticationError,
        UiPathBadRequestError,
        UiPathConflictError,
        UiPathGatewayTimeoutError,
        UiPathInternalServerError,
        UiPathNotFoundError,
        UiPathPermissionDeniedError,
        UiPathRateLimitError,
        UiPathRequestTooLargeError,
        UiPathServiceUnavailableError,
        UiPathTooManyRequestsError,
        UiPathUnprocessableEntityError,
    ]

    for exc in specific_exceptions:
        assert issubclass(exc, UiPathAPIError), f"{exc.__name__} should inherit from UiPathAPIError"

    # Verify patch function is callable
    assert callable(patch_raise_for_status), "patch_raise_for_status should be callable"

    # Verify status codes are set
    assert UiPathBadRequestError.status_code == 400, (
        "UiPathBadRequestError should have status_code 400"
    )
    assert UiPathAuthenticationError.status_code == 401, (
        "UiPathAuthenticationError should have status_code 401"
    )
    assert UiPathPermissionDeniedError.status_code == 403, (
        "UiPathPermissionDeniedError should have status_code 403"
    )
    assert UiPathNotFoundError.status_code == 404, "UiPathNotFoundError should have status_code 404"
    assert UiPathRateLimitError.status_code == 429, (
        "UiPathRateLimitError should have status_code 429"
    )
    assert UiPathInternalServerError.status_code == 500, (
        "UiPathInternalServerError should have status_code 500"
    )

    print("  Exceptions module imports OK")


def test_retry_module_imports():
    """Test that the retry module exports can be imported directly."""
    print("Testing retry module imports...")

    from httpx import AsyncHTTPTransport, HTTPTransport

    from uipath_llm_client.utils.retry import (
        RetryableAsyncHTTPTransport,
        RetryableHTTPTransport,
        RetryConfig,
    )

    # Verify classes are types
    assert isinstance(RetryableHTTPTransport, type), "RetryableHTTPTransport should be a class"
    assert isinstance(RetryableAsyncHTTPTransport, type), (
        "RetryableAsyncHTTPTransport should be a class"
    )

    # Verify RetryConfig is importable (TypedDict)
    assert RetryConfig is not None, "RetryConfig should be importable"

    # Verify inheritance from httpx transports
    assert issubclass(RetryableHTTPTransport, HTTPTransport), (
        "RetryableHTTPTransport should inherit from HTTPTransport"
    )
    assert issubclass(RetryableAsyncHTTPTransport, AsyncHTTPTransport), (
        "RetryableAsyncHTTPTransport should inherit from AsyncHTTPTransport"
    )

    print("  Retry module imports OK")


def test_openai_client_imports():
    """Test OpenAI client imports."""
    print("Testing OpenAI client imports...")

    from uipath_llm_client.clients.openai import (
        UiPathAsyncAzureOpenAI,
        UiPathAsyncOpenAI,
        UiPathAzureOpenAI,
        UiPathOpenAI,
    )

    # Verify all are types
    assert isinstance(UiPathOpenAI, type), "UiPathOpenAI should be a class"
    assert isinstance(UiPathAsyncOpenAI, type), "UiPathAsyncOpenAI should be a class"
    assert isinstance(UiPathAzureOpenAI, type), "UiPathAzureOpenAI should be a class"
    assert isinstance(UiPathAsyncAzureOpenAI, type), "UiPathAsyncAzureOpenAI should be a class"

    print("  OpenAI client imports OK")


def test_anthropic_client_imports():
    """Test Anthropic client imports."""
    print("Testing Anthropic client imports...")

    from uipath_llm_client.clients.anthropic import (
        UiPathAnthropic,
        UiPathAnthropicBedrock,
        UiPathAnthropicFoundry,
        UiPathAnthropicVertex,
        UiPathAsyncAnthropic,
        UiPathAsyncAnthropicBedrock,
        UiPathAsyncAnthropicFoundry,
        UiPathAsyncAnthropicVertex,
    )

    # Verify all are types
    clients = [
        UiPathAnthropic,
        UiPathAsyncAnthropic,
        UiPathAnthropicBedrock,
        UiPathAsyncAnthropicBedrock,
        UiPathAnthropicVertex,
        UiPathAsyncAnthropicVertex,
        UiPathAnthropicFoundry,
        UiPathAsyncAnthropicFoundry,
    ]

    for client in clients:
        assert isinstance(client, type), f"{client.__name__} should be a class"

    print(f"  All {len(clients)} Anthropic client classes are importable")
    print("  Anthropic client imports OK")


def test_google_client_imports():
    """Test Google client imports."""
    print("Testing Google client imports...")

    from uipath_llm_client.clients.google import UiPathGoogle

    assert isinstance(UiPathGoogle, type), "UiPathGoogle should be a class"

    print("  Google client imports OK")


def test_openai_client_inheritance():
    """Test OpenAI clients inherit from OpenAI SDK classes."""
    print("Testing OpenAI client inheritance...")

    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

    from uipath_llm_client.clients.openai import (
        UiPathAsyncAzureOpenAI,
        UiPathAsyncOpenAI,
        UiPathAzureOpenAI,
        UiPathOpenAI,
    )

    assert issubclass(UiPathOpenAI, OpenAI), "UiPathOpenAI should inherit from OpenAI"
    assert issubclass(UiPathAsyncOpenAI, AsyncOpenAI), (
        "UiPathAsyncOpenAI should inherit from AsyncOpenAI"
    )
    assert issubclass(UiPathAzureOpenAI, AzureOpenAI), (
        "UiPathAzureOpenAI should inherit from AzureOpenAI"
    )
    assert issubclass(UiPathAsyncAzureOpenAI, AsyncAzureOpenAI), (
        "UiPathAsyncAzureOpenAI should inherit from AsyncAzureOpenAI"
    )

    print("  OpenAI client inheritance OK")


def test_anthropic_client_inheritance():
    """Test Anthropic clients inherit from Anthropic SDK classes."""
    print("Testing Anthropic client inheritance...")

    from anthropic import (
        Anthropic,
        AnthropicBedrock,
        AnthropicVertex,
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicVertex,
    )

    from uipath_llm_client.clients.anthropic import (
        UiPathAnthropic,
        UiPathAnthropicBedrock,
        UiPathAnthropicVertex,
        UiPathAsyncAnthropic,
        UiPathAsyncAnthropicBedrock,
        UiPathAsyncAnthropicVertex,
    )

    assert issubclass(UiPathAnthropic, Anthropic), "UiPathAnthropic should inherit from Anthropic"
    assert issubclass(UiPathAsyncAnthropic, AsyncAnthropic), (
        "UiPathAsyncAnthropic should inherit from AsyncAnthropic"
    )
    assert issubclass(UiPathAnthropicBedrock, AnthropicBedrock), (
        "UiPathAnthropicBedrock should inherit from AnthropicBedrock"
    )
    assert issubclass(UiPathAsyncAnthropicBedrock, AsyncAnthropicBedrock), (
        "UiPathAsyncAnthropicBedrock should inherit from AsyncAnthropicBedrock"
    )
    assert issubclass(UiPathAnthropicVertex, AnthropicVertex), (
        "UiPathAnthropicVertex should inherit from AnthropicVertex"
    )
    assert issubclass(UiPathAsyncAnthropicVertex, AsyncAnthropicVertex), (
        "UiPathAsyncAnthropicVertex should inherit from AsyncAnthropicVertex"
    )

    print("  Anthropic client inheritance OK")


def test_google_client_inheritance():
    """Test Google client inherits from Google SDK class."""
    print("Testing Google client inheritance...")

    from google.genai import Client as GoogleClient

    from uipath_llm_client.clients.google import UiPathGoogle

    assert issubclass(UiPathGoogle, GoogleClient), (
        "UiPathGoogle should inherit from google.genai.Client"
    )

    print("  Google client inheritance OK")


def test_uipath_api_config():
    """Test UiPathAPIConfig can be instantiated with valid configurations."""
    print("Testing UiPathAPIConfig instantiation...")

    from uipath_llm_client.settings import UiPathAPIConfig

    # Test passthrough config (requires vendor_type)
    passthrough_config = UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="openai",
    )
    assert passthrough_config.api_type == "completions"
    assert passthrough_config.client_type == "passthrough"
    assert passthrough_config.vendor_type == "openai"

    # Test normalized config (no vendor_type required)
    normalized_config = UiPathAPIConfig(
        api_type="completions",
        client_type="normalized",
    )
    assert normalized_config.api_type == "completions"
    assert normalized_config.client_type == "normalized"

    # Test embeddings config
    embeddings_config = UiPathAPIConfig(
        api_type="embeddings",
        client_type="passthrough",
        vendor_type="vertexai",
    )
    assert embeddings_config.api_type == "embeddings"

    print("  UiPathAPIConfig instantiation OK")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running uipath-llm-client smoke tests")
    print("=" * 60)

    tests = [
        test_main_package_imports,
        test_exception_imports,
        test_settings_module_imports,
        test_httpx_client_module_imports,
        test_exceptions_module_imports,
        test_retry_module_imports,
        test_openai_client_imports,
        test_anthropic_client_imports,
        test_google_client_imports,
        test_openai_client_inheritance,
        test_anthropic_client_inheritance,
        test_google_client_inheritance,
        test_uipath_api_config,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            failed.append((test.__name__, str(e)))
            print(f"  FAILED: {e}")

    print("=" * 60)
    if failed:
        print(f"SMOKE TESTS FAILED ({len(failed)}/{len(tests)} failed)")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"ALL SMOKE TESTS PASSED ({len(tests)}/{len(tests)})")
        sys.exit(0)


if __name__ == "__main__":
    main()
