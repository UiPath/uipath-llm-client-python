#!/usr/bin/env python3
"""Smoke tests for uipath-langchain-client package.

This script is executed by the CD pipeline to verify that the package
is correctly built and all essential imports work before publishing.

Run with: python tests/langchain/langchain_smoke_test.py
"""

import sys


def test_main_package_imports():
    """Test that the main package and its exports can be imported."""
    print("Testing main package imports...")

    from uipath_langchain_client import (
        AgentHubSettings,
        LLMGatewaySettings,
        __version__,
        get_chat_model,
        get_default_client_settings,
        get_embedding_model,
    )

    # Verify version is a non-empty string
    assert isinstance(__version__, str), f"__version__ should be str, got {type(__version__)}"
    assert len(__version__) > 0, "__version__ should not be empty"
    print(f"  Package version: {__version__}")

    # Verify factory functions are callable
    assert callable(get_chat_model), "get_chat_model should be callable"
    assert callable(get_embedding_model), "get_embedding_model should be callable"
    assert callable(get_default_client_settings), "get_default_client_settings should be callable"
    print("  Factory functions are callable")

    # Verify settings classes exist and are types
    assert isinstance(AgentHubSettings, type), "AgentHubSettings should be a class"
    assert isinstance(LLMGatewaySettings, type), "LLMGatewaySettings should be a class"
    print("  Settings classes are importable")

    print("  Main package imports OK")


def test_settings_module_imports():
    """Test that the settings module exports can be imported directly."""
    print("Testing settings module imports...")

    from uipath_langchain_client.settings import (
        AgentHubSettings,
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

    # Verify inheritance - settings should inherit from UiPathBaseSettings
    assert issubclass(AgentHubSettings, UiPathBaseSettings), (
        "AgentHubSettings should inherit from UiPathBaseSettings"
    )
    assert issubclass(LLMGatewaySettings, UiPathBaseSettings), (
        "LLMGatewaySettings should inherit from UiPathBaseSettings"
    )

    print("  Settings module imports OK")


def test_factory_module_imports():
    """Test that the factory module exports can be imported directly."""
    print("Testing factory module imports...")

    from uipath_langchain_client.factory import get_chat_model, get_embedding_model

    # Verify factory functions are callable
    assert callable(get_chat_model), "get_chat_model should be callable"
    assert callable(get_embedding_model), "get_embedding_model should be callable"

    # Verify they have proper function signatures (checking they accept model_name)
    import inspect

    chat_sig = inspect.signature(get_chat_model)
    embed_sig = inspect.signature(get_embedding_model)

    assert "model_name" in chat_sig.parameters, "get_chat_model should have model_name parameter"
    assert "model" in embed_sig.parameters, "get_embedding_model should have model parameter"
    assert "client_settings" in chat_sig.parameters, (
        "get_chat_model should have client_settings parameter"
    )
    assert "client_settings" in embed_sig.parameters, (
        "get_embedding_model should have client_settings parameter"
    )

    print("  Factory module imports OK")


def test_base_client_module_imports():
    """Test that the base_client module exports can be imported directly."""
    print("Testing base_client module imports...")

    from uipath_langchain_client.base_client import UiPathBaseLLMClient

    # Verify it's a class
    assert isinstance(UiPathBaseLLMClient, type), "UiPathBaseLLMClient should be a class"

    # Verify it has essential methods
    assert hasattr(UiPathBaseLLMClient, "uipath_request"), (
        "UiPathBaseLLMClient should have uipath_request method"
    )
    assert hasattr(UiPathBaseLLMClient, "uipath_arequest"), (
        "UiPathBaseLLMClient should have uipath_arequest method"
    )
    assert hasattr(UiPathBaseLLMClient, "uipath_stream"), (
        "UiPathBaseLLMClient should have uipath_stream method"
    )
    assert hasattr(UiPathBaseLLMClient, "uipath_astream"), (
        "UiPathBaseLLMClient should have uipath_astream method"
    )

    print("  Base client module imports OK")


def test_openai_client_imports():
    """Test OpenAI client imports."""
    print("Testing OpenAI client imports...")

    from uipath_langchain_client.clients.openai import (
        UiPathAzureChatOpenAI,
        UiPathAzureOpenAIEmbeddings,
        UiPathChatOpenAI,
        UiPathOpenAIEmbeddings,
    )

    assert isinstance(UiPathChatOpenAI, type), "UiPathChatOpenAI should be a class"
    assert isinstance(UiPathOpenAIEmbeddings, type), "UiPathOpenAIEmbeddings should be a class"
    assert isinstance(UiPathAzureChatOpenAI, type), "UiPathAzureChatOpenAI should be a class"
    assert isinstance(UiPathAzureOpenAIEmbeddings, type), (
        "UiPathAzureOpenAIEmbeddings should be a class"
    )

    print("  OpenAI client imports OK")


def test_anthropic_client_imports():
    """Test Anthropic client imports."""
    print("Testing Anthropic client imports...")

    from uipath_langchain_client.clients.anthropic import UiPathChatAnthropic

    assert isinstance(UiPathChatAnthropic, type), "UiPathChatAnthropic should be a class"

    print("  Anthropic client imports OK")


def test_google_client_imports():
    """Test Google client imports."""
    print("Testing Google client imports...")

    from uipath_langchain_client.clients.google import (
        UiPathChatGoogleGenerativeAI,
        UiPathGoogleGenerativeAIEmbeddings,
    )

    assert isinstance(UiPathChatGoogleGenerativeAI, type), (
        "UiPathChatGoogleGenerativeAI should be a class"
    )
    assert isinstance(UiPathGoogleGenerativeAIEmbeddings, type), (
        "UiPathGoogleGenerativeAIEmbeddings should be a class"
    )

    print("  Google client imports OK")


def test_bedrock_client_imports():
    """Test Bedrock client imports."""
    print("Testing Bedrock client imports...")

    from uipath_langchain_client.clients.bedrock import (
        UiPathBedrockEmbeddings,
        UiPathChatBedrock,
        UiPathChatBedrockConverse,
    )

    assert isinstance(UiPathChatBedrock, type), "UiPathChatBedrock should be a class"
    assert isinstance(UiPathChatBedrockConverse, type), (
        "UiPathChatBedrockConverse should be a class"
    )
    assert isinstance(UiPathBedrockEmbeddings, type), "UiPathBedrockEmbeddings should be a class"

    print("  Bedrock client imports OK")


def test_azure_client_imports():
    """Test Azure client imports."""
    print("Testing Azure client imports...")

    from uipath_langchain_client.clients.azure import (
        UiPathAzureAIChatCompletionsModel,
        UiPathAzureAIEmbeddingsModel,
    )

    assert isinstance(UiPathAzureAIChatCompletionsModel, type), (
        "UiPathAzureAIChatCompletionsModel should be a class"
    )
    assert isinstance(UiPathAzureAIEmbeddingsModel, type), (
        "UiPathAzureAIEmbeddingsModel should be a class"
    )

    print("  Azure client imports OK")


def test_normalized_client_imports():
    """Test Normalized client imports."""
    print("Testing Normalized client imports...")

    from uipath_langchain_client.clients.normalized import (
        UiPathNormalizedChatModel,
        UiPathNormalizedEmbeddings,
    )

    assert isinstance(UiPathNormalizedChatModel, type), (
        "UiPathNormalizedChatModel should be a class"
    )
    assert isinstance(UiPathNormalizedEmbeddings, type), (
        "UiPathNormalizedEmbeddings should be a class"
    )

    print("  Normalized client imports OK")


def test_vertexai_client_imports():
    """Test VertexAI client imports."""
    print("Testing VertexAI client imports...")

    from uipath_langchain_client.clients.vertexai import UiPathChatAnthropicVertex

    assert isinstance(UiPathChatAnthropicVertex, type), (
        "UiPathChatAnthropicVertex should be a class"
    )

    print("  VertexAI client imports OK")


def test_inheritance_openai():
    """Test OpenAI classes inherit from LangChain and UiPath base classes."""
    print("Testing OpenAI inheritance...")

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from uipath_langchain_client.base_client import UiPathBaseLLMClient
    from uipath_langchain_client.clients.openai import (
        UiPathAzureChatOpenAI,
        UiPathAzureOpenAIEmbeddings,
        UiPathChatOpenAI,
        UiPathOpenAIEmbeddings,
    )

    # Chat models
    assert issubclass(UiPathChatOpenAI, BaseChatModel), (
        "UiPathChatOpenAI should inherit from BaseChatModel"
    )
    assert issubclass(UiPathChatOpenAI, UiPathBaseLLMClient), (
        "UiPathChatOpenAI should inherit from UiPathBaseLLMClient"
    )
    assert issubclass(UiPathAzureChatOpenAI, BaseChatModel), (
        "UiPathAzureChatOpenAI should inherit from BaseChatModel"
    )
    assert issubclass(UiPathAzureChatOpenAI, UiPathBaseLLMClient), (
        "UiPathAzureChatOpenAI should inherit from UiPathBaseLLMClient"
    )

    # Embeddings
    assert issubclass(UiPathOpenAIEmbeddings, Embeddings), (
        "UiPathOpenAIEmbeddings should inherit from Embeddings"
    )
    assert issubclass(UiPathOpenAIEmbeddings, UiPathBaseLLMClient), (
        "UiPathOpenAIEmbeddings should inherit from UiPathBaseLLMClient"
    )
    assert issubclass(UiPathAzureOpenAIEmbeddings, Embeddings), (
        "UiPathAzureOpenAIEmbeddings should inherit from Embeddings"
    )
    assert issubclass(UiPathAzureOpenAIEmbeddings, UiPathBaseLLMClient), (
        "UiPathAzureOpenAIEmbeddings should inherit from UiPathBaseLLMClient"
    )

    print("  OpenAI inheritance OK")


def test_inheritance_anthropic():
    """Test Anthropic classes inherit from LangChain and UiPath base classes."""
    print("Testing Anthropic inheritance...")

    from langchain_core.language_models.chat_models import BaseChatModel
    from uipath_langchain_client.base_client import UiPathBaseLLMClient
    from uipath_langchain_client.clients.anthropic import UiPathChatAnthropic

    assert issubclass(UiPathChatAnthropic, BaseChatModel), (
        "UiPathChatAnthropic should inherit from BaseChatModel"
    )
    assert issubclass(UiPathChatAnthropic, UiPathBaseLLMClient), (
        "UiPathChatAnthropic should inherit from UiPathBaseLLMClient"
    )

    print("  Anthropic inheritance OK")


def test_inheritance_google():
    """Test Google classes inherit from LangChain and UiPath base classes."""
    print("Testing Google inheritance...")

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from uipath_langchain_client.base_client import UiPathBaseLLMClient
    from uipath_langchain_client.clients.google import (
        UiPathChatGoogleGenerativeAI,
        UiPathGoogleGenerativeAIEmbeddings,
    )

    assert issubclass(UiPathChatGoogleGenerativeAI, BaseChatModel), (
        "UiPathChatGoogleGenerativeAI should inherit from BaseChatModel"
    )
    assert issubclass(UiPathChatGoogleGenerativeAI, UiPathBaseLLMClient), (
        "UiPathChatGoogleGenerativeAI should inherit from UiPathBaseLLMClient"
    )
    assert issubclass(UiPathGoogleGenerativeAIEmbeddings, Embeddings), (
        "UiPathGoogleGenerativeAIEmbeddings should inherit from Embeddings"
    )
    assert issubclass(UiPathGoogleGenerativeAIEmbeddings, UiPathBaseLLMClient), (
        "UiPathGoogleGenerativeAIEmbeddings should inherit from UiPathBaseLLMClient"
    )

    print("  Google inheritance OK")


def test_inheritance_bedrock():
    """Test Bedrock classes inherit from LangChain and UiPath base classes."""
    print("Testing Bedrock inheritance...")

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from uipath_langchain_client.base_client import UiPathBaseLLMClient
    from uipath_langchain_client.clients.bedrock import (
        UiPathBedrockEmbeddings,
        UiPathChatBedrock,
        UiPathChatBedrockConverse,
    )

    assert issubclass(UiPathChatBedrock, BaseChatModel), (
        "UiPathChatBedrock should inherit from BaseChatModel"
    )
    assert issubclass(UiPathChatBedrock, UiPathBaseLLMClient), (
        "UiPathChatBedrock should inherit from UiPathBaseLLMClient"
    )
    assert issubclass(UiPathChatBedrockConverse, BaseChatModel), (
        "UiPathChatBedrockConverse should inherit from BaseChatModel"
    )
    assert issubclass(UiPathChatBedrockConverse, UiPathBaseLLMClient), (
        "UiPathChatBedrockConverse should inherit from UiPathBaseLLMClient"
    )
    assert issubclass(UiPathBedrockEmbeddings, Embeddings), (
        "UiPathBedrockEmbeddings should inherit from Embeddings"
    )
    assert issubclass(UiPathBedrockEmbeddings, UiPathBaseLLMClient), (
        "UiPathBedrockEmbeddings should inherit from UiPathBaseLLMClient"
    )

    print("  Bedrock inheritance OK")


def test_inheritance_azure():
    """Test Azure classes inherit from LangChain and UiPath base classes."""
    print("Testing Azure inheritance...")

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from uipath_langchain_client.base_client import UiPathBaseLLMClient
    from uipath_langchain_client.clients.azure import (
        UiPathAzureAIChatCompletionsModel,
        UiPathAzureAIEmbeddingsModel,
    )

    assert issubclass(UiPathAzureAIChatCompletionsModel, BaseChatModel), (
        "UiPathAzureAIChatCompletionsModel should inherit from BaseChatModel"
    )
    assert issubclass(UiPathAzureAIChatCompletionsModel, UiPathBaseLLMClient), (
        "UiPathAzureAIChatCompletionsModel should inherit from UiPathBaseLLMClient"
    )
    assert issubclass(UiPathAzureAIEmbeddingsModel, Embeddings), (
        "UiPathAzureAIEmbeddingsModel should inherit from Embeddings"
    )
    assert issubclass(UiPathAzureAIEmbeddingsModel, UiPathBaseLLMClient), (
        "UiPathAzureAIEmbeddingsModel should inherit from UiPathBaseLLMClient"
    )

    print("  Azure inheritance OK")


def test_inheritance_normalized():
    """Test Normalized classes inherit from LangChain and UiPath base classes."""
    print("Testing Normalized inheritance...")

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from uipath_langchain_client.base_client import UiPathBaseLLMClient
    from uipath_langchain_client.clients.normalized import (
        UiPathNormalizedChatModel,
        UiPathNormalizedEmbeddings,
    )

    assert issubclass(UiPathNormalizedChatModel, BaseChatModel), (
        "UiPathNormalizedChatModel should inherit from BaseChatModel"
    )
    assert issubclass(UiPathNormalizedChatModel, UiPathBaseLLMClient), (
        "UiPathNormalizedChatModel should inherit from UiPathBaseLLMClient"
    )
    assert issubclass(UiPathNormalizedEmbeddings, Embeddings), (
        "UiPathNormalizedEmbeddings should inherit from Embeddings"
    )
    assert issubclass(UiPathNormalizedEmbeddings, UiPathBaseLLMClient), (
        "UiPathNormalizedEmbeddings should inherit from UiPathBaseLLMClient"
    )

    print("  Normalized inheritance OK")


def test_inheritance_vertexai():
    """Test VertexAI classes inherit from LangChain and UiPath base classes."""
    print("Testing VertexAI inheritance...")

    from langchain_core.language_models.chat_models import BaseChatModel
    from uipath_langchain_client.base_client import UiPathBaseLLMClient
    from uipath_langchain_client.clients.vertexai import UiPathChatAnthropicVertex

    assert issubclass(UiPathChatAnthropicVertex, BaseChatModel), (
        "UiPathChatAnthropicVertex should inherit from BaseChatModel"
    )
    assert issubclass(UiPathChatAnthropicVertex, UiPathBaseLLMClient), (
        "UiPathChatAnthropicVertex should inherit from UiPathBaseLLMClient"
    )

    print("  VertexAI inheritance OK")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running uipath-langchain-client smoke tests")
    print("=" * 60)

    tests = [
        test_main_package_imports,
        test_settings_module_imports,
        test_factory_module_imports,
        test_base_client_module_imports,
        test_openai_client_imports,
        test_anthropic_client_imports,
        test_google_client_imports,
        test_bedrock_client_imports,
        test_azure_client_imports,
        test_normalized_client_imports,
        test_vertexai_client_imports,
        test_inheritance_openai,
        test_inheritance_anthropic,
        test_inheritance_google,
        test_inheritance_bedrock,
        test_inheritance_azure,
        test_inheritance_normalized,
        test_inheritance_vertexai,
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
