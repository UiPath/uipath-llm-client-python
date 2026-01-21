import json
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_langchain_client.clients.bedrock.embeddings import UiPathBedrockEmbeddings
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.google.embeddings import UiPathGoogleGenerativeAIEmbeddings
from uipath_langchain_client.clients.normalized.chat_models import UiPathNormalizedChatModel
from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
)
from uipath_langchain_client.clients.openai.embeddings import (
    UiPathAzureOpenAIEmbeddings,
    UiPathOpenAIEmbeddings,
)
from uipath_langchain_client.clients.vertexai.chat_models import UiPathChatAnthropicVertex

from uipath_llm_client.settings import UiPathBaseSettings

GPT_MODELS_NON_REASONING_CONFIGS = [
    {"model_class": UiPathAzureChatOpenAI},
    {"model_class": UiPathAzureChatOpenAI, "model_kwargs": {"use_responses_api": True}},
    # {"model_class": UiPathNormalizedChatModel},
]

GPT_MODELS_WITH_REASONING_CONFIGS = [
    {"model_class": UiPathAzureChatOpenAI, "model_kwargs": {"reasoning_effort": "low"}},
    {
        "model_class": UiPathAzureChatOpenAI,
        "model_kwargs": {
            "reasoning": {
                "effort": "low",
                "summary": "auto",
            },
            "verbosity": "low",
        },
    },
    # {"model_class": UiPathNormalizedChatModel, "model_kwargs": {"reasoning_effort": "low"}},
]

GEMINI_2_5_CONFIGS = [
    {
        "model_class": UiPathChatGoogleGenerativeAI,
        "model_kwargs": {"thinking_budget": 128, "include_thoughts": False},
    },
    {
        "model_class": UiPathChatGoogleGenerativeAI,
        "model_kwargs": {"thinking_budget": 128, "include_thoughts": True},
    },
    # {
    #     "model_class": UiPathNormalizedChatModel,
    #     "model_kwargs": {"thinking_budget": 128, "include_thoughts": True},
    # },
]

GEMINI_3_CONFIGS = [
    {
        "model_class": UiPathChatGoogleGenerativeAI,
        "model_kwargs": {"thinking_level": "low", "include_thoughts": False},
    },
    {
        "model_class": UiPathChatGoogleGenerativeAI,
        "model_kwargs": {"thinking_level": "low", "include_thoughts": True},
    },
    # {
    #     "model_class": UiPathNormalizedChatModel,
    #     "model_kwargs": {"thinking_level": "low", "include_thoughts": True},
    # },
]

CLAUDE_MODELS_VERTEXAI_CONFIGS = [
    {
        "model_class": UiPathChatAnthropicVertex,
    },
    {
        "model_class": UiPathChatAnthropicVertex,
        "model_kwargs": {
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    },
    {
        "model_class": UiPathChatAnthropic,
        "model_kwargs": {
            "vendor_type": "vertexai",
        },
    },
    {
        "model_class": UiPathChatAnthropic,
        "model_kwargs": {
            "vendor_type": "vertexai",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    },
    # {"model_class": UiPathNormalizedChatModel},
]

CLAUDE_MODELS_AWSBEDROCK_CONFIGS = [
    {
        "model_class": UiPathChatAnthropic,
        "model_kwargs": {
            "vendor_type": "awsbedrock",
        },
    },
    {
        "model_class": UiPathChatAnthropic,
        "model_kwargs": {
            "vendor_type": "awsbedrock",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    },
    {
        "model_class": UiPathChatBedrock,
    },
    {
        "model_class": UiPathChatBedrock,
        "model_kwargs": {
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    },
    {
        "model_class": UiPathChatBedrockConverse,
    },
    {
        "model_class": UiPathChatBedrockConverse,
        "model_kwargs": {
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    },
    # {"model_class": UiPathNormalizedChatModel},
]

COMPLETIONS_MODELS_WITH_CONFIGS = {
    "gpt-4o-2024-11-20": GPT_MODELS_NON_REASONING_CONFIGS,
    "gpt-4.1-2025-04-14": GPT_MODELS_NON_REASONING_CONFIGS,
    "gpt-5-2025-08-07": GPT_MODELS_WITH_REASONING_CONFIGS,
    "gpt-5.1-2025-11-13": GPT_MODELS_WITH_REASONING_CONFIGS,
    "gpt-5.2-2025-12-11": GPT_MODELS_WITH_REASONING_CONFIGS,
    "gemini-2.5-flash": GEMINI_2_5_CONFIGS,
    "gemini-2.5-pro": GEMINI_2_5_CONFIGS,
    "gemini-3-flash-preview": GEMINI_3_CONFIGS,
    "gemini-3-pro-preview": GEMINI_3_CONFIGS,
    "claude-sonnet-4-5@20250929": CLAUDE_MODELS_VERTEXAI_CONFIGS,
    "claude-haiku-4-5@20251001": CLAUDE_MODELS_VERTEXAI_CONFIGS,
    "claude-opus-4-5@20251101": CLAUDE_MODELS_VERTEXAI_CONFIGS,
    "anthropic.claude-sonnet-4-5-20250929-v1:0": CLAUDE_MODELS_AWSBEDROCK_CONFIGS,
    "anthropic.claude-haiku-4-5-20251001-v1:0": CLAUDE_MODELS_AWSBEDROCK_CONFIGS,
    "anthropic.claude-opus-4-5-20251101-v1:0": CLAUDE_MODELS_AWSBEDROCK_CONFIGS,
}

COMPLETION_MODEL_NAMES = list(COMPLETIONS_MODELS_WITH_CONFIGS.keys())


COMPLETION_CLIENTS_CLASSES = [
    UiPathNormalizedChatModel,
    UiPathChatOpenAI,
    UiPathAzureChatOpenAI,
    # UiPathAzureAIChatCompletionsModel,
    UiPathChatGoogleGenerativeAI,
    UiPathChatAnthropicVertex,
    UiPathChatAnthropic,
    UiPathChatBedrock,
    # UiPathChatBedrockConverse,
]
EMBEDDINGS_CLIENTS_CLASSES = [
    # UiPathNormalizedEmbeddings,
    UiPathOpenAIEmbeddings,
    UiPathAzureOpenAIEmbeddings,
    # UiPathAzureAIEmbeddingsModel,
    UiPathGoogleGenerativeAIEmbeddings,
    UiPathBedrockEmbeddings,
]


@pytest.fixture(
    scope="session",
    params=[
        (model_name, model_config)
        for model_name, model_configs in COMPLETIONS_MODELS_WITH_CONFIGS.items()
        for model_config in model_configs
    ],
    ids=[
        f"{model_name}_{model_config['model_class'].__name__}_{json.dumps(model_config.get('model_kwargs', {}))}"
        for model_name, model_configs in COMPLETIONS_MODELS_WITH_CONFIGS.items()
        for model_config in model_configs
    ],
)
def completions_config(
    request: pytest.FixtureRequest,
    client_settings: UiPathBaseSettings,
) -> tuple[type[BaseChatModel], dict[str, Any]]:
    model_name, model_config = request.param
    model_class = model_config["model_class"]
    model_kwargs = model_config.get("model_kwargs", {})
    return model_class, {
        "model": model_name,
        **model_kwargs,
        "client_settings": client_settings,
    }


EMBEDDINGS_MODELS_WITH_CONFIGS = {
    "text-embedding-3-large": [
        {"model_class": UiPathAzureOpenAIEmbeddings},
        # {"model_class": UiPathNormalizedEmbeddings},
    ],
    "gemini-embedding-001": [
        {"model_class": UiPathGoogleGenerativeAIEmbeddings},
        # {"model_class": UiPathNormalizedEmbeddings},
    ],
}

EMBEDDING_MODEL_NAMES = list(EMBEDDINGS_MODELS_WITH_CONFIGS.keys())


@pytest.fixture(
    scope="session",
    params=[
        (model_name, model_config)
        for model_name, model_configs in EMBEDDINGS_MODELS_WITH_CONFIGS.items()
        for model_config in model_configs
    ],
    ids=[
        f"{model_name}_{model_config['model_class'].__name__}"
        for model_name, model_configs in EMBEDDINGS_MODELS_WITH_CONFIGS.items()
        for model_config in model_configs
    ],
)
def embeddings_config(
    request: pytest.FixtureRequest, client_settings: UiPathBaseSettings
) -> tuple[type[Embeddings], dict[str, Any]]:
    model_name, model_config = request.param
    model_class = model_config["model_class"]
    model_kwargs = model_config.get("model_kwargs", {})
    return model_class, {
        "model": model_name,
        **model_kwargs,
        "client_settings": client_settings,
    }
