"""LiteLLM client configurations for LangChain integration tests.

Tests across multiple providers: OpenAI (chat-completions + responses),
Gemini, Bedrock (invoke + converse), and Vertex AI Claude.
"""

import json
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.clients.litellm.chat_models import UiPathChatLiteLLM
from uipath_langchain_client.clients.litellm.embeddings import UiPathLiteLLMEmbeddings

from uipath.llm_client.settings import UiPathBaseSettings
from uipath.llm_client.settings.constants import ApiFlavor

OPENAI_CONFIGS = [
    {"model_class": UiPathChatLiteLLM, "model_kwargs": {"api_flavor": ApiFlavor.CHAT_COMPLETIONS}},
]

OPENAI_RESPONSES_CONFIGS = [
    {"model_class": UiPathChatLiteLLM, "model_kwargs": {"api_flavor": ApiFlavor.RESPONSES}},
]

GEMINI_CONFIGS = [
    {"model_class": UiPathChatLiteLLM},
]

BEDROCK_INVOKE_CONFIGS = [
    {"model_class": UiPathChatLiteLLM},
]

BEDROCK_CONVERSE_CONFIGS = [
    {"model_class": UiPathChatLiteLLM, "model_kwargs": {"api_flavor": ApiFlavor.CONVERSE}},
]

VERTEX_CLAUDE_CONFIGS = [
    {"model_class": UiPathChatLiteLLM},
]

COMPLETIONS_MODELS_WITH_CONFIGS = {
    "gpt-5.2-2025-12-11": OPENAI_CONFIGS,
    "gemini-2.5-flash": GEMINI_CONFIGS,
    "gemini-3-flash-preview": GEMINI_CONFIGS,
    "anthropic.claude-sonnet-4-5-20250929-v1:0": BEDROCK_INVOKE_CONFIGS,
    "claude-sonnet-4-5@20250929": VERTEX_CLAUDE_CONFIGS,
}


@pytest.fixture(
    scope="session",
    params=[
        (model_name, model_config)
        for model_name, model_configs in COMPLETIONS_MODELS_WITH_CONFIGS.items()
        for model_config in model_configs
    ],
    ids=[
        f"{model_name}_{json.dumps(model_config.get('model_kwargs', {}))}"
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
        "settings": client_settings,
    }


EMBEDDINGS_MODELS_WITH_CONFIGS = {
    "text-embedding-3-large": [
        {"model_class": UiPathLiteLLMEmbeddings},
    ],
}


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
    request: pytest.FixtureRequest,
    client_settings: UiPathBaseSettings,
) -> tuple[type[Embeddings], dict[str, Any]]:
    model_name, model_config = request.param
    model_class = model_config["model_class"]
    model_kwargs = model_config.get("model_kwargs", {})
    return model_class, {
        "model": model_name,
        **model_kwargs,
        "settings": client_settings,
    }
