"""Normalized client configurations for integration tests.

UiPathChat is tested with ALL models (GPT, Gemini, Claude) across providers.
This is the cross-provider test with thinking configs.
"""

import json
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.normalized.embeddings import UiPathEmbeddings

from uipath.llm_client.settings import UiPathBaseSettings

GPT_NON_REASONING_CONFIGS = [
    {"model_class": UiPathChat},
]

GPT_REASONING_CONFIGS = [
    {"model_class": UiPathChat, "model_kwargs": {"reasoning_effort": "low"}},
]

GEMINI_2_5_CONFIGS = [
    {
        "model_class": UiPathChat,
        "model_kwargs": {"thinking_budget": 128, "include_thoughts": True},
    },
]

GEMINI_3_CONFIGS = [
    {
        "model_class": UiPathChat,
        "model_kwargs": {"thinking_level": "low", "include_thoughts": True},
    },
]

CLAUDE_VERTEXAI_CONFIGS = [
    {"model_class": UiPathChat},
]

CLAUDE_BEDROCK_CONFIGS = [
    {"model_class": UiPathChat},
]

COMPLETIONS_MODELS_WITH_CONFIGS = {
    "gpt-4o-2024-11-20": GPT_NON_REASONING_CONFIGS,
    "gpt-5.2-2025-12-11": GPT_REASONING_CONFIGS,
    "gemini-2.5-flash": GEMINI_2_5_CONFIGS,
    "gemini-3-flash-preview": GEMINI_3_CONFIGS,
    "claude-haiku-4-5@20251001": CLAUDE_VERTEXAI_CONFIGS,
    "anthropic.claude-haiku-4-5-20251001-v1:0": CLAUDE_BEDROCK_CONFIGS,
}

EMBEDDINGS_MODELS_WITH_CONFIGS = {
    "text-embedding-3-large": [
        {"model_class": UiPathEmbeddings},
    ],
    "gemini-embedding-001": [
        {"model_class": UiPathEmbeddings},
    ],
}


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
