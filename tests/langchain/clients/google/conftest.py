"""Google-specific model configurations for integration tests."""

import json
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.google.embeddings import UiPathGoogleGenerativeAIEmbeddings

from uipath.llm_client.settings import UiPathBaseSettings

GEMINI_2_5_CONFIGS = [
    {
        "model_class": UiPathChatGoogleGenerativeAI,
        "model_kwargs": {"thinking_budget": 128, "include_thoughts": False},
    },
    {
        "model_class": UiPathChatGoogleGenerativeAI,
        "model_kwargs": {"thinking_budget": 128, "include_thoughts": True},
    },
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
]

COMPLETIONS_MODELS_WITH_CONFIGS = {
    "gemini-2.5-flash": GEMINI_2_5_CONFIGS,
    "gemini-3-flash-preview": GEMINI_3_CONFIGS,
}

EMBEDDINGS_MODELS_WITH_CONFIGS = {
    "gemini-embedding-001": [
        {"model_class": UiPathGoogleGenerativeAIEmbeddings},
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
