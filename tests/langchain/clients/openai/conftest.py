"""OpenAI-specific model configurations for integration tests."""

import json
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
)
from uipath_langchain_client.clients.openai.embeddings import (
    UiPathAzureOpenAIEmbeddings,
)

from uipath.llm_client.settings import UiPathBaseSettings

GPT_MODELS_NON_REASONING_CONFIGS = [
    {"model_class": UiPathAzureChatOpenAI},
    {"model_class": UiPathAzureChatOpenAI, "model_kwargs": {"use_responses_api": True}},
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
]

COMPLETIONS_MODELS_WITH_CONFIGS = {
    "gpt-4o-2024-11-20": GPT_MODELS_NON_REASONING_CONFIGS,
    "gpt-5.2-2025-12-11": GPT_MODELS_WITH_REASONING_CONFIGS,
}

EMBEDDINGS_MODELS_WITH_CONFIGS = {
    "text-embedding-3-large": [
        {"model_class": UiPathAzureOpenAIEmbeddings},
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
