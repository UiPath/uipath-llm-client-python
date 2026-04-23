"""Bedrock-specific model configurations for integration tests."""

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)

from uipath.llm_client.settings import UiPathBaseSettings

CLAUDE_BEDROCK_CONFIGS = [
    {
        "model_class": UiPathChatAnthropicBedrock,
    },
    {
        "model_class": UiPathChatAnthropicBedrock,
        "model_kwargs": {
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
]

CLAUDE_ANTHROPIC_BEDROCK_CONFIGS_NO_THINKING = [
    c
    for c in CLAUDE_BEDROCK_CONFIGS
    if c["model_class"] is UiPathChatAnthropicBedrock
    and "thinking" not in c.get("model_kwargs", {})
]

COMPLETIONS_MODELS_WITH_CONFIGS = {
    "anthropic.claude-haiku-4-5-20251001-v1:0": CLAUDE_BEDROCK_CONFIGS,
    # claude-opus-4-7 via Bedrock: tested with UiPathChatAnthropicBedrock only (no thinking;
    # thinking cassettes not yet recorded for this model).
    "anthropic.claude-opus-4-7": CLAUDE_ANTHROPIC_BEDROCK_CONFIGS_NO_THINKING,
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
