"""Anthropic-specific model configurations for integration tests.

Tests UiPathChatAnthropic with both vertexai and awsbedrock vendor_types.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic

from uipath.llm_client.settings import UiPathBaseSettings

CLAUDE_VERTEXAI_CONFIGS = [
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
]

CLAUDE_BEDROCK_CONFIGS = [
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
]

COMPLETIONS_MODELS_WITH_CONFIGS = {
    "claude-haiku-4-5@20251001": CLAUDE_VERTEXAI_CONFIGS,
    "anthropic.claude-haiku-4-5-20251001-v1:0": CLAUDE_BEDROCK_CONFIGS,
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
