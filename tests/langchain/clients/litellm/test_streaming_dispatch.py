"""Regression tests for streaming dispatch under langchain-core >= 1.4.

ChatLiteLLM's before-validator materializes a value for every field, marking
``streaming`` as explicitly set. langchain-core >= 1.4 treats an explicitly-set
``streaming=False`` as a hard opt-out, so ``.stream()``/``.astream()`` silently
fall back to non-streaming ``invoke`` calls. ``UiPathChatLiteLLM`` un-marks the
field unless the caller actually passed it.
"""

from typing import Any

import pytest
from uipath_langchain_client.clients.litellm.chat_models import UiPathChatLiteLLM

from uipath.llm_client.settings import UiPathBaseSettings


@pytest.fixture
def chat_factory(client_settings: UiPathBaseSettings, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        type(client_settings),
        "get_model_info",
        lambda self, *args, **kwargs: {"vendor": "openai"},
    )

    def _build(**kwargs: Any) -> UiPathChatLiteLLM:
        return UiPathChatLiteLLM(
            model="gpt-5.2-2025-12-11", settings=client_settings, model_details={}, **kwargs
        )

    return _build


def test_stream_not_disabled_by_default(chat_factory: Any) -> None:
    chat = chat_factory()
    assert "streaming" not in chat.model_fields_set
    assert chat._should_stream(async_api=False, stream=True)
    assert chat._should_stream(async_api=True, stream=True)


def test_explicit_streaming_false_still_respected(chat_factory: Any) -> None:
    chat = chat_factory(streaming=False)
    assert "streaming" in chat.model_fields_set
    assert not chat._should_stream(async_api=False, stream=True)


def test_explicit_streaming_true_kept(chat_factory: Any) -> None:
    chat = chat_factory(streaming=True)
    assert "streaming" in chat.model_fields_set
    assert chat._should_stream(async_api=False)
