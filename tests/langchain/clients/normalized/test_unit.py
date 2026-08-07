"""LangChain unit tests for Normalized provider clients."""

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_tests.unit_tests import ChatModelUnitTests, EmbeddingsUnitTests
from pydantic import BaseModel
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.normalized.embeddings import UiPathEmbeddings

from uipath.llm_client.settings import UiPathBaseSettings

NORMALIZED_CHAT_CLASSES = [UiPathChat]
NORMALIZED_EMBEDDINGS_CLASSES = [UiPathEmbeddings]


class StructuredAnswer(BaseModel):
    answer: str


@pytest.mark.parametrize(
    ("model_name", "expected_method"),
    [
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "json_mode"),
        ("claude-haiku-4-5@20251001", "json_mode"),
        ("gemini-2.5-flash", "function_calling"),
        ("gpt-4o-2024-11-20", "function_calling"),
    ],
)
def test_auto_structured_output_selects_provider_compatible_method(
    client_settings: UiPathBaseSettings,
    model_name: str,
    expected_method: str,
) -> None:
    model = UiPathChat(model=model_name, settings=client_settings)
    raw = AIMessage(content='{"answer":"ok"}')

    with (
        patch.object(
            UiPathChat,
            "bind",
            autospec=True,
            return_value=RunnableLambda(lambda _: raw),
        ) as bind,
        patch.object(
            UiPathChat,
            "bind_tools",
            autospec=True,
            return_value=RunnableLambda(lambda _: raw),
        ) as bind_tools,
    ):
        model.with_structured_output(StructuredAnswer, method="auto")

    if expected_method == "json_mode":
        bind.assert_called_once()
        bind_tools.assert_not_called()
        assert bind.call_args.kwargs["response_format"] == {"type": "json_object"}
    else:
        bind.assert_not_called()
        bind_tools.assert_called_once()
        assert bind_tools.call_args.kwargs["parallel_tool_calls"] is False


def test_explicit_function_calling_keeps_existing_parallel_default(
    client_settings: UiPathBaseSettings,
) -> None:
    model = UiPathChat(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        settings=client_settings,
    )

    with patch.object(
        UiPathChat,
        "bind_tools",
        autospec=True,
        return_value=RunnableLambda(lambda _: AIMessage(content="")),
    ) as bind_tools:
        model.with_structured_output(StructuredAnswer, method="function_calling")

    assert "parallel_tool_calls" not in bind_tools.call_args.kwargs


@pytest.mark.parametrize(
    ("content", "expected_answer", "has_error"),
    [
        ('{"answer":"ok"}', "ok", False),
        ("not json", None, True),
    ],
)
def test_include_raw_accepts_message_list_input(
    client_settings: UiPathBaseSettings,
    content: str,
    expected_answer: str | None,
    has_error: bool,
) -> None:
    model = UiPathChat(model="gpt-4o-2024-11-20", settings=client_settings)
    raw = AIMessage(content=content)

    with patch.object(
        UiPathChat,
        "bind",
        autospec=True,
        return_value=RunnableLambda(lambda _: raw),
    ):
        runnable = model.with_structured_output(
            StructuredAnswer,
            method="json_mode",
            include_raw=True,
        )
        result = runnable.invoke([HumanMessage(content="answer the question")])

    assert isinstance(result, dict)
    assert result["raw"] is raw
    assert (result["parsing_error"] is not None) is has_error
    if expected_answer is None:
        assert result["parsed"] is None
    else:
        assert result["parsed"] == StructuredAnswer(answer=expected_answer)


class TestNormalizedChatModel(ChatModelUnitTests):
    @pytest.fixture(autouse=True, params=NORMALIZED_CHAT_CLASSES)
    def setup_models(self, request: pytest.FixtureRequest, client_settings: UiPathBaseSettings):
        self._completions_class = request.param
        self._completions_kwargs = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "client_settings": client_settings,
            "vendor_type": "awsbedrock",
        }

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self._completions_kwargs

    @pytest.mark.xfail(reason="Skipping serdes test for now")
    def test_serdes(self, *args: Any, **kwargs: Any) -> None: ...


class TestNormalizedEmbeddings(EmbeddingsUnitTests):
    @pytest.fixture(autouse=True, params=NORMALIZED_EMBEDDINGS_CLASSES)
    def setup_models(self, request: pytest.FixtureRequest, client_settings: UiPathBaseSettings):
        self._embeddings_class = request.param
        self._embeddings_kwargs = {
            "model": "PLACEHOLDER",
            "client_settings": client_settings,
        }

    @property
    def embeddings_class(self) -> type[Embeddings]:
        return self._embeddings_class

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return self._embeddings_kwargs
