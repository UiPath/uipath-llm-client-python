"""Tests for the LLM Gateway marker that UiPathBaseChatModel merges into ``metadata``.

Tracers (for example uipath-langchain's) read the marker from each run's metadata
to tell Gateway-backed calls, which the Gateway traces itself, apart from
direct-provider calls, which only the client can trace.
"""

from typing import Any
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from uipath_langchain_client import UIPATH_LLM_GATEWAY_METADATA_KEY
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.openai.chat_models import UiPathChatOpenAI

from tests.langchain.conftest import COMPLETION_CLIENTS_CLASSES
from tests.langchain.transports import route_transport
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta

CHAT_RESPONSE_JSON = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class RunMetadataRecorder(BaseCallbackHandler):
    """Records the metadata LangChain hands to each chat model run."""

    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.metadata.append(dict(metadata or {}))


@pytest.fixture(autouse=True)
def clear_singletons():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


def _routed_chat(settings: LLMGatewaySettings, **kwargs: Any) -> UiPathChat:
    chat = UiPathChat(model="gpt-4o", settings=settings, **kwargs)
    route_transport(
        chat,
        httpx.MockTransport(lambda _: httpx.Response(200, json=CHAT_RESPONSE_JSON)),
    )
    return chat


@pytest.mark.parametrize("cls", COMPLETION_CLIENTS_CLASSES, ids=lambda c: c.__name__)
def test_every_completion_client_carries_the_marker(
    cls: type[Any], llmgw_settings: LLMGatewaySettings
) -> None:
    # Built for real, so a vendor class whose own validator resets ``metadata``
    # would fail here. ``client_settings=`` rather than the ``settings`` alias:
    # ChatLiteLLM's before-validator drops the alias. get_model_info is patched
    # because some clients (LiteLLM) run discovery while validating.
    # ``type[Any]``: each concrete class defaults ``api_config``, which the base
    # class declares as required, so the base signature doesn't fit them.
    with patch.object(
        LLMGatewaySettings, "get_model_info", lambda self, *a, **k: {"vendor": "openai"}
    ):
        chat: UiPathBaseChatModel = cls(
            model="gpt-4o-2024-11-20",
            client_settings=llmgw_settings,
            model_details={},
            metadata={"team": "agents"},
        )

    assert chat.metadata is not None
    assert chat.metadata[UIPATH_LLM_GATEWAY_METADATA_KEY] is True
    assert chat.metadata["team"] == "agents"


def test_marker_is_set_without_user_metadata(llmgw_settings: LLMGatewaySettings) -> None:
    chat = UiPathChat(model="gpt-4o", settings=llmgw_settings)

    assert chat.metadata is not None
    assert chat.metadata[UIPATH_LLM_GATEWAY_METADATA_KEY] is True


def test_marker_is_set_on_passthrough_clients(llmgw_settings: LLMGatewaySettings) -> None:
    chat = UiPathChatOpenAI(model="gpt-4o-2024-11-20", settings=llmgw_settings, model_details={})

    assert chat.metadata is not None
    assert chat.metadata[UIPATH_LLM_GATEWAY_METADATA_KEY] is True


def test_user_metadata_is_kept_alongside_the_marker(llmgw_settings: LLMGatewaySettings) -> None:
    chat = UiPathChat(model="gpt-4o", settings=llmgw_settings, metadata={"team": "agents"})

    assert chat.metadata is not None
    assert chat.metadata["team"] == "agents"
    assert chat.metadata[UIPATH_LLM_GATEWAY_METADATA_KEY] is True


def test_user_cannot_switch_the_marker_off(llmgw_settings: LLMGatewaySettings) -> None:
    chat = UiPathChat(
        model="gpt-4o", settings=llmgw_settings, metadata={UIPATH_LLM_GATEWAY_METADATA_KEY: False}
    )

    assert chat.metadata is not None
    assert chat.metadata[UIPATH_LLM_GATEWAY_METADATA_KEY] is True


def test_marker_survives_model_settings_that_set_metadata(
    llmgw_settings: LLMGatewaySettings,
) -> None:
    chat = UiPathChat(
        model="gpt-4o",
        settings=llmgw_settings,
        model_settings={"metadata": {"source": "agent.json"}},
    )

    assert chat.metadata is not None
    assert chat.metadata["source"] == "agent.json"
    assert chat.metadata[UIPATH_LLM_GATEWAY_METADATA_KEY] is True


def test_marker_reaches_the_run_metadata(llmgw_settings: LLMGatewaySettings) -> None:
    recorder = RunMetadataRecorder()
    chat = _routed_chat(llmgw_settings, metadata={"team": "agents"})

    chat.invoke("hello", config={"callbacks": [recorder], "metadata": {"node": "model"}})

    assert len(recorder.metadata) == 1
    run_metadata = recorder.metadata[0]
    assert run_metadata[UIPATH_LLM_GATEWAY_METADATA_KEY] is True
    assert run_metadata["team"] == "agents"
    assert run_metadata["node"] == "model"


async def test_marker_reaches_the_run_metadata_async(llmgw_settings: LLMGatewaySettings) -> None:
    recorder = RunMetadataRecorder()
    chat = UiPathChat(model="gpt-4o", settings=llmgw_settings)
    route_transport(
        chat,
        httpx.MockTransport(lambda _: httpx.Response(200, json=CHAT_RESPONSE_JSON)),
        httpx.MockTransport(lambda _: httpx.Response(200, json=CHAT_RESPONSE_JSON)),
    )

    await chat.ainvoke("hello", config={"callbacks": [recorder]})

    assert recorder.metadata[0][UIPATH_LLM_GATEWAY_METADATA_KEY] is True


def test_direct_provider_runs_are_not_marked() -> None:
    recorder = RunMetadataRecorder()
    native = FakeListChatModel(responses=["Hi"])

    native.invoke("hello", config={"callbacks": [recorder]})

    assert UIPATH_LLM_GATEWAY_METADATA_KEY not in recorder.metadata[0]


def test_marker_does_not_leak_to_sibling_runs(llmgw_settings: LLMGatewaySettings) -> None:
    """A Gateway model's marker is local to its own runs, so in a mixed chain only
    the Gateway call is marked (inheritable metadata would mark both)."""
    recorder = RunMetadataRecorder()
    gateway = _routed_chat(llmgw_settings)
    native = FakeListChatModel(responses=["Hi"])

    chain = gateway | (lambda message: message.content) | native
    chain.invoke("hello", config={"callbacks": [recorder]})

    assert [m.get(UIPATH_LLM_GATEWAY_METADATA_KEY) for m in recorder.metadata] == [True, None]
