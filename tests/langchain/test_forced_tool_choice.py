"""Unit tests for relaxing a forced ``tool_choice`` on models flagged
``modelDetails.shouldSkipForcedToolChoice`` (e.g. Claude Opus 5.5, which 400s on
``tool_choice`` ``any`` / ``tool``)."""

import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.settings import VendorType
from uipath_langchain_client.tool_choice import relax_tool_choice

from uipath.llm_client.settings import UiPathBaseSettings

FLAG = {"shouldSkipForcedToolChoice": True}


@tool
def end_execution(result: str) -> str:
    """Finish the run with a result."""
    return result


def _capture_generate(
    monkeypatch: pytest.MonkeyPatch, instance: Any, captured: dict[str, Any]
) -> None:
    def _stub(messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        captured.update(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    async def _astub(messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        captured.update(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def _stream(messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        captured.update(kwargs)
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

    monkeypatch.setattr(instance, "_uipath_generate", _stub)
    monkeypatch.setattr(instance, "_uipath_agenerate", _astub)
    monkeypatch.setattr(instance, "_uipath_stream", _stream)


@pytest.mark.parametrize(
    ("forced", "relaxed"),
    [
        ("any", "auto"),
        ("required", "auto"),
        ("end_execution", "auto"),
        ({"type": "any"}, {"type": "auto"}),
        ({"type": "tool", "name": "end_execution"}, {"type": "auto"}),
        (
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "auto", "disable_parallel_tool_use": True},
        ),
        ({"type": "function", "function": {"name": "end_execution"}}, "auto"),
        ({"any": {}}, {"auto": {}}),
        ({"tool": {"name": "end_execution"}}, {"auto": {}}),
    ],
)
def test_relax_tool_choice_forced_shapes(forced: Any, relaxed: Any) -> None:
    assert relax_tool_choice(forced) == relaxed


@pytest.mark.parametrize(
    "unforced", ["auto", "none", {"type": "auto"}, {"type": "none"}, {"auto": {}}]
)
def test_relax_tool_choice_leaves_unforced_untouched(unforced: Any) -> None:
    assert relax_tool_choice(unforced) == unforced


def test_invoke_relaxes_forced_tool_choice_when_flag_set(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(model="claude-opus-5-5", settings=client_settings, model_details=FLAG)
    captured: dict[str, Any] = {}
    _capture_generate(monkeypatch, llm, captured)

    llm.invoke("hi", tool_choice="required")

    assert captured["tool_choice"] == "auto"


async def test_ainvoke_relaxes_forced_tool_choice_when_flag_set(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(model="claude-opus-5-5", settings=client_settings, model_details=FLAG)
    captured: dict[str, Any] = {}
    _capture_generate(monkeypatch, llm, captured)

    await llm.ainvoke("hi", tool_choice={"type": "any"})

    assert captured["tool_choice"] == {"type": "auto"}


def test_stream_relaxes_forced_tool_choice_when_flag_set(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(model="claude-opus-5-5", settings=client_settings, model_details=FLAG)
    captured: dict[str, Any] = {}
    _capture_generate(monkeypatch, llm, captured)

    list(llm.stream("hi", tool_choice={"type": "tool", "name": "end_execution"}))

    assert captured["tool_choice"] == {"type": "auto"}


def test_forced_tool_choice_kept_when_flag_absent(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(model="claude-opus-5", settings=client_settings, model_details={})
    captured: dict[str, Any] = {}
    _capture_generate(monkeypatch, llm, captured)

    llm.invoke("hi", tool_choice={"type": "any"})

    assert captured["tool_choice"] == {"type": "any"}


def test_anthropic_bind_tools_any_is_relaxed_end_to_end(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChatAnthropic(
        model="claude-opus-5-5",
        settings=client_settings,
        vendor_type=VendorType.VERTEXAI,
        model_details=FLAG,
    )
    captured: dict[str, Any] = {}
    _capture_generate(monkeypatch, llm, captured)

    llm.bind_tools([end_execution], tool_choice="any").invoke("hi")

    assert captured["tool_choice"] == {"type": "auto"}
    assert [t["name"] for t in captured["tools"]] == ["end_execution"]


def test_relaxation_logs_warning_when_logger_set(
    monkeypatch: pytest.MonkeyPatch,
    client_settings: UiPathBaseSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test_forced_tool_choice")
    llm = UiPathChat(
        model="claude-opus-5-5", settings=client_settings, model_details=FLAG, logger=logger
    )
    _capture_generate(monkeypatch, llm, {})

    with caplog.at_level(logging.WARNING, logger="test_forced_tool_choice"):
        llm.invoke("hi", tool_choice="required")

    assert any("Relaxing forced tool_choice" in r.getMessage() for r in caplog.records)
