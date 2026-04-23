"""Unit tests for invocation-time stripping of sampling params based on
``modelDetails.shouldSkipTemperature``.

These tests monkeypatch ``client_settings.get_model_info`` and the instance's
``_uipath_generate`` / ``_uipath_agenerate`` to capture kwargs, so no HTTP is
made by this file.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.factory import get_chat_model
from uipath_langchain_client.utils import DISABLED_SAMPLING_PARAMS

from uipath.llm_client.settings import UiPathBaseSettings

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _stub_model_info(
    monkeypatch: pytest.MonkeyPatch,
    settings: UiPathBaseSettings,
    *,
    model_details: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    raises: BaseException | None = None,
) -> None:
    """Replace ``client_settings.get_model_info`` with a stub that returns
    (or raises) a controlled value. ``monkeypatch`` reverts this at teardown.
    """

    def _stub(model_name: str, **kwargs: Any) -> dict[str, Any]:
        if raises is not None:
            raise raises
        info: dict[str, Any] = {
            "modelName": model_name,
            "vendor": "AwsBedrock",
            "modelSubscriptionType": "UiPathOwned",
            "modelDetails": model_details,
        }
        if extra:
            info.update(extra)
        return info

    monkeypatch.setattr(settings, "get_model_info", _stub)


def _stub_generate(
    monkeypatch: pytest.MonkeyPatch, instance: UiPathChat, captured: dict[str, Any]
) -> None:
    """Replace ``_uipath_generate`` on the instance with a stub that records the
    kwargs it receives and returns a minimal ChatResult.
    """

    def _stub(
        messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        captured.update(kwargs)
        captured["__stop__"] = stop
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr(instance, "_uipath_generate", _stub)


def _stub_agenerate(
    monkeypatch: pytest.MonkeyPatch, instance: UiPathChat, captured: dict[str, Any]
) -> None:
    async def _stub(
        messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        captured.update(kwargs)
        captured["__stop__"] = stop
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr(instance, "_uipath_agenerate", _stub)


# --------------------------------------------------------------------------- #
# invocation-time stripping — sync
# --------------------------------------------------------------------------- #


def test_invoke_strips_sampling_kwargs_when_flag_set(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("hi", temperature=0.3, top_p=0.9, top_k=5, seed=42, max_tokens=100)

    # All sampling kwargs stripped; non-sampling kwargs preserved.
    for p in ("temperature", "top_p", "top_k", "seed"):
        assert p not in captured, f"{p} should have been stripped"
    assert captured.get("max_tokens") == 100


def test_invoke_strips_all_listed_sampling_params(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    # Pass every sampling param plus an unrelated kwarg; assert every
    # sampling param is stripped and only max_tokens survives.
    kwargs: dict[str, Any] = {p: 0.1 for p in DISABLED_SAMPLING_PARAMS}
    kwargs["max_tokens"] = 50
    llm.invoke("x", **kwargs)  # type: ignore[arg-type]

    for p in DISABLED_SAMPLING_PARAMS:
        assert p not in captured
    assert captured["max_tokens"] == 50


def test_n_is_not_stripped(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    # `n` (candidate count) is intentionally NOT part of _SAMPLING_PARAMS.
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("x", n=3)
    assert captured.get("n") == 3


# --------------------------------------------------------------------------- #
# invocation-time stripping — async
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ainvoke_strips_sampling_kwargs_when_flag_set(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
    )
    captured: dict[str, Any] = {}
    _stub_agenerate(monkeypatch, llm, captured)

    await llm.ainvoke("hi", temperature=0.3, top_p=0.9)

    assert "temperature" not in captured
    assert "top_p" not in captured


# --------------------------------------------------------------------------- #
# no-flag -> pass-through
# --------------------------------------------------------------------------- #


def test_invoke_preserves_kwargs_when_flag_absent(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="some-chatty-model",
        settings=client_settings,
        model_details={},  # empty — no flag
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("hi", temperature=0.3, top_p=0.9)

    assert captured["temperature"] == 0.3
    assert captured["top_p"] == 0.9


def test_invoke_preserves_kwargs_when_flag_false(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="some-chatty-model",
        settings=client_settings,
        model_details={"shouldSkipTemperature": False},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("hi", temperature=0.3)

    assert captured["temperature"] == 0.3


# --------------------------------------------------------------------------- #
# warning gating via self.logger
# --------------------------------------------------------------------------- #


def test_warning_logged_when_logger_set(
    monkeypatch: pytest.MonkeyPatch,
    client_settings: UiPathBaseSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("uipath.test.skip-sampling")
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
        logger=logger,
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        llm.invoke("x", temperature=0.3)

    assert any(
        "temperature" in rec.getMessage() and "shouldSkipTemperature" in rec.getMessage()
        for rec in caplog.records
    ), "expected a warning mentioning 'temperature' and 'shouldSkipTemperature'"


def test_no_warning_when_logger_is_none(
    monkeypatch: pytest.MonkeyPatch,
    client_settings: UiPathBaseSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
        logger=None,
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    with caplog.at_level(logging.DEBUG):
        llm.invoke("x", temperature=0.3)

    # temperature was still stripped — we just don't log when logger is None.
    assert "temperature" not in captured
    assert not any("shouldSkipTemperature" in rec.getMessage() for rec in caplog.records)


def test_no_warning_when_nothing_to_strip(
    monkeypatch: pytest.MonkeyPatch,
    client_settings: UiPathBaseSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("uipath.test.skip-sampling-quiet")
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
        logger=logger,
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        llm.invoke("x", max_tokens=50)  # no sampling kwargs at all

    assert not any("shouldSkipTemperature" in rec.getMessage() for rec in caplog.records)


# --------------------------------------------------------------------------- #
# eager model_details resolution via the UiPathBaseLLMClient validator
# --------------------------------------------------------------------------- #


def test_validator_populates_model_details_on_direct_instantiation(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    _stub_model_info(monkeypatch, client_settings, model_details={"shouldSkipTemperature": True})
    # No model_details passed — the validator should fetch and populate it.
    llm = UiPathChat(model="anthropic.claude-opus-4-7", settings=client_settings)
    assert llm.model_details == {"shouldSkipTemperature": True}

    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)
    llm.invoke("x", temperature=0.5)
    assert "temperature" not in captured


def test_validator_swallows_discovery_errors(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    _stub_model_info(monkeypatch, client_settings, raises=RuntimeError("boom"))
    llm = UiPathChat(model="anthropic.claude-opus-4-7", settings=client_settings)

    # Discovery failure => we fall back to empty model_details and don't strip.
    assert llm.model_details == {}

    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)
    llm.invoke("x", temperature=0.5)
    assert captured["temperature"] == 0.5


def test_validator_does_not_overwrite_explicitly_provided_model_details(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    # If a caller (or the factory) already passed model_details, post_init
    # must not call get_model_info and must not overwrite the forwarded value.
    called: dict[str, bool] = {"called": False}

    def _stub(*args: Any, **kwargs: Any) -> dict[str, Any]:
        called["called"] = True
        return {"modelDetails": {"shouldSkipTemperature": False}}

    monkeypatch.setattr(client_settings, "get_model_info", _stub)

    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
    )
    assert llm.model_details == {"shouldSkipTemperature": True}
    assert called["called"] is False


# --------------------------------------------------------------------------- #
# factory forwarding
# --------------------------------------------------------------------------- #


def test_factory_forwards_model_details_to_normalized_chat(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    from uipath_langchain_client.settings import RoutingMode

    _stub_model_info(monkeypatch, client_settings, model_details={"shouldSkipTemperature": True})

    llm = get_chat_model(
        "anthropic.claude-opus-4-7",
        client_settings=client_settings,
        routing_mode=RoutingMode.NORMALIZED,
    )

    assert isinstance(llm, UiPathChat)
    assert llm.model_details == {"shouldSkipTemperature": True}


def test_factory_forwards_empty_dict_when_no_model_details(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    from uipath_langchain_client.settings import RoutingMode

    _stub_model_info(monkeypatch, client_settings, model_details=None)

    llm = get_chat_model(
        "gpt-4o",
        client_settings=client_settings,
        routing_mode=RoutingMode.NORMALIZED,
    )

    assert isinstance(llm, UiPathChat)
    # None -> {} via `or {}` in the factory
    assert llm.model_details == {}
