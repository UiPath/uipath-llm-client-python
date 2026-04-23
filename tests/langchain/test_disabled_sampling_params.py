"""Unit tests for the ``disabled_params`` + ``model_details`` wiring.

Covers two layers:

1. **Init-time clearing** via the ``@model_validator(mode="before")`` on
   ``UiPathBaseLLMClient``: fields that match ``disabled_params`` (either
   provided or derived from ``model_details.shouldSkipTemperature``) are
   popped before pydantic field validation runs, so they never enter
   ``model_fields_set``.

2. **Invocation-time stripping** via ``strip_disabled_kwargs`` wired into
   ``_generate``/``_agenerate``/``_stream``/``_astream`` on
   ``UiPathBaseChatModel``.

Tests monkeypatch ``client_settings.get_model_info`` and stub the
``_uipath_generate``/``_uipath_agenerate`` methods so no HTTP is ever made.
"""

import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.factory import get_chat_model

from uipath.llm_client.settings import UiPathBaseSettings
from uipath.llm_client.utils.sampling import DISABLED_SAMPLING_PARAMS

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
    """Replace ``client_settings.get_model_info`` with a stub."""

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
# init-time clearing (the new layer)
# --------------------------------------------------------------------------- #


def test_init_clears_sampling_fields_when_model_details_flag_set(
    client_settings: UiPathBaseSettings,
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.1,
    )
    # The popped fields must not be in model_fields_set, and _default_params
    # (which filters by that set) must not emit them.
    for param in ("temperature", "top_p", "frequency_penalty"):
        assert param not in llm.model_fields_set
    assert "temperature" not in llm._default_params
    assert "top_p" not in llm._default_params
    # disabled_params is derived from modelDetails when not explicitly passed.
    assert llm.disabled_params is not None
    assert set(llm.disabled_params) >= set(DISABLED_SAMPLING_PARAMS)


def test_user_provided_disabled_params_wins_over_model_details(
    client_settings: UiPathBaseSettings,
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
        disabled_params={"logit_bias": None},  # overrides the derivation
        temperature=0.5,  # should survive because logit_bias is the only disabled key
    )
    assert llm.disabled_params == {"logit_bias": None}
    # temperature is NOT disabled here, so the user's value survives.
    assert llm.temperature == 0.5


def test_init_clears_from_model_kwargs_too(
    client_settings: UiPathBaseSettings,
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
        model_kwargs={"temperature": 0.5, "max_tokens": 100},
    )
    # temperature popped from model_kwargs; max_tokens preserved.
    assert "temperature" not in llm.model_kwargs
    assert llm.model_kwargs == {"max_tokens": 100}


def test_init_no_op_when_flag_absent(client_settings: UiPathBaseSettings) -> None:
    llm = UiPathChat(
        model="some-chatty-model",
        settings=client_settings,
        model_details={},
        temperature=0.5,
    )
    assert llm.temperature == 0.5
    assert "temperature" in llm.model_fields_set


# --------------------------------------------------------------------------- #
# invocation-time stripping
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

    for p in ("temperature", "top_p", "top_k", "seed"):
        assert p not in captured, f"{p} should have been stripped"
    assert captured.get("max_tokens") == 100


def test_invoke_strips_every_listed_sampling_param(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    kwargs: dict[str, Any] = {p: 0.1 for p in DISABLED_SAMPLING_PARAMS}
    kwargs["max_tokens"] = 50
    llm.invoke("x", **kwargs)  # type: ignore[arg-type]

    for p in DISABLED_SAMPLING_PARAMS:
        assert p not in captured
    assert captured["max_tokens"] == 50


def test_n_is_not_stripped(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    # `n` (candidate count) is intentionally NOT part of DISABLED_SAMPLING_PARAMS.
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("x", n=3)
    assert captured.get("n") == 3


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


def test_invoke_preserves_kwargs_when_flag_absent(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="some-chatty-model",
        settings=client_settings,
        model_details={},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("hi", temperature=0.3, top_p=0.9)

    assert captured["temperature"] == 0.3
    assert captured["top_p"] == 0.9


def test_invoke_honors_user_supplied_disabled_params(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    llm = UiPathChat(
        model="some-chatty-model",
        settings=client_settings,
        model_details={},
        disabled_params={"frequency_penalty": None},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("x", temperature=0.3, frequency_penalty=1.0)

    assert captured.get("temperature") == 0.3
    assert "frequency_penalty" not in captured


def test_invoke_honors_disabled_params_value_list(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    # langchain-openai semantics: list spec means "disabled when value is in list".
    llm = UiPathChat(
        model="some-chatty-model",
        settings=client_settings,
        model_details={},
        disabled_params={"temperature": [0.0, 1.5]},
    )
    captured: dict[str, Any] = {}
    _stub_generate(monkeypatch, llm, captured)

    llm.invoke("x", temperature=1.5)  # matches -> stripped
    assert "temperature" not in captured

    captured.clear()
    llm.invoke("x", temperature=0.7)  # does not match -> preserved
    assert captured.get("temperature") == 0.7


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
        "temperature" in rec.getMessage() and "disabled" in rec.getMessage()
        for rec in caplog.records
    ), "expected a warning mentioning 'temperature' and 'disabled'"


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

    assert "temperature" not in captured
    assert not any("disabled invocation param" in rec.getMessage() for rec in caplog.records)


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
        llm.invoke("x", max_tokens=50)

    assert not any("disabled invocation param" in rec.getMessage() for rec in caplog.records)


# --------------------------------------------------------------------------- #
# discovery fallback
# --------------------------------------------------------------------------- #


def test_validator_fetches_model_details_when_not_provided(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    _stub_model_info(monkeypatch, client_settings, model_details={"shouldSkipTemperature": True})
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        temperature=0.5,
    )
    # model_details resolved from discovery; disabled_params derived; temperature stripped.
    assert llm.model_details == {"shouldSkipTemperature": True}
    assert llm.disabled_params is not None
    assert "temperature" in llm.disabled_params
    assert "temperature" not in llm.model_fields_set


def test_validator_swallows_discovery_errors(
    monkeypatch: pytest.MonkeyPatch, client_settings: UiPathBaseSettings
) -> None:
    _stub_model_info(monkeypatch, client_settings, raises=RuntimeError("boom"))
    llm = UiPathChat(
        model="anthropic.claude-opus-4-7",
        settings=client_settings,
        temperature=0.5,
    )
    # Discovery failure => model_details is {} and nothing is stripped.
    assert llm.model_details == {}
    assert llm.disabled_params is None
    assert llm.temperature == 0.5


# --------------------------------------------------------------------------- #
# factory forwarding
# --------------------------------------------------------------------------- #


def test_factory_forwards_model_details(
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
    assert llm.disabled_params is not None
    assert "temperature" in llm.disabled_params


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
    assert llm.model_details == {}
    assert llm.disabled_params is None
