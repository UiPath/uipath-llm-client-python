"""Tests for ``UIPATH_SKIP_CLIENT_LLM_SPANS_KEY``: no client-side span for LLM Gateway calls.

The LLM Gateway records each call ("Model run") in the caller's trace, so in a
run whose metadata sets the key, a ``UiPathBaseChatModel`` starts its LLM run
under OpenTelemetry's ``suppress_instrumentation()`` and OpenInference doesn't
create the duplicate client-side span. Runs against the real OpenInference
LangChain instrumentation.
"""

import ast
import inspect
import os
from collections.abc import AsyncGenerator, Generator, Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from langchain_core.language_models import chat_models
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from uipath_langchain_client import UIPATH_SKIP_CLIENT_LLM_SPANS_KEY
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.settings import ApiType, RoutingMode, UiPathAPIConfig

from tests.langchain.conftest import LLMGW_ENV
from uipath.llm_client.settings import LLMGatewaySettings
from uipath.llm_client.settings.utils import SingletonMeta

SKIP: RunnableConfig = {"metadata": {UIPATH_SKIP_CLIENT_LLM_SPANS_KEY: True}}


class _GatewayChat(UiPathBaseChatModel, FakeListChatModel):
    """A Gateway model with canned replies: the real UiPathBaseChatModel paths."""

    api_config: UiPathAPIConfig = UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.NORMALIZED,
        freeze_base_url=True,
    )


class _DirectChat(FakeListChatModel):
    """A direct-provider model: not a UiPathBaseChatModel."""


class _Recorder(SpanProcessor):
    def __init__(self) -> None:
        self.started: list[str] = []

    def on_start(self, span: Span, parent_context: Any = None) -> None:
        self.started.append(span.name)

    def on_end(self, span: ReadableSpan) -> None:
        pass


@pytest.fixture(autouse=True)
def clear_singletons() -> Iterator[None]:
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


@pytest.fixture
def provider() -> Iterator[tuple[TracerProvider, _Recorder]]:
    instrumentor = LangChainInstrumentor()
    instrumentor.uninstrument()
    provider = TracerProvider()
    recorder = _Recorder()
    provider.add_span_processor(recorder)
    instrumentor.instrument(tracer_provider=provider)
    yield provider, recorder
    instrumentor.uninstrument()


@pytest.fixture
def recorder(provider: tuple[TracerProvider, _Recorder]) -> _Recorder:
    return provider[1]


def _gateway_chat() -> _GatewayChat:
    with patch.dict(os.environ, LLMGW_ENV, clear=True):
        settings = LLMGatewaySettings()
    return _GatewayChat(model="gpt-4o", settings=settings, responses=["a b"])


async def _call(llm: Any, how: str, config: RunnableConfig | None = None) -> None:
    if how == "invoke":
        llm.invoke("hi", config)
    elif how == "ainvoke":
        await llm.ainvoke("hi", config)
    elif how == "batch":
        llm.batch(["hi"], config)
    elif how == "abatch":
        await llm.abatch(["hi"], config)
    elif how == "stream":
        list(llm.stream("hi", config))
    elif how == "astream":
        [chunk async for chunk in llm.astream("hi", config)]
    elif how == "astream_events":
        [event async for event in llm.astream_events("hi", config, version="v2")]
    else:
        raise AssertionError(how)


async def _call_inside_run(
    llm: Any, how: str, run_config: RunnableConfig, call_config: RunnableConfig | None = None
) -> None:
    """Call the model from inside a parent run, as a graph node does."""

    async def node(_: Any) -> None:
        await _call(llm, how, call_config)

    await RunnableLambda(node).ainvoke(None, run_config)


CALLS = ["invoke", "ainvoke", "batch", "abatch", "stream", "astream", "astream_events"]


@pytest.mark.parametrize("how", CALLS)
async def test_gateway_call_inside_a_skipping_run_has_no_client_span(
    recorder: _Recorder, how: str
) -> None:
    await _call_inside_run(_gateway_chat(), how, SKIP)

    # Never started, not just dropped at export: live tracking can't leave it
    # behind as "running".
    assert "_GatewayChat" not in recorder.started
    assert "node" in recorder.started  # the parent run is still traced


@pytest.mark.parametrize("how", CALLS)
async def test_gateway_call_with_the_key_on_its_own_config_has_no_client_span(
    recorder: _Recorder, how: str
) -> None:
    await _call(_gateway_chat(), how, SKIP)

    assert "_GatewayChat" not in recorder.started


@pytest.mark.parametrize("how", CALLS)
async def test_own_metadata_inside_a_skipping_run_still_skips(
    recorder: _Recorder, how: str
) -> None:
    # The call's metadata replaces the inherited dict in its config; the key
    # still reaches it through the parent run's callback manager.
    await _call_inside_run(_gateway_chat(), how, SKIP, {"metadata": {"mine": 1}})

    assert "_GatewayChat" not in recorder.started


@pytest.mark.parametrize("how", CALLS)
async def test_gateway_call_keeps_client_span_without_the_key(
    recorder: _Recorder, how: str
) -> None:
    await _call_inside_run(_gateway_chat(), how, {"metadata": {"other": True}})

    assert "_GatewayChat" in recorder.started


@pytest.mark.parametrize("how", CALLS)
async def test_direct_provider_call_keeps_client_span(recorder: _Recorder, how: str) -> None:
    # Not a Gateway model: no "Model run" replaces its span.
    await _call_inside_run(_DirectChat(responses=["a b"]), how, SKIP)

    assert "_DirectChat" in recorder.started


@pytest.mark.parametrize("how", CALLS)
async def test_instrumentation_is_enabled_again_after_the_call(
    recorder: _Recorder, how: str
) -> None:
    await _call_inside_run(_gateway_chat(), how, SKIP)

    assert is_instrumentation_enabled()


async def test_skip_is_scoped_to_its_run(recorder: _Recorder) -> None:
    # No process-wide state: a run without the key, after one with it, keeps
    # its span.
    await _call_inside_run(_gateway_chat(), "ainvoke", SKIP)
    await _call_inside_run(_gateway_chat(), "ainvoke", {})

    assert recorder.started.count("_GatewayChat") == 1


def test_caller_code_between_chunks_is_traced(
    provider: tuple[TracerProvider, _Recorder],
) -> None:
    tracer_provider, recorder = provider
    tracer = tracer_provider.get_tracer(__name__)

    for i, _ in enumerate(_gateway_chat().stream("hi", SKIP)):
        assert is_instrumentation_enabled()
        with tracer.start_as_current_span(f"between-{i}"):
            pass

    assert "_GatewayChat" not in recorder.started
    assert [n for n in recorder.started if n.startswith("between-")] == [
        "between-0",
        "between-1",
        "between-2",
    ]


async def test_caller_code_between_async_chunks_is_traced(
    provider: tuple[TracerProvider, _Recorder],
) -> None:
    tracer_provider, recorder = provider
    tracer = tracer_provider.get_tracer(__name__)

    i = 0
    async for _ in _gateway_chat().astream("hi", SKIP):
        assert is_instrumentation_enabled()
        with tracer.start_as_current_span(f"between-{i}"):
            pass
        i += 1

    assert "_GatewayChat" not in recorder.started
    assert "between-0" in recorder.started


def test_abandoned_stream_leaves_instrumentation_enabled(recorder: _Recorder) -> None:
    # stream() is a generator, typed as the Iterator its base class returns.
    stream = cast(Generator[Any, None, None], _gateway_chat().stream("hi", SKIP))
    next(stream)
    stream.close()

    assert is_instrumentation_enabled()
    assert "_GatewayChat" not in recorder.started


async def test_abandoned_async_stream_leaves_instrumentation_enabled(
    recorder: _Recorder,
) -> None:
    stream = cast(AsyncGenerator[Any, None], _gateway_chat().astream("hi", SKIP))
    await anext(stream)
    await stream.aclose()

    assert is_instrumentation_enabled()
    assert "_GatewayChat" not in recorder.started


def test_every_method_that_starts_a_chat_model_run_is_covered() -> None:
    # Guards against a langchain-core upgrade adding another public method
    # that starts a run (calls on_chat_model_start) without going through the
    # ones UiPathBaseChatModel overrides. The v3 streaming helpers are the
    # documented exception (beta; they start the run lazily).
    (cls,) = ast.parse(inspect.getsource(chat_models.BaseChatModel)).body
    assert isinstance(cls, ast.ClassDef)
    starts_a_run = {
        node.name
        for node in cls.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and "on_chat_model_start(" in ast.unparse(node)
    }

    assert starts_a_run == {
        "generate",
        "agenerate",
        "stream",
        "astream",
        "_chat_model_stream_v3",
        "_achat_model_stream_v3",
    }
    for name in ("generate", "agenerate", "stream", "astream"):
        assert name in vars(UiPathBaseChatModel), name
