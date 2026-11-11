"""Leave the LLM span of LLM Gateway calls to the Gateway.

The LLM Gateway records each model call ("Model run", with its governance and
guardrail spans) in the caller's trace, under the span named by the
``x-uipath-traceparent-id`` header. A client-side span for the same call, such
as the one OpenInference's LangChain instrumentation creates, would duplicate
it.

A run asks for the skip through its config metadata::

    graph.invoke(inputs, config={"metadata": {UIPATH_SKIP_CLIENT_LLM_SPANS_KEY: True}})

LangChain hands a run's metadata down to every child run, so each
``UiPathBaseChatModel`` called inside it starts its LLM run under OpenTelemetry's
public ``suppress_instrumentation()``, which instrumentations (OpenInference
included) check before creating a span. Runs that don't set the key, such as
low-code agents that keep their own client-side LLM spans, aren't affected.
Direct-provider models aren't ``UiPathBaseChatModel`` and keep their spans
either way, since nothing else traces them.
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.runnables.config import ensure_config

UIPATH_SKIP_CLIENT_LLM_SPANS_KEY = "uipath_skip_client_llm_spans"


def skips_client_llm_span(metadata: Mapping[str, Any] | None, callbacks: Callbacks) -> bool:
    """Whether a model run started with this metadata and these callbacks skips its span.

    The key is looked up in the call's own metadata and in the metadata the
    parent run passes down through its callback manager, so a caller that
    passes its own ``metadata=`` inside a run that set the key still skips.
    """
    if metadata and metadata.get(UIPATH_SKIP_CLIENT_LLM_SPANS_KEY):
        return True
    return isinstance(callbacks, BaseCallbackManager) and bool(
        callbacks.inheritable_metadata.get(UIPATH_SKIP_CLIENT_LLM_SPANS_KEY)
    )


def skips_client_llm_span_in_context() -> bool:
    """As ``skips_client_llm_span``, for the run config of the current context."""
    config = ensure_config()
    return skips_client_llm_span(config.get("metadata"), config.get("callbacks"))


@contextmanager
def llm_run_start(skip: bool) -> Generator[None]:
    """Suppress instrumentation while a Gateway call starts, if ``skip``.

    ``generate`` / ``agenerate`` run whole inside it. For a stream it wraps
    only the step that starts the run, so the caller's code between chunks is
    never suppressed.
    """
    if not skip:
        yield
        return
    try:
        from opentelemetry.instrumentation.utils import suppress_instrumentation
    except ImportError:
        # No OpenTelemetry instrumentation installed: nothing creates the span.
        yield
        return
    with suppress_instrumentation():
        yield
