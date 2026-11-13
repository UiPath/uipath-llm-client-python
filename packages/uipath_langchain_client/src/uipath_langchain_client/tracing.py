"""Skip client-side LLM spans for LLM Gateway calls; the Gateway already traces them."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.runnables.config import ensure_config

UIPATH_SKIP_CLIENT_LLM_SPANS_KEY = "uipath_skip_client_llm_spans"


def skips_client_llm_span(metadata: Mapping[str, Any] | None, callbacks: Callbacks) -> bool:
    """Check the key in the call's metadata and in the metadata inherited from the parent run."""
    if metadata and metadata.get(UIPATH_SKIP_CLIENT_LLM_SPANS_KEY):
        return True
    return isinstance(callbacks, BaseCallbackManager) and bool(
        callbacks.inheritable_metadata.get(UIPATH_SKIP_CLIENT_LLM_SPANS_KEY)
    )


def skips_client_llm_span_in_context() -> bool:
    """Check the key in the current context's run config."""
    config = ensure_config()
    return skips_client_llm_span(config.get("metadata"), config.get("callbacks"))


@contextmanager
def llm_run_start(skip: bool) -> Generator[None]:
    """Suppress OpenTelemetry instrumentation if ``skip``."""
    if not skip:
        yield
        return
    try:
        from opentelemetry.instrumentation.utils import suppress_instrumentation
    except ImportError:
        yield
        return
    with suppress_instrumentation():
        yield
