"""Shared helpers for stripping sampling parameters that a model rejects.

Reasoning-style models (e.g. ``anthropic.claude-opus-4-7``) advertise
``modelDetails.shouldSkipTemperature: true`` on the discovery endpoint. When
that flag is set, the gateway rejects the entire sampling set, not just
``temperature``. The helpers here centralize that knowledge so every framework
wrapper (LangChain chat models, future LlamaIndex wrappers, the core
normalized client, etc.) can reuse the same rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from logging import Logger
from typing import Any

# Parameters the gateway rejects when ``shouldSkipTemperature`` is true.
# ``n`` (candidate count) is intentionally NOT here — it is not a sampling knob.
DISABLED_SAMPLING_PARAMS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
)


def should_skip_sampling(model_details: Mapping[str, Any] | None) -> bool:
    """True iff the provided ``modelDetails`` carries ``shouldSkipTemperature``."""
    return bool(model_details and model_details.get("shouldSkipTemperature"))


def strip_disabled_sampling_kwargs(
    kwargs: Mapping[str, Any],
    *,
    model_details: Mapping[str, Any] | None,
    model_name: str,
    logger: Logger | None,
) -> dict[str, Any]:
    """Return a copy of ``kwargs`` with disabled sampling params removed.

    When ``model_details`` does not flag the model as sampling-less, the
    input is returned unchanged (as a new dict so callers can mutate safely).
    A warning is logged per stripped parameter when a logger is provided.
    """
    out = dict(kwargs)
    if not should_skip_sampling(model_details):
        return out
    for param in DISABLED_SAMPLING_PARAMS:
        if param in out:
            if logger is not None:
                logger.warning(
                    "Stripping unsupported invocation param %r for model %r "
                    "(shouldSkipTemperature=True)",
                    param,
                    model_name,
                )
            out.pop(param, None)
    return out
