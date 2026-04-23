"""Helpers for the ``disabled_params`` convention.

``disabled_params`` is the langchain-openai-style declaration that certain
parameters must not be sent to a model. It maps param names to either:

- ``None``: the parameter is always disabled, regardless of its value.
- ``list[Any]``: the parameter is disabled only when its value is in the list.

We reuse this shape so that classes inheriting from
``langchain_openai.BaseChatOpenAI`` also benefit from its native
``_filter_disabled_params`` path inside ``bind_tools``.

The sampling-specific knowledge lives in ``disabled_params_from_model_details``:
when the gateway's discovery endpoint advertises
``modelDetails.shouldSkipTemperature: true`` on a reasoning-style model (e.g.
``anthropic.claude-opus-4-7``), the entire sampling set gets disabled.
"""

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


def disabled_params_from_model_details(
    model_details: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Derive ``disabled_params`` from a discovery-endpoint ``modelDetails`` dict.

    Returns None when no capability flags warrant disabling anything, so callers
    can distinguish "nothing to disable" from "disabled empty mapping".
    """
    if not model_details:
        return None
    disabled: dict[str, Any] = {}
    if model_details.get("shouldSkipTemperature"):
        for param in DISABLED_SAMPLING_PARAMS:
            disabled[param] = None
    # Future gateway flags (e.g. per-param ``shouldSkipTopP``) can extend this.
    return disabled or None


def is_disabled_value(value: Any, disabled_spec: Any) -> bool:
    """Match the langchain-openai ``_filter_disabled_params`` semantics.

    ``disabled_spec`` is either None (always disabled) or an iterable of values
    (disabled only when ``value`` is in the iterable).
    """
    if disabled_spec is None:
        return True
    try:
        return value in disabled_spec
    except TypeError:
        return False


def strip_disabled_kwargs(
    kwargs: Mapping[str, Any],
    *,
    disabled_params: Mapping[str, Any] | None,
    model_name: str,
    logger: Logger | None,
) -> dict[str, Any]:
    """Return a copy of ``kwargs`` with entries matching ``disabled_params`` removed.

    Uses the same matching rule as langchain-openai: a key is stripped when it
    is in ``disabled_params`` AND either the spec is None or the kwarg value
    matches one of the listed disabled values. Logs a warning per strip if a
    logger is supplied; silent otherwise.
    """
    out = dict(kwargs)
    if not disabled_params:
        return out
    for key in list(out.keys()):
        if key in disabled_params and is_disabled_value(out[key], disabled_params[key]):
            if logger is not None:
                logger.warning(
                    "Stripping disabled invocation param %r for model %r",
                    key,
                    model_name,
                )
            out.pop(key, None)
    return out
