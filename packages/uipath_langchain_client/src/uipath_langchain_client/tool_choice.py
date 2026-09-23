"""Relax a forced ``tool_choice`` for models that only accept ``auto`` / ``none``.

Some models reject any forced tool choice with a 400 (e.g. Claude Opus 5.5:
``tool_choice: type "tool" and "any" are not supported for this model``). The
gateway advertises this as ``modelDetails.shouldSkipForcedToolChoice``; when it
is set, a forced choice is rewritten to the vendor's ``auto`` in the same shape
the vendor SDK produced, so callers that force tool use (agent loops) keep
working and simply tolerate a tool-less turn.
"""

from collections.abc import Mapping
from logging import Logger
from typing import Any

_UNFORCED_STRINGS = frozenset({"auto", "none"})


def rejects_forced_tool_choice(model_details: Mapping[str, Any] | None) -> bool:
    return bool(model_details and model_details.get("shouldSkipForcedToolChoice"))


def relax_tool_choice(tool_choice: Any) -> Any:
    """Return the ``auto`` equivalent of a forced ``tool_choice``, else the input unchanged.

    Shapes by vendor SDK:
    - strings (OpenAI / normalized / pre-bind): ``any`` / ``required`` / a tool name
    - Anthropic: ``{"type": "any" | "tool", ...}``
    - OpenAI: ``{"type": "function", "function": {...}}``
    - Bedrock Converse: ``{"any": {}}`` / ``{"tool": {"name": ...}}``
    """
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in _UNFORCED_STRINGS else "auto"
    if not isinstance(tool_choice, Mapping):
        return tool_choice
    choice_type = tool_choice.get("type")
    if choice_type in ("any", "tool"):
        relaxed: dict[str, Any] = {"type": "auto"}
        if "disable_parallel_tool_use" in tool_choice:
            relaxed["disable_parallel_tool_use"] = tool_choice["disable_parallel_tool_use"]
        return relaxed
    if choice_type == "function":
        return "auto"
    if choice_type is None and ("any" in tool_choice or "tool" in tool_choice):
        return {"auto": {}}
    return tool_choice


def relax_forced_tool_choice_kwargs(
    kwargs: dict[str, Any],
    *,
    model_details: Mapping[str, Any] | None,
    model_name: str,
    logger: Logger | None,
) -> dict[str, Any]:
    if not rejects_forced_tool_choice(model_details) or kwargs.get("tool_choice") is None:
        return kwargs
    original = kwargs["tool_choice"]
    relaxed = relax_tool_choice(original)
    if relaxed == original:
        return kwargs
    if logger is not None:
        logger.warning(
            "Relaxing forced tool_choice %r to %r for model %r — model rejects forced tool choice",
            original,
            relaxed,
            model_name,
        )
    return {**kwargs, "tool_choice": relaxed}
