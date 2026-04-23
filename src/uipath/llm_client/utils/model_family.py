"""Heuristic helpers for identifying a model's family from its name.

Discovery metadata (``modelFamily``) is the authoritative source, but BYOM
deployments do not expose it. These helpers provide a name-based fallback.
"""

import re

ANTHROPIC_MODEL_NAME_KEYWORDS: tuple[str, ...] = (
    "anthropic",
    "claude",
    "opus",
    "sonnet",
    "haiku",
    "mythos",
)

# Sampling parameters that Claude Opus 4+ (and similar reasoning models) do not support.
# The Anthropic API returns 400 Bad Request if any of these appear in the request payload.
CLAUDE_OPUS_4_UNSUPPORTED_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {"temperature", "top_k", "top_p"}
)


def is_anthropic_model_name(model_name: str) -> bool:
    """Return True if ``model_name`` looks like an Anthropic Claude-family model."""
    lower = model_name.lower()
    return any(kw in lower for kw in ANTHROPIC_MODEL_NAME_KEYWORDS)


def is_claude_opus_4_or_above(model_name: str) -> bool:
    """Return True for Claude models confirmed to reject sampling parameters.

    Tested against the Anthropic API: only ``claude-opus-4-7`` rejects
    ``temperature``, ``top_k``, and ``top_p`` with ``400 Bad Request``.
    ``claude-opus-4-5`` and ``claude-opus-4-6`` accept sampling parameters.

    Extend the pattern here as additional models are confirmed to restrict them.
    """
    return bool(re.search(r"claude-opus-4-7", model_name, re.IGNORECASE))
