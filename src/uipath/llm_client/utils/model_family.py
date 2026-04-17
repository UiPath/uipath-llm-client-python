"""Heuristic helpers for identifying a model's family from its name.

Discovery metadata (``modelFamily``) is the authoritative source, but BYOM
deployments do not expose it. These helpers provide a name-based fallback.
"""

ANTHROPIC_MODEL_NAME_KEYWORDS: tuple[str, ...] = (
    "anthropic",
    "claude",
    "opus",
    "sonnet",
    "haiku",
    "mythos",
)


def is_anthropic_model_name(model_name: str) -> bool:
    """Return True if ``model_name`` looks like an Anthropic Claude-family model."""
    lower = model_name.lower()
    return any(kw in lower for kw in ANTHROPIC_MODEL_NAME_KEYWORDS)
