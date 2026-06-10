"""Bedrock backing-model and provider resolution."""

from typing import Any

_BEDROCK_REGIONS = (
    "eu",
    "us",
    "us-gov",
    "apac",
    "sa",
    "amer",
    "global",
    "jp",
    "au",
)


def _split_model_id(model_id: str) -> tuple[str, str] | None:
    model_id = model_id.strip()
    parts = model_id.split(".")

    if parts[0] in _BEDROCK_REGIONS:
        parts = parts[1:]

    if len(parts) < 2:
        return None

    provider = parts[0]
    if not provider or " " in provider or "/" in provider:
        return None

    return model_id, provider


def _resolve_backing_model(model_info: dict[str, Any]) -> tuple[str, str] | None:
    byo_details = model_info.get("byomDetails") or {}
    customer_model = byo_details.get("customerModel")
    if isinstance(customer_model, str):
        return _split_model_id(customer_model)
    return None


def apply_backing_model_detection_hints(
    model_kwargs: dict[str, Any], model_info: dict[str, Any]
) -> None:
    if "base_model_id" in model_kwargs or "base_model" in model_kwargs:
        return
    backing = _resolve_backing_model(model_info)
    if not backing:
        return
    base_model_id, provider = backing
    model_kwargs["base_model_id"] = base_model_id
    model_kwargs.setdefault("provider", provider)
