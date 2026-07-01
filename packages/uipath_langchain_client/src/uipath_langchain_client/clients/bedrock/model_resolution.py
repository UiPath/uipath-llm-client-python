"""Bedrock backing-model and provider resolution.

Pure string/dict helpers that derive the real upstream Bedrock model id (and its
provider) from an LLM Gateway discovery record. Kept free of ``botocore`` /
``langchain_aws`` so they can be imported and tested without the ``bedrock`` extra,
and so both the factory and the Bedrock clients depend on this module rather than on
each other.
"""

from typing import Any

_BEDROCK_REGION_PREFIXES = (
    "eu.",
    "us.",
    "us-gov.",
    "apac.",
    "sa.",
    "amer.",
    "global.",
    "jp.",
    "au.",
)


def _normalize_model_id(model_id: Any) -> str | None:
    if not isinstance(model_id, str):
        return None

    model_id = model_id.strip()
    if not model_id:
        return None

    foundation_model_marker = "/foundation-model/"
    if foundation_model_marker in model_id:
        model_id = model_id.rsplit(foundation_model_marker, 1)[1]

    model_id_without_region = model_id
    for region_prefix in _BEDROCK_REGION_PREFIXES:
        if model_id_without_region.startswith(region_prefix):
            model_id_without_region = model_id_without_region[len(region_prefix) :]
            break

    if "." not in model_id_without_region:
        return None

    provider = model_id_without_region.split(".", 1)[0]
    if not provider or " " in provider or "/" in provider:
        return None

    return model_id


def _resolve_backing_model_id(model_info: dict[str, Any]) -> str | None:
    byo_details = model_info.get("byomDetails") or {}

    candidates = (
        # Authoritative source: the upstream model the customer configured for a
        # BYO ("add your own") connection. LLM Gateway discovery exposes it as
        # byomDetails.customerModel (LLM-3900). For UiPath-owned models it is
        # absent and we fall back to the discovery model name (the alias the
        # factory was instantiated with, which is itself the real model id).
        byo_details.get("customerModel"),
        model_info.get("modelName"),
    )
    for candidate in candidates:
        backing_model_id = _normalize_model_id(candidate)
        if backing_model_id:
            return backing_model_id
    return None


def provider_from_model(base_model_id: str | None) -> str | None:
    base_model_id = _normalize_model_id(base_model_id)
    if not base_model_id:
        return None
    model_id_without_region = base_model_id
    for region_prefix in _BEDROCK_REGION_PREFIXES:
        if model_id_without_region.startswith(region_prefix):
            model_id_without_region = model_id_without_region[len(region_prefix) :]
            break
    provider = model_id_without_region.split(".", 1)[0]
    return provider


def apply_backing_model_detection_hints(
    model_kwargs: dict[str, Any], model_info: dict[str, Any]
) -> None:
    # These fields are read by both Invoke and Converse clients to determine vendor
    backing_model_id = _resolve_backing_model_id(model_info)
    if backing_model_id:
        model_kwargs.setdefault("base_model_id", backing_model_id)
    provider = provider_from_model(backing_model_id)
    if provider:
        model_kwargs.setdefault("provider", provider)
