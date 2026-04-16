"""Shared model discovery helpers."""

from typing import Any


def get_model_info(
    available_models: list[dict[str, Any]],
    model_name: str,
    *,
    vendor_type: str | None = None,
    byo_connection_id: str | None = None,
) -> dict[str, Any]:
    """Find and return a single model entry from the discovery endpoint results.

    Applies the following filters in order:

    1. Match by ``modelName`` (case-insensitive).
    2. If ``vendor_type`` is given, keep only models whose ``vendor`` matches.
    3. If ``byo_connection_id`` is given, keep only models whose
       ``byomDetails.integrationServiceConnectionId`` matches.
    4. When no ``byo_connection_id`` is provided and multiple candidates remain,
       prefer UiPath-owned (non-BYOM) models.

    Args:
        available_models: Full list of model dictionaries from the discovery
            endpoint (as returned by :meth:`UiPathBaseSettings.get_available_models`).
        model_name: Name of the model to look up.
        vendor_type: Optional vendor filter (e.g. ``"openai"``).
        byo_connection_id: Optional BYOM connection ID filter.

    Returns:
        The first matching model dictionary.

    Raises:
        ValueError: If no model matches the given criteria.
    """
    matching = [m for m in available_models if m["modelName"].lower() == model_name.lower()]

    if vendor_type is not None:
        matching = [m for m in matching if m.get("vendor", "").lower() == str(vendor_type).lower()]

    if byo_connection_id:
        matching = [
            m
            for m in matching
            if (byom_details := m.get("byomDetails"))
            and byom_details.get("integrationServiceConnectionId", "").lower()
            == byo_connection_id.lower()
        ]

    if not byo_connection_id and len(matching) > 1:
        matching = [
            m
            for m in matching
            if (
                (m.get("modelSubscriptionType", "") == "UiPathOwned")
                or (m.get("byomDetails") is None)
            )
        ]

    if not matching:
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {[m['modelName'] for m in available_models]}"
        )

    return matching[0]
