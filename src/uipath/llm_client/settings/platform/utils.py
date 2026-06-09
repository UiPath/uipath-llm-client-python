import base64
import binascii
import json
import time
from typing import Any


def parse_access_token(access_token: str) -> dict[str, Any]:
    """Parse a JWT access token and return the payload as a dict.

    Args:
        access_token: A JWT token string (header.payload.signature).

    Returns:
        The decoded payload as a dictionary.

    Raises:
        ValueError: If the token is malformed or cannot be decoded.
    """
    token_parts = access_token.split(".")
    if len(token_parts) < 2:
        raise ValueError("Invalid access token: expected JWT with at least 2 dot-separated parts")
    try:
        payload = base64.urlsafe_b64decode(token_parts[1] + "=" * (-len(token_parts[1]) % 4))
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid access token: failed to decode payload: {e}") from e


def try_parse_access_token(access_token: str) -> dict[str, Any] | None:
    """Best-effort parse of an access token's JWT payload.

    Access tokens are not guaranteed to be JWTs — UiPath also issues opaque
    tokens (e.g. reference tokens) that carry no client-readable claims. This
    returns the decoded payload when the token is a parseable JWT, or ``None``
    when it is not, instead of raising.

    Args:
        access_token: An access token string of any form.

    Returns:
        The decoded payload as a dictionary, or ``None`` if the token is not a
        parseable JWT.
    """
    try:
        return parse_access_token(access_token)
    except (ValueError, binascii.Error):
        return None


def is_token_expired(token: str) -> bool:
    """Check whether an access token has expired.

    Args:
        token: An access token string of any form.

    Returns:
        True if the token is a JWT with an ``exp`` claim in the past; False if
        it is still valid, has no ``exp`` claim, or is an opaque token whose
        expiry cannot be inspected.
    """
    token_data = try_parse_access_token(token)
    if token_data is None:
        return False
    exp = token_data.get("exp")
    if exp is None:
        return False
    return exp < time.time()
