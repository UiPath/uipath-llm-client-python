import base64
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


def is_token_expired(token: str) -> bool:
    """Check whether a JWT access token has expired.

    Args:
        token: A JWT token string.

    Returns:
        True if the token is expired, False if it is still valid or has no ``exp`` claim.
    """
    token_data = parse_access_token(token)
    exp = token_data.get("exp")
    if exp is None:
        return False
    return exp < time.time()
