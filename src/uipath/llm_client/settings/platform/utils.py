import base64
import json
import time
from pathlib import Path

from uipath.platform.common.auth import TokenData


def get_auth_data() -> TokenData:
    auth_file = Path.cwd() / ".uipath" / ".auth.json"
    if not auth_file.exists():
        raise FileNotFoundError("No authentication file found")
    return TokenData.model_validate(json.load(open(auth_file)))


def parse_access_token(access_token: str):
    token_parts = access_token.split(".")
    if len(token_parts) < 2:
        raise Exception("Invalid access token")
    payload = base64.urlsafe_b64decode(token_parts[1] + "=" * (-len(token_parts[1]) % 4))
    return json.loads(payload)


def is_token_expired(token: str) -> bool:
    token_data = parse_access_token(token)
    return token_data["exp"] < time.time()
