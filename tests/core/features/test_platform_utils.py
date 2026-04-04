"""Tests for platform utility functions (JWT parsing and token expiry)."""

import base64
import json
import time

import pytest

from uipath.llm_client.settings.platform.utils import is_token_expired, parse_access_token


class TestParseAccessToken:
    def test_valid_jwt(self):
        payload = {"sub": "user-123", "exp": 9999999999, "iss": "uipath"}
        encoded_payload = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )
        token = f"eyJhbGciOiJSUzI1NiJ9.{encoded_payload}.fake-signature"

        result = parse_access_token(token)
        assert result["sub"] == "user-123"
        assert result["exp"] == 9999999999
        assert result["iss"] == "uipath"

    def test_malformed_token_no_dots(self):
        with pytest.raises(ValueError, match="Invalid access token"):
            parse_access_token("not-a-jwt-token")

    def test_malformed_token_bad_base64(self):
        # base64 decode of invalid padding raises binascii.Error (not caught by source)
        import binascii

        with pytest.raises(binascii.Error):
            parse_access_token("header.!!!invalid-base64!!!.signature")

    def test_malformed_token_not_json(self):
        # Valid base64 but not JSON content
        encoded = base64.urlsafe_b64encode(b"not json content").decode().rstrip("=")
        with pytest.raises(ValueError, match="Invalid access token"):
            parse_access_token(f"header.{encoded}.signature")


class TestIsTokenExpired:
    def test_expired_token(self):
        expired_time = int(time.time()) - 3600  # 1 hour ago
        payload = {"sub": "user-123", "exp": expired_time}
        encoded_payload = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )
        token = f"header.{encoded_payload}.signature"

        assert is_token_expired(token) is True

    def test_valid_token(self):
        future_time = int(time.time()) + 3600  # 1 hour from now
        payload = {"sub": "user-123", "exp": future_time}
        encoded_payload = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )
        token = f"header.{encoded_payload}.signature"

        assert is_token_expired(token) is False

    def test_missing_exp_claim(self):
        payload = {"sub": "user-123"}  # no "exp" claim
        encoded_payload = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )
        token = f"header.{encoded_payload}.signature"

        assert is_token_expired(token) is False
