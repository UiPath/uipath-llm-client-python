"""Tests for UiPathAPIConfig and enum constants."""

import pytest

from uipath.llm_client.settings import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiFlavor, ApiType, RoutingMode, VendorType


class TestUiPathAPIConfig:
    """Tests for UiPathAPIConfig."""

    def test_passthrough_requires_vendor_type(self):
        """Test that passthrough mode requires vendor_type."""
        with pytest.raises(ValueError, match="vendor_type required"):
            UiPathAPIConfig(
                api_type=ApiType.COMPLETIONS,
                routing_mode=RoutingMode.PASSTHROUGH,
                vendor_type=None,
            )

    def test_normalized_does_not_require_vendor_type(self):
        """Test that normalized mode doesn't require vendor_type."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
        )
        assert config.vendor_type is None

    def test_passthrough_with_vendor_type(self):
        """Test passthrough config with vendor_type."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
        )
        assert config.api_type == ApiType.COMPLETIONS
        assert config.routing_mode == RoutingMode.PASSTHROUGH
        assert config.vendor_type == "openai"

    def test_freeze_base_url_default(self):
        """Test freeze_base_url defaults to False."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
        )
        assert config.freeze_base_url is False

    def test_api_flavor_and_version(self):
        """Test api_flavor and api_version can be set."""
        config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
            api_flavor="chat-completions",
            api_version="2025-03-01-preview",
        )
        assert config.api_flavor == "chat-completions"
        assert config.api_version == "2025-03-01-preview"


class TestEnumConstants:
    """Tests for StrEnum constants."""

    def test_api_type_values(self):
        assert ApiType.COMPLETIONS == "completions"
        assert ApiType.EMBEDDINGS == "embeddings"

    def test_routing_mode_values(self):
        assert RoutingMode.PASSTHROUGH == "passthrough"
        assert RoutingMode.NORMALIZED == "normalized"

    def test_vendor_type_values(self):
        assert VendorType.OPENAI == "openai"
        assert VendorType.VERTEXAI == "vertexai"
        assert VendorType.AWSBEDROCK == "awsbedrock"
        assert VendorType.AZURE == "azure"
        assert VendorType.ANTHROPIC == "anthropic"

    def test_api_flavor_values(self):
        assert ApiFlavor.CHAT_COMPLETIONS == "chat-completions"
        assert ApiFlavor.RESPONSES == "responses"
        assert ApiFlavor.GENERATE_CONTENT == "generate-content"
        assert ApiFlavor.CONVERSE == "converse"
        assert ApiFlavor.INVOKE == "invoke"
        assert ApiFlavor.ANTHROPIC_CLAUDE == "anthropic-claude"

    def test_enum_string_comparison(self):
        assert ApiType.COMPLETIONS == "completions"
        assert RoutingMode.PASSTHROUGH == "passthrough"
        assert VendorType.OPENAI == "openai"

    def test_enum_is_str_subclass(self):
        assert isinstance(ApiType.COMPLETIONS, str)
        assert isinstance(RoutingMode.PASSTHROUGH, str)
        assert isinstance(VendorType.OPENAI, str)
        assert isinstance(ApiFlavor.CHAT_COMPLETIONS, str)
