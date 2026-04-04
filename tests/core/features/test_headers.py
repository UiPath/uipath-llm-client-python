"""Tests for header utilities."""

from httpx import Headers

from uipath.llm_client.settings import UiPathAPIConfig
from uipath.llm_client.settings.constants import ApiType, RoutingMode


class TestHeaderUtilities:
    """Tests for extract_matching_headers and context var functions."""

    def test_extract_matching_headers_basic(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"x-uipath-foo": "bar", "content-type": "json", "x-uipath-baz": "qux"})
        result = extract_matching_headers(headers, ["x-uipath-"])
        assert result == {"x-uipath-foo": "bar", "x-uipath-baz": "qux"}

    def test_extract_matching_headers_case_insensitive(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"X-UiPath-Foo": "bar"})
        result = extract_matching_headers(headers, ["x-uipath-"])
        assert len(result) == 1
        assert "bar" in result.values()

    def test_extract_matching_headers_no_match(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"content-type": "json"})
        result = extract_matching_headers(headers, ["x-uipath-"])
        assert result == {}

    def test_extract_matching_headers_empty_prefixes(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"x-uipath-foo": "bar"})
        result = extract_matching_headers(headers, [])
        assert result == {}

    def test_extract_matching_headers_multiple_prefixes(self):
        from uipath.llm_client.utils.headers import extract_matching_headers

        headers = Headers({"x-uipath-foo": "1", "x-custom-bar": "2", "content-type": "json"})
        result = extract_matching_headers(headers, ["x-uipath-", "x-custom-"])
        assert len(result) == 2

    def test_context_var_get_default(self):
        from uipath.llm_client.utils.headers import get_captured_response_headers

        result = get_captured_response_headers()
        assert isinstance(result, dict)

    def test_context_var_set_and_get(self):
        from uipath.llm_client.utils.headers import (
            get_captured_response_headers,
            set_captured_response_headers,
        )

        _ = set_captured_response_headers({"x-uipath-test": "value"})
        result = get_captured_response_headers()
        assert result == {"x-uipath-test": "value"}
        # Returns a copy
        result["new-key"] = "new-value"
        assert "new-key" not in get_captured_response_headers()

    def test_passthrough_embeddings_no_flavor_header(self):
        from uipath.llm_client.utils.headers import build_routing_headers

        api_config = UiPathAPIConfig(
            api_type=ApiType.EMBEDDINGS,
            routing_mode=RoutingMode.PASSTHROUGH,
            vendor_type="openai",
        )
        headers = build_routing_headers(model_name="text-embedding-3-large", api_config=api_config)
        assert "X-UiPath-LlmGateway-ApiFlavor" not in headers
        assert "X-UiPath-LlmGateway-NormalizedApi-ModelName" not in headers

    def test_normalized_without_model_name(self):
        from uipath.llm_client.utils.headers import build_routing_headers

        api_config = UiPathAPIConfig(
            api_type=ApiType.COMPLETIONS,
            routing_mode=RoutingMode.NORMALIZED,
        )
        headers = build_routing_headers(model_name=None, api_config=api_config)
        assert "X-UiPath-LlmGateway-NormalizedApi-ModelName" not in headers
