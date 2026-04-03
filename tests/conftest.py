import pytest

from uipath.llm_client.settings import UiPathBaseSettings
from uipath.llm_client.settings.llmgateway import LLMGatewaySettings
from uipath.llm_client.settings.platform import PlatformSettings


@pytest.fixture(autouse=True, scope="session")
def setup_env():
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())


@pytest.fixture(scope="session")
def vcr_config():
    import re

    from vcr import VCR
    from vcr.request import Request as VCRRequest

    def filter_requests(request: VCRRequest) -> VCRRequest | None:
        if "identity_/connect/token" in request.uri:
            return None
        is_uuid = lambda s: bool(
            re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", s, re.I)
        )
        request.uri = "/".join(
            ["PLACEHOLDER" if is_uuid(item) else item for item in request.uri.split("/") if item]
        )
        request.headers = None
        return request

    def filter_responses(response: dict) -> dict | None:
        """Don't record responses with 4xx or 5xx status codes."""
        status_code = response.get("status", {}).get("code", 200)
        if 200 <= status_code < 300:
            return response
        return None

    return {
        "record_mode": "new_episodes",
        "record_on_exception": False,
        "allow_playback_repeats": False,
        "decode_compressed_response": True,
        "cassette_library_dir": "tests/cassettes",
        "path_transformer": VCR.ensure_suffix(".yaml"),
        "before_record_request": filter_requests,
        "before_record_response": filter_responses,
    }


def pytest_recording_configure(config, vcr):
    from tests.sqlite_persister import SQLitePersister

    vcr.register_persister(SQLitePersister)


# Only "llmgw" is parameterized because Platform (agenthub) requires `uipath auth`
# credentials that are not available in CI. Platform-specific logic is tested
# via mocked settings in test_base_client.py.
@pytest.fixture(scope="session", params=["llmgw"])
def client_settings(request: pytest.FixtureRequest) -> UiPathBaseSettings:
    match request.param:
        case "llmgw":
            return LLMGatewaySettings()
        case "agenthub":
            return PlatformSettings()
        case _:
            raise ValueError(f"Invalid client type: {request.param}")
