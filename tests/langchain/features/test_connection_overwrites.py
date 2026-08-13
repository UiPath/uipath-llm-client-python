"""Tests that a job's connection binding reaches BYOM models.

A solution deployment binds the agent to a connection in the target folder, while the
design-time ``byo_connection_id`` travels with the package. Callers that build their own
chat models never see the binding, so the ``connection.<id>`` overwrite has to be applied
where the id is consumed: the discovery lookup and the routing header.
"""

import os
from unittest.mock import MagicMock, patch

from httpx import Client
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.factory import get_chat_model

from uipath.llm_client.settings import LLMGatewaySettings, RoutingMode
from uipath.llm_client.settings.base import UiPathBaseSettings
from uipath.llm_client.settings.utils import SingletonMeta

LLMGW_ENV = {
    "LLMGW_URL": "https://cloud.uipath.com",
    "LLMGW_SEMANTIC_ORG_ID": "test-org-id",
    "LLMGW_SEMANTIC_TENANT_ID": "test-tenant-id",
    "LLMGW_REQUESTING_PRODUCT": "test-product",
    "LLMGW_REQUESTING_FEATURE": "test-feature",
    "LLMGW_ACCESS_TOKEN": "test-access-token",
}

DESIGN_TIME_CONNECTION_ID = "design-time-conn"
BOUND_CONNECTION_ID = "bound-conn"
BYO_MODEL_DETAILS = {"contextWindowSize": 128000}
BYO_CONNECTION_HEADER = "x-uipath-llmgateway-byoisconnectionid"

MODELS = [
    {"modelName": "gpt-4o", "vendor": "OpenAi", "modelSubscriptionType": "UiPathOwned"},
    {
        "modelName": "gpt-4o",
        "vendor": "OpenAi",
        "modelSubscriptionType": "BYO",
        "byomDetails": {"integrationServiceConnectionId": BOUND_CONNECTION_ID},
        "modelDetails": BYO_MODEL_DETAILS,
    },
]


class TestConnectionOverwriteReachesByoModels:
    def setup_method(self):
        SingletonMeta._instances.clear()
        UiPathBaseSettings._discovery_cache.clear()

    def teardown_method(self):
        SingletonMeta._instances.clear()
        UiPathBaseSettings._discovery_cache.clear()

    def _settings(self):
        """Build settings with the discovery cache pre-populated from ``MODELS``."""
        settings = LLMGatewaySettings()
        response = MagicMock()
        response.is_error = False
        response.json.return_value = MODELS
        with patch.object(Client, "get", return_value=response):
            settings.get_available_models()
        return settings

    def test_direct_instantiation_routes_to_bound_connection(self, activate_connection_overwrite):
        activate_connection_overwrite(DESIGN_TIME_CONNECTION_ID, BOUND_CONNECTION_ID)

        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(
                model="gpt-4o",
                settings=self._settings(),
                byo_connection_id=DESIGN_TIME_CONNECTION_ID,
            )
            headers = chat.uipath_sync_client.headers

        assert chat.model_details == BYO_MODEL_DETAILS
        assert headers[BYO_CONNECTION_HEADER] == BOUND_CONNECTION_ID

    def test_factory_routes_to_bound_connection(self, activate_connection_overwrite):
        activate_connection_overwrite(DESIGN_TIME_CONNECTION_ID, BOUND_CONNECTION_ID)

        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = get_chat_model(
                model_name="gpt-4o",
                client_settings=self._settings(),
                byo_connection_id=DESIGN_TIME_CONNECTION_ID,
                routing_mode=RoutingMode.NORMALIZED,
            )
            headers = chat.uipath_sync_client.headers

        assert chat.model_details == BYO_MODEL_DETAILS
        assert headers[BYO_CONNECTION_HEADER] == BOUND_CONNECTION_ID

    def test_unbound_connection_id_is_left_alone(self, activate_connection_overwrite):
        activate_connection_overwrite("some-other-conn", BOUND_CONNECTION_ID)

        with patch.dict(os.environ, LLMGW_ENV, clear=True):
            chat = UiPathChat(
                model="gpt-4o",
                settings=self._settings(),
                byo_connection_id=BOUND_CONNECTION_ID,
            )
            headers = chat.uipath_sync_client.headers

        assert chat.model_details == BYO_MODEL_DETAILS
        assert headers[BYO_CONNECTION_HEADER] == BOUND_CONNECTION_ID
