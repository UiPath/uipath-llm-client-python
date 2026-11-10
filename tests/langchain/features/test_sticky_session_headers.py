"""The LLM Gateway sticky-session headers reach the chat model's HTTP clients."""

import os
from unittest.mock import patch

from uipath_langchain_client.clients.normalized.chat_models import UiPathChat

from uipath.llm_client.settings import PlatformSettings

PLATFORM_ENV = {
    "UIPATH_ACCESS_TOKEN": "test-access-token",
    "UIPATH_URL": "https://cloud.uipath.com/org/tenant",
    "UIPATH_TENANT_ID": "test-tenant-id",
    "UIPATH_ORGANIZATION_ID": "test-org-id",
    "UIPATH_JOB_KEY": "job-123",
}
SETTINGS_MODULE = "uipath.llm_client.settings.platform.settings"


def test_chat_model_clients_carry_the_sticky_session_headers():
    with (
        patch.dict(os.environ, PLATFORM_ENV, clear=True),
        patch(f"{SETTINGS_MODULE}.is_token_expired", return_value=False),
        patch(f"{SETTINGS_MODULE}.try_parse_access_token", return_value=None),
        patch(f"{SETTINGS_MODULE}.UiPathConfig") as config,
    ):
        config.licensing_context = None
        config.conversation_id = "conversation-456"
        chat = UiPathChat(
            model="gpt-5-mini-2025-08-07", settings=PlatformSettings(), model_details={}
        )
        for client in (chat.uipath_sync_client, chat.uipath_async_client):
            assert client.headers["x-uipath-llmgateway-stickysession"] == "true"
            assert (
                client.headers["x-uipath-llmgateway-stickysession-transactionid"]
                == "conversation-456"
            )
