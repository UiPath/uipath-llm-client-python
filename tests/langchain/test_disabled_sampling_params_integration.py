"""End-to-end check: constructor-level ``temperature`` survives ``shouldSkipTemperature``.

Recorded against the live LLM Gateway via the SQLite-backed VCR persister
(see ``tests/conftest.py`` and ``tests/sqlite_persister.py``). The cassette
captures a 200 response — which is itself proof that ``strip_disabled_fields``
nulled the constructor-set field before the vendor SDK serialized the request
body. Without the fix, the gateway returns 400 for any sampling-knob value on
``anthropic.claude-opus-4-7`` (modelDetails advertises
``shouldSkipTemperature: True``), and the ``before_record_response`` filter in
``conftest.py`` would refuse to persist the failed exchange.

We exercise both vendor SDK families because they read ``self.temperature`` at
different layers:
- ``UiPathChatAnthropicBedrock`` -> langchain-anthropic's ``ChatAnthropic``
- ``UiPathChatBedrockConverse`` -> langchain-aws's ``ChatBedrockConverse``
"""

import pytest
from langchain_core.messages import HumanMessage
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrockConverse,
)

from uipath.llm_client.settings import UiPathBaseSettings

OPUS_4_7 = "anthropic.claude-opus-4-7"


@pytest.mark.vcr
def test_opus_4_7_constructor_temperature_with_anthropic_bedrock(
    client_settings: UiPathBaseSettings,
) -> None:
    chat = UiPathChatAnthropicBedrock(
        model=OPUS_4_7,
        settings=client_settings,
        # Skip discovery so the cassette only captures the chat completion.
        model_details={"shouldSkipTemperature": True},
        temperature=0.7,
    )
    # Eager strip: temperature was nulled at construction so the vendor SDK
    # serializes the request body without it.
    assert chat.temperature is None

    response = chat.invoke([HumanMessage(content="Reply with the single word: pong")])
    assert response.content, "expected a non-empty response from the gateway"


@pytest.mark.vcr
def test_opus_4_7_constructor_temperature_with_bedrock_converse(
    client_settings: UiPathBaseSettings,
) -> None:
    chat = UiPathChatBedrockConverse(
        model=OPUS_4_7,
        settings=client_settings,
        model_details={"shouldSkipTemperature": True},
        temperature=0.7,
    )
    assert chat.temperature is None

    response = chat.invoke([HumanMessage(content="Reply with the single word: pong")])
    assert response.content
