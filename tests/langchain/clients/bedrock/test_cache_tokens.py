import httpx
from uipath_langchain_client.clients.bedrock.chat_models import UiPathChatBedrockConverse

from tests.langchain.transports import route_transport
from uipath.llm_client.settings import LLMGatewaySettings

CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 40,
        "outputTokens": 5,
        "totalTokens": 1005,
        "cacheReadInputTokens": 900,
        "cacheWriteInputTokens": 60,
    },
    "metrics": {"latencyMs": 1},
}


def test_converse_input_tokens_include_cached_tokens(llmgw_settings: LLMGatewaySettings) -> None:
    chat = UiPathChatBedrockConverse(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        settings=llmgw_settings,
        model_details={},
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=CONVERSE_RESPONSE))
    route_transport(chat, transport)

    usage = chat.invoke("hi").usage_metadata

    assert usage is not None
    assert usage["input_tokens"] == 40 + 900 + 60
    assert usage.get("input_token_details", {}).get("cache_read") == 900
