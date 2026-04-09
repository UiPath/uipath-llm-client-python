import base64
import json
from collections.abc import Generator
from typing import Any

from httpx import Client

try:
    from botocore.eventstream import EventStreamBuffer
except ImportError as e:
    raise ImportError(
        "The 'bedrock' extra is required to use WrappedBotoClient. "
        "Install it with: uv add uipath-langchain-client[bedrock]"
    ) from e


class _MockEventHooks:
    """Mock event hooks that mimics boto3's event registration system."""

    def register(self, event_name: str, handler: Any) -> None:
        """No-op register method to satisfy langchain_aws's header registration."""
        pass


class _MockClientMeta:
    """Mock client meta that mimics boto3's client.meta structure."""

    def __init__(self, region_name: str = "PLACEHOLDER"):
        self.region_name = region_name
        self.events = _MockEventHooks()


def _serialize_bytes(obj: Any) -> Any:
    """Recursively encode bytes values to base64 strings for JSON serialization.

    This mimics boto3's serializer which re-encodes bytes to base64 before
    sending as JSON. Needed because LangChain's ChatBedrockConverse decodes
    base64 content (images, PDFs) into raw bytes objects.
    """
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    if isinstance(obj, dict):
        return {k: _serialize_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_bytes(item) for item in obj]
    return obj


class WrappedBotoClient:
    def __init__(self, httpx_client: Client | None = None, region_name: str = "PLACEHOLDER"):
        self.httpx_client = httpx_client
        self.meta = _MockClientMeta(region_name=region_name)

    def _stream_generator(
        self, request_body: dict[str, Any]
    ) -> Generator[dict[str, Any], None, None]:
        if self.httpx_client is None:
            raise ValueError("httpx_client is not set")
        with self.httpx_client.stream("POST", "/", json=_serialize_bytes(request_body)) as response:
            buffer = EventStreamBuffer()
            for chunk in response.iter_bytes():
                buffer.add_data(chunk)
                for event in buffer:
                    event_as_dict = event.to_response_dict()
                    dict_key = event_as_dict["headers"][":event-type"]
                    dict_value = json.loads(event_as_dict["body"].decode("utf-8"))
                    if "bytes" in dict_value:
                        dict_value["bytes"] = base64.b64decode(dict_value["bytes"])
                    yield {dict_key: dict_value}

    def invoke_model(self, **kwargs: Any) -> Any:
        if self.httpx_client is None:
            raise ValueError("httpx_client is not set")
        return {
            "body": self.httpx_client.post(
                "/",
                json=json.loads(kwargs.get("body", "{}")),
            )
        }

    def invoke_model_with_response_stream(self, **kwargs: Any) -> Any:
        return {"body": self._stream_generator(json.loads(kwargs.get("body", "{}")))}

    def converse(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> Any:
        if self.httpx_client is None:
            raise ValueError("httpx_client is not set")
        return self.httpx_client.post(
            "/",
            json=_serialize_bytes(
                {
                    "messages": messages,
                    "system": system,
                    **params,
                }
            ),
        ).json()

    def converse_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> Any:
        return {
            "stream": self._stream_generator(
                {
                    "messages": messages,
                    "system": system,
                    **params,
                }
            ),
        }
