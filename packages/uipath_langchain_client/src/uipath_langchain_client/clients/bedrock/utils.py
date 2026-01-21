import base64
import json
from typing import Any, Iterator

from httpx import Client

try:
    from botocore.eventstream import EventStreamBuffer
except ImportError as e:
    raise ImportError(
        "The 'aws' extra is required to use WrappedBotoClient. "
        "Install it with: uv add uipath-langchain-client[aws]"
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


class WrappedBotoClient:
    def __init__(self, httpx_client: Client | None = None, region_name: str = "PLACEHOLDER"):
        self.httpx_client = httpx_client
        self.meta = _MockClientMeta(region_name=region_name)

    def _stream_generator(self, request_body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if self.httpx_client is None:
            raise ValueError("httpx_client is not set")
        with self.httpx_client.stream("POST", "/", json=request_body) as response:
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
                json=json.loads(kwargs.get("body", {})),
            )
        }

    def invoke_model_with_response_stream(self, **kwargs: Any) -> Any:
        return {"body": self._stream_generator(json.loads(kwargs.get("body", {})))}

    def converse(
        self, *, messages: list[dict[str, Any]], system: str | None = None, **params: Any
    ) -> Any:
        if self.httpx_client is None:
            raise ValueError("httpx_client is not set")
        return self.httpx_client.post(
            "/",
            json={
                "messages": messages,
                "system": system,
                **params,
            },
        ).json()

    def converse_stream(
        self, *, messages: list[dict[str, Any]], system: str | None = None, **params: Any
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
