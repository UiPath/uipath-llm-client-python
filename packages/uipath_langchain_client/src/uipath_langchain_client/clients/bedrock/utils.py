import base64
import json
import logging
from collections.abc import Generator, Iterable
from typing import Any

from httpx import Client

try:
    from botocore.eventstream import EventStreamBuffer
except ImportError as e:
    raise ImportError(
        "The 'bedrock' extra is required to use WrappedBotoClient. "
        "Install it with: uv add uipath-langchain-client[bedrock]"
    ) from e

logger = logging.getLogger(__name__)

CONVERSE_STREAM_EVENT_TYPES = frozenset(
    {
        "messageStart",
        "contentBlockStart",
        "contentBlockDelta",
        "contentBlockStop",
        "messageStop",
        "metadata",
    }
)
"""Converse stream event types `langchain_aws` can parse.

It also turns AWS `*Exception` events into errors and raises on everything else.
"""

GATEWAY_COST_EVENT_TYPE = "costMetadata"
"""Per-call cost frame the LLM Gateway appends after the terminal AWS frame."""


def reconcile_converse_stream_events(
    events: Iterable[dict[str, Any]],
) -> Generator[dict[str, Any], None, None]:
    """Keep gateway-only event-stream frames out of an AWS-shaped stream.

    AWS events pass through untouched and in order. The cost frame's payload is
    folded into the terminal `metadata` event, so it reaches `response_metadata`
    under the same keys the gateway uses on non-streamed responses, and AWS keys
    win a collision. Every other unrecognized event is dropped.

    Only `metadata` is buffered, so content events are still yielded as they
    arrive.
    """
    pending_metadata: dict[str, Any] | None = None
    cost: dict[str, Any] = {}

    for event in events:
        event_type = next(iter(event), "")
        if event_type in CONVERSE_STREAM_EVENT_TYPES or "Exception" in event_type:
            if pending_metadata is not None:
                yield pending_metadata
                pending_metadata = None
            if event_type == "metadata":
                pending_metadata = event
                continue
            yield event
        elif event_type == GATEWAY_COST_EVENT_TYPE:
            payload = event[event_type]
            if isinstance(payload, dict):
                cost.update(payload)
        else:
            logger.debug("Dropping unrecognized Bedrock stream event %r", event_type)

    if pending_metadata is not None:
        metadata = pending_metadata["metadata"]
        for key, value in cost.items():
            metadata.setdefault(key, value)
        yield pending_metadata
    elif cost:
        logger.debug("No metadata event to carry gateway cost %r", cost)


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
            if response.is_error:
                # The gateway returns a non-streamed JSON error body; read it so
                # the patched raise_for_status surfaces it (with detail) instead
                # of the EventStreamBuffer choking on a non-event payload.
                response.read()
            response.raise_for_status()
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
        response = self.httpx_client.post(
            "/",
            json=json.loads(kwargs.get("body", "{}")),
        )
        response.raise_for_status()
        return {"body": response}

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
        response = self.httpx_client.post(
            "/",
            json=_serialize_bytes(
                {
                    "messages": messages,
                    "system": system,
                    **params,
                }
            ),
        )
        response.raise_for_status()
        return response.json()

    def converse_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> Any:
        return {
            "stream": reconcile_converse_stream_events(
                self._stream_generator(
                    {
                        "messages": messages,
                        "system": system,
                        **params,
                    }
                )
            ),
        }
