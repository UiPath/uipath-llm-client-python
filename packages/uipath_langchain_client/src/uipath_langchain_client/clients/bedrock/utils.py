import base64
import json
from collections.abc import Generator
from typing import Any

from httpx import Client, Request, Response

from uipath.llm_client.utils.dollar_cost import COST_METADATA_EVENT_TYPE
from uipath.llm_client.utils.exceptions import UiPathAPIError

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


# HTTP status for each Bedrock in-stream exception, as the AWS API documents it
# for the same exception raised before the stream starts. The exception is the
# only signal: the HTTP response itself was already a 200.
# modelStreamErrorException is 424 in AWS; it is mapped to 502 here because it is
# an upstream failure part-way through the model's output, which AWS documents
# as retryable and which callers should treat as a provider (5xx) error.
_STREAM_EXCEPTION_STATUS: dict[str, int] = {
    "validationException": 400,
    "accessDeniedException": 403,
    "resourceNotFoundException": 404,
    "modelTimeoutException": 408,
    "throttlingException": 429,
    "internalServerException": 500,
    "modelStreamErrorException": 502,
    "serviceUnavailableException": 503,
}
_DEFAULT_STREAM_EXCEPTION_STATUS = 500


def _stream_exception_error(
    headers: dict[str, Any], payload: bytes, request: Request
) -> UiPathAPIError:
    """Turn an AWS event-stream ``exception`` / ``error`` frame into a UiPath error.

    Bedrock reports a failure part-way through a stream as a frame with
    ``:message-type`` ``exception`` (type in ``:exception-type``, JSON
    ``{"message": ...}`` payload) or ``error`` (``:error-code`` /
    ``:error-message`` headers). Neither carries ``:event-type``. The error is
    raised as the ``UiPathAPIError`` subclass of the status AWS gives that
    exception, on a synthetic response whose body names the exception and its
    message, so callers see the provider failure instead of a parse error.
    """
    if headers.get(":message-type") == "exception":
        error_type = str(headers.get(":exception-type") or "unknownException")
        try:
            body = json.loads(payload.decode("utf-8"))
            message = (
                (body.get("message") or body.get("Message")) if isinstance(body, dict) else None
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            message = payload.decode("utf-8", errors="replace") or None
    else:
        error_type = str(headers.get(":error-code") or "unknownError")
        message = headers.get(":error-message")
    # ``error`` frames may spell the code ``ThrottlingException``; the table is camelCase.
    lookup_key = error_type[:1].lower() + error_type[1:]
    status_code = _STREAM_EXCEPTION_STATUS.get(lookup_key, _DEFAULT_STREAM_EXCEPTION_STATUS)
    response = Response(
        status_code,
        json={"type": error_type, "message": message},
        request=request,
    )
    error = UiPathAPIError.from_response(response, request)
    error.error_code = error_type
    error.detail = f"Bedrock stream {error_type}: {message}" if message else error_type
    return error


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
                    if event_as_dict["headers"].get(":message-type") in ("exception", "error"):
                        # The model failed after the 200: surface the provider error.
                        raise _stream_exception_error(
                            event_as_dict["headers"], event_as_dict["body"], response.request
                        )
                    dict_key = event_as_dict["headers"][":event-type"]
                    dict_value = json.loads(event_as_dict["body"].decode("utf-8"))
                    if dict_key == COST_METADATA_EVENT_TYPE:
                        # Gateway cost frame, already captured from the raw bytes by the
                        # httpx client; langchain-aws raises on unknown events.
                        continue
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
            "stream": self._stream_generator(
                {
                    "messages": messages,
                    "system": system,
                    **params,
                }
            ),
        }
