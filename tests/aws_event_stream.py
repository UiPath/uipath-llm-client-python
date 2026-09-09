"""Hand-encoded AWS event-stream frames for tests (botocore only ships a decoder)."""

import base64
import json
import struct
import zlib

import httpx

from tests.lazy_stream import LazyByteStream


def event_frame(event_type: str, payload: dict) -> bytes:
    """Encode one AWS event-stream message (prelude + string header + payload + CRCs)."""
    name = b":event-type"
    value = event_type.encode()
    headers = bytes([len(name)]) + name + bytes([7]) + struct.pack(">H", len(value)) + value
    body = json.dumps(payload).encode()
    total_length = 12 + len(headers) + len(body) + 4
    prelude = struct.pack(">II", total_length, len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


def bedrock_chunk(event: dict) -> bytes:
    """An invoke-with-response-stream ``chunk`` frame carrying one base64-encoded vendor event."""
    return event_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})


def event_stream_response(*frames: bytes) -> httpx.Response:
    """A lazily delivered event-stream response, one frame per chunk."""
    return httpx.Response(
        200,
        stream=LazyByteStream(list(frames)),
        headers={"content-type": "application/vnd.amazon.eventstream"},
    )
