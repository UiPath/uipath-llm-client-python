"""Hand-encoded AWS event-stream frames for tests (botocore only ships a decoder)."""

import base64
import json
import struct
import zlib

import httpx

from tests.lazy_stream import LazyByteStream


def raw_frame(headers: dict[str, str], body: bytes) -> bytes:
    """Encode one AWS event-stream message (prelude + string headers + payload + CRCs)."""
    encoded_headers = b""
    for name, value in headers.items():
        name_bytes = name.encode()
        value_bytes = value.encode()
        encoded_headers += (
            bytes([len(name_bytes)])
            + name_bytes
            + bytes([7])
            + struct.pack(">H", len(value_bytes))
            + value_bytes
        )
    total_length = 12 + len(encoded_headers) + len(body) + 4
    prelude = struct.pack(">II", total_length, len(encoded_headers))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + encoded_headers + body
    return message + struct.pack(">I", zlib.crc32(message))


def event_frame(event_type: str, payload: dict) -> bytes:
    """Encode one AWS event-stream event message carrying ``:event-type``."""
    return raw_frame({":event-type": event_type}, json.dumps(payload).encode())


def exception_frame(exception_type: str, payload: dict) -> bytes:
    """Encode an in-stream exception message, as Bedrock sends when the model fails mid-stream."""
    return raw_frame(
        {
            ":message-type": "exception",
            ":exception-type": exception_type,
            ":content-type": "application/json",
        },
        json.dumps(payload).encode(),
    )


def error_frame(error_code: str, error_message: str) -> bytes:
    """Encode an in-stream ``error`` message (code and message in headers, empty payload)."""
    return raw_frame(
        {
            ":message-type": "error",
            ":error-code": error_code,
            ":error-message": error_message,
        },
        b"",
    )


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
