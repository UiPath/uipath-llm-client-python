"""Opt-in per-request dollar cost reported by the LLM Gateway.

The gateway prices a call only on the Passthrough API and only when the request
carries ``X-UiPath-LlmGateway-IncludeAssociatedDollarCost: true``. Non-streaming
JSON responses get a top-level ``associated_dollar_cost`` field. Streaming
responses get a trailing frame after the vendor's terminal event: an SSE
``data: {"associated_dollar_cost": n}`` event, or an AWS event-stream message
with ``:event-type=costMetadata``.

A missing value means "not priced", never $0.
"""

import base64
import binascii
import contextvars
import json
import logging
import re
from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from httpx import AsyncByteStream, Request, Response, SyncByteStream

logger = logging.getLogger(__name__)

INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER = "X-UiPath-LlmGateway-IncludeAssociatedDollarCost"
ASSOCIATED_DOLLAR_COST_FIELD = "associated_dollar_cost"
COST_METADATA_EVENT_TYPE = "costMetadata"

_CAPTURED_DOLLAR_COST: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "_captured_dollar_cost", default=None
)


def get_captured_dollar_cost() -> float | None:
    """Dollar cost captured from the most recent response in this context, or None if it was not priced."""
    return _CAPTURED_DOLLAR_COST.get()


def set_captured_dollar_cost(cost: float | None) -> contextvars.Token[float | None]:
    """Set the captured per-request dollar cost for the current context."""
    return _CAPTURED_DOLLAR_COST.set(cost)


def requests_dollar_cost(request: Request) -> bool:
    """Whether the request opted in; same bool parse as the gateway, so we only parse bodies it could have priced."""
    return request.headers.get(INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER, "").strip().lower() == "true"


def dollar_cost_from_body(body: object) -> float | None:
    """Read ``associated_dollar_cost`` from a decoded JSON payload, or None if absent."""
    if not isinstance(body, dict) or ASSOCIATED_DOLLAR_COST_FIELD not in body:
        return None
    cost = body[ASSOCIATED_DOLLAR_COST_FIELD]  # pyright: ignore[reportUnknownVariableType]
    # bool is excluded: True would otherwise read as $1.
    if isinstance(cost, bool) or not isinstance(cost, int | float):
        # Unlike absence ("not priced"), a present but unusable value is contract drift.
        logger.warning(
            "Ignoring non-numeric %s=%r in gateway response", ASSOCIATED_DOLLAR_COST_FIELD, cost
        )
        return None
    return float(cost)


def extract_associated_dollar_cost(response: Response) -> float | None:
    """Read ``associated_dollar_cost`` from a buffered JSON response body, or None if absent."""
    if "application/json" not in response.headers.get("content-type", ""):
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    return dollar_cost_from_body(body)


def _dollar_cost_from_json_bytes(payload: bytes | str) -> float | None:
    try:
        body = json.loads(payload)
    except ValueError:
        return None
    return dollar_cost_from_body(body)


# --- Streaming ---------------------------------------------------------------------


class CostFrames(Protocol):
    """Tracks the last complete frame of one wire format."""

    def feed(self, chunk: bytes) -> None: ...

    def ends_with_done(self) -> bool:
        """Whether the last frame is the vendor's terminal one, where some SDKs stop reading."""
        ...

    def dollar_cost(self) -> float | None:
        """The cost if the last frame is the gateway's cost frame, else None."""
        ...


# The cost frame is ~50 bytes and always last, so a small tail is enough.
_SSE_TAIL_BYTES = 4096
_SSE_EVENT_SEPARATOR = re.compile(rb"\r?\n\r?\n")
# The openai SDK stops reading here and closes, leaving the gateway's frame unread.
_SSE_DONE_DATA = "[DONE]"


def _sse_event_data(event: bytes) -> str | None:
    lines = [
        line[5:].removeprefix(" ")
        for line in event.decode("utf-8", errors="replace").splitlines()
        if line.startswith("data:")
    ]
    return "\n".join(lines) if lines else None


class SseCostFrames:
    """``text/event-stream``: the cost is a trailing ``data: {...}`` event."""

    def __init__(self) -> None:
        self._tail = b""

    def feed(self, chunk: bytes) -> None:
        self._tail = (self._tail + chunk)[-_SSE_TAIL_BYTES:]

    def _last_complete_event_data(self) -> str | None:
        # Ignore an unterminated remainder (no closing blank line yet).
        events = [e for e in _SSE_EVENT_SEPARATOR.split(self._tail)[:-1] if e.strip()]
        return _sse_event_data(events[-1]) if events else None

    def ends_with_done(self) -> bool:
        return self._last_complete_event_data() == _SSE_DONE_DATA

    def dollar_cost(self) -> float | None:
        data = self._last_complete_event_data()
        return None if data is None else _dollar_cost_from_json_bytes(data)


# AWS event-stream frame: 12-byte prelude (total length, headers length, CRC),
# headers, payload, 4-byte CRC. Headers are matched as their exact encoded bytes
# (name length, name, type 7 = string, value length, value) instead of being parsed.
_EVENTSTREAM_PRELUDE_BYTES = 12
_EVENTSTREAM_MIN_FRAME_BYTES = _EVENTSTREAM_PRELUDE_BYTES + 4
# Above the AWS maximum we have lost framing; stop buffering rather than grow.
_EVENTSTREAM_MAX_FRAME_BYTES = 16 * 1024 * 1024


def _event_type_header(event_type: str) -> bytes:
    return (
        bytes([len(b":event-type")])
        + b":event-type"
        + b"\x07"
        + len(event_type).to_bytes(2, "big")
        + event_type.encode()
    )


_EVENTSTREAM_COST_HEADER = _event_type_header(COST_METADATA_EVENT_TYPE)
# Bedrock's terminal markers: converse ends with a `metadata` event; invoke tags the
# vendor's last chunk with invocation metrics.
_EVENTSTREAM_METADATA_HEADER = _event_type_header("metadata")
_BEDROCK_INVOCATION_METRICS = b"amazon-bedrock-invocationMetrics"


class EventStreamCostFrames:
    """``application/vnd.amazon.eventstream``: the cost is a trailing ``costMetadata`` message."""

    def __init__(self) -> None:
        self._buffer = b""
        self._last_frame = b""
        self._broken = False

    def feed(self, chunk: bytes) -> None:
        if self._broken:
            return
        self._buffer += chunk
        while len(self._buffer) >= _EVENTSTREAM_PRELUDE_BYTES:
            total = int.from_bytes(self._buffer[:4], "big")
            if total < _EVENTSTREAM_MIN_FRAME_BYTES or total > _EVENTSTREAM_MAX_FRAME_BYTES:
                logger.warning(
                    "Lost AWS event-stream framing (frame length %d); dollar cost not tracked",
                    total,
                )
                self._broken = True
                self._buffer = b""
                return
            if len(self._buffer) < total:
                return
            self._last_frame, self._buffer = self._buffer[:total], self._buffer[total:]

    def _last_frame_parts(self) -> tuple[bytes, bytes] | None:
        frame = self._last_frame
        if len(frame) < _EVENTSTREAM_MIN_FRAME_BYTES:
            return None
        headers_end = _EVENTSTREAM_PRELUDE_BYTES + int.from_bytes(frame[4:8], "big")
        return frame[_EVENTSTREAM_PRELUDE_BYTES:headers_end], frame[headers_end:-4]

    def ends_with_done(self) -> bool:
        """langchain-aws's invoke adapter stops at the vendor's stop event and closes without reading to EOF."""
        parts = self._last_frame_parts()
        if parts is None:
            return False
        headers, payload = parts
        if _EVENTSTREAM_METADATA_HEADER in headers:
            return True
        # invoke wraps the vendor JSON as base64 under "bytes".
        try:
            body = json.loads(payload)
            raw = body.get("bytes") if isinstance(body, dict) else None
            decoded = base64.b64decode(raw) if isinstance(raw, str) else b""
        except (ValueError, binascii.Error):
            return False
        return _BEDROCK_INVOCATION_METRICS in decoded

    def dollar_cost(self) -> float | None:
        parts = self._last_frame_parts()
        if parts is None:
            return None
        headers, payload = parts
        if _EVENTSTREAM_COST_HEADER not in headers:
            return None
        return _dollar_cost_from_json_bytes(payload)


# Safety net for the drain in close(): only the cost frame should follow the terminal one.
_DRAIN_LIMIT_BYTES = 64 * 1024


class DollarCostSyncStream(SyncByteStream):
    """Pass-through byte stream that captures the gateway's trailing cost frame.

    Recorded on exhaustion or on ``close()``. Some consumers (openai SDK at
    ``[DONE]``, langchain-aws at the vendor's stop event) close without reading
    further, so ``close()`` first drains what follows the terminal frame. A close
    mid-stream drains nothing: the frame cannot have arrived yet.

    The drain is bounded in bytes only; httpcore fixes the read timeout before the
    body loop, so it inherits the client's. The gateway ends the response right
    after the frame, and the anthropic/google SDKs already read to EOF anyway.
    """

    def __init__(self, inner: SyncByteStream, frames: CostFrames) -> None:
        self._inner = inner
        self._frames = frames
        self._iterator: Iterator[bytes] | None = None
        self._done = False

    def __iter__(self) -> Iterator[bytes]:
        self._iterator = iter(self._inner)
        for chunk in self._iterator:
            self._frames.feed(chunk)
            yield chunk
        self._finish()

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            set_captured_dollar_cost(self._frames.dollar_cost())

    def close(self) -> None:
        try:
            if self._iterator is not None and not self._done and self._frames.ends_with_done():
                drained = 0
                for chunk in self._iterator:
                    self._frames.feed(chunk)
                    drained += len(chunk)
                    if drained > _DRAIN_LIMIT_BYTES:
                        break
        except Exception:  # noqa: BLE001 - cost capture must never break closing the response
            logger.debug("Failed to drain the trailing cost frame", exc_info=True)
        finally:
            if self._iterator is not None:
                self._finish()
            self._inner.close()


class DollarCostAsyncStream(AsyncByteStream):
    """Async counterpart of :class:`DollarCostSyncStream`."""

    def __init__(self, inner: AsyncByteStream, frames: CostFrames) -> None:
        self._inner = inner
        self._frames = frames
        self._iterator: AsyncIterator[bytes] | None = None
        self._done = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._iterator = self._inner.__aiter__()
        async for chunk in self._iterator:
            self._frames.feed(chunk)
            yield chunk
        self._finish()

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            set_captured_dollar_cost(self._frames.dollar_cost())

    async def aclose(self) -> None:
        try:
            if self._iterator is not None and not self._done and self._frames.ends_with_done():
                drained = 0
                async for chunk in self._iterator:
                    self._frames.feed(chunk)
                    drained += len(chunk)
                    if drained > _DRAIN_LIMIT_BYTES:
                        break
        except Exception:  # noqa: BLE001 - cost capture must never break closing the response
            logger.debug("Failed to drain the trailing cost frame", exc_info=True)
        finally:
            if self._iterator is not None:
                self._finish()
            await self._inner.aclose()


def _cost_frames_for(response: Response) -> CostFrames | None:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return SseCostFrames()
    if "application/vnd.amazon.eventstream" in content_type:
        return EventStreamCostFrames()
    return None


def attach_streaming_dollar_cost_capture(response: Response) -> None:
    """Wrap a streaming response so its trailing cost frame is captured as it is consumed."""
    frames = _cost_frames_for(response)
    if frames is None:
        return
    # The wrapper sees raw bytes; a compressed stream would silently read as "not priced".
    encoding = response.headers.get("content-encoding", "identity").lower()
    if encoding not in ("", "identity"):
        logger.warning("Not capturing dollar cost from a %s-encoded stream", encoding)
        return
    if isinstance(response.stream, SyncByteStream):
        response.stream = DollarCostSyncStream(response.stream, frames)
    elif isinstance(response.stream, AsyncByteStream):
        response.stream = DollarCostAsyncStream(response.stream, frames)
