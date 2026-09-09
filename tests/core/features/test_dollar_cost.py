"""Tests for the opt-in per-request dollar cost helpers."""

import logging

from httpx import AsyncByteStream, Request, Response, SyncByteStream

from tests.aws_event_stream import bedrock_chunk, event_frame
from tests.lazy_stream import LazyByteStream
from uipath.llm_client.utils.dollar_cost import (
    INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER,
    DollarCostAsyncStream,
    DollarCostSyncStream,
    EventStreamCostFrames,
    SseCostFrames,
    attach_streaming_dollar_cost_capture,
    dollar_cost_from_body,
    extract_associated_dollar_cost,
    get_captured_dollar_cost,
    requests_dollar_cost,
)

LOGGER = "uipath.llm_client.utils.dollar_cost"


def _json_response(content: bytes, content_type: str = "application/json") -> Response:
    return Response(200, headers={"content-type": content_type}, content=content)


def _request(header_value: str | None) -> Request:
    headers = {} if header_value is None else {INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: header_value}
    return Request("POST", "https://example.com", headers=headers)


class TestRequestsDollarCost:
    def test_header_value_is_parsed_as_bool_like_the_gateway(self):
        assert requests_dollar_cost(_request(" True ")) is True

    def test_false_header_does_not_opt_in(self):
        assert requests_dollar_cost(_request("false")) is False

    def test_missing_header_does_not_opt_in(self):
        assert requests_dollar_cost(_request(None)) is False


class TestExtractAssociatedDollarCost:
    def test_extracts_cost_from_json_body(self):
        response = _json_response(b'{"choices": [], "associated_dollar_cost": 0.002145}')
        assert extract_associated_dollar_cost(response) == 0.002145

    def test_absent_field_returns_none(self):
        """Absence must read as 'not priced', never $0."""
        assert extract_associated_dollar_cost(_json_response(b'{"choices": []}')) is None

    def test_non_json_content_type_returns_none(self):
        response = _json_response(
            b'data: {"associated_dollar_cost": 0.002145}\n', content_type="text/event-stream"
        )
        assert extract_associated_dollar_cost(response) is None

    def test_malformed_json_returns_none(self):
        assert extract_associated_dollar_cost(_json_response(b"not json")) is None

    def test_non_object_json_returns_none(self):
        assert extract_associated_dollar_cost(_json_response(b"[1, 2]")) is None

    def test_boolean_field_returns_none(self):
        response = _json_response(b'{"associated_dollar_cost": true}')
        assert extract_associated_dollar_cost(response) is None

    def test_non_numeric_field_returns_none_and_warns(self, caplog):
        """A present but unusable value must not look like "not priced"."""
        response = _json_response(b'{"associated_dollar_cost": "n/a"}')
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert extract_associated_dollar_cost(response) is None
        assert "associated_dollar_cost='n/a'" in caplog.text

    def test_integer_cost_reads_as_float(self):
        assert dollar_cost_from_body({"associated_dollar_cost": 3}) == 3.0


# ============================================================================
# Streaming capture
# ============================================================================

COST_FRAME = b'data: {"associated_dollar_cost": 0.002145}\n\n'
CONTENT_EVENTS = [
    b'data: {"id": "chatcmpl-1", "choices": [{"delta": {"content": "Hel"}}]}\n\n',
    b'data: {"id": "chatcmpl-1", "choices": [{"delta": {"content": "lo"}}]}\n\n',
]
DONE_EVENT = b"data: [DONE]\n\n"
EVENTSTREAM_COST_FRAME = event_frame("costMetadata", {"associated_dollar_cost": 0.002145})
EVENTSTREAM_STOP_FRAME = bedrock_chunk(
    {"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}
)


def _read_until_done(stream: DollarCostSyncStream) -> None:
    """Consume like the openai SDK: stop as soon as the [DONE] event is seen."""
    for chunk in stream:
        if DONE_EVENT in chunk:
            break


def _sse_stream(inner: SyncByteStream) -> DollarCostSyncStream:
    return DollarCostSyncStream(inner, SseCostFrames())


def _sse_astream(inner: AsyncByteStream) -> DollarCostAsyncStream:
    return DollarCostAsyncStream(inner, SseCostFrames())


class TestSseCostFrames:
    def test_cost_in_last_event(self):
        frames = SseCostFrames()
        frames.feed(b"".join([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME]))
        assert frames.dollar_cost() == 0.002145

    def test_cost_frame_split_across_feeds(self):
        frames = SseCostFrames()
        for chunk in [*CONTENT_EVENTS, DONE_EVENT, COST_FRAME[:10], COST_FRAME[10:]]:
            frames.feed(chunk)
        assert frames.dollar_cost() == 0.002145

    def test_crlf_and_multi_line_data(self):
        frames = SseCostFrames()
        frames.feed(b'data: {"associated_dollar_cost":\r\ndata:  0.002145}\r\n\r\n')
        assert frames.dollar_cost() == 0.002145

    def test_unterminated_trailing_event_is_ignored(self):
        frames = SseCostFrames()
        frames.feed(DONE_EVENT + COST_FRAME.rstrip(b"\n"))
        assert frames.ends_with_done()
        assert frames.dollar_cost() is None

    def test_event_and_comment_lines_are_skipped(self):
        frames = SseCostFrames()
        frames.feed(b': keep-alive\nevent: cost\ndata: {"associated_dollar_cost": 1}\n\n')
        assert frames.dollar_cost() == 1.0

    def test_done_is_terminal_and_not_a_cost(self):
        frames = SseCostFrames()
        frames.feed(DONE_EVENT)
        assert frames.ends_with_done()
        assert frames.dollar_cost() is None


class TestEventStreamCostFrames:
    def test_cost_in_last_frame(self):
        frames = EventStreamCostFrames()
        frames.feed(event_frame("messageStop", {"stopReason": "end_turn"}))
        frames.feed(EVENTSTREAM_COST_FRAME)
        assert frames.dollar_cost() == 0.002145
        assert frames.ends_with_done() is False

    def test_frame_split_across_chunks(self):
        frames = EventStreamCostFrames()
        for i in range(0, len(EVENTSTREAM_COST_FRAME), 7):
            frames.feed(EVENTSTREAM_COST_FRAME[i : i + 7])
        assert frames.dollar_cost() == 0.002145

    def test_two_frames_in_one_chunk(self):
        frames = EventStreamCostFrames()
        frames.feed(event_frame("messageStop", {}) + EVENTSTREAM_COST_FRAME)
        assert frames.dollar_cost() == 0.002145

    def test_last_frame_not_cost_reads_none(self):
        frames = EventStreamCostFrames()
        frames.feed(EVENTSTREAM_COST_FRAME)
        frames.feed(event_frame("messageStop", {"associated_dollar_cost": 0.002145}))
        assert frames.dollar_cost() is None

    def test_converse_metadata_event_is_terminal(self):
        frames = EventStreamCostFrames()
        frames.feed(event_frame("metadata", {"usage": {}}))
        assert frames.ends_with_done() is True

    def test_invoke_chunk_with_invocation_metrics_is_terminal(self):
        frames = EventStreamCostFrames()
        frames.feed(EVENTSTREAM_STOP_FRAME)
        assert frames.ends_with_done() is True

    def test_ordinary_frames_are_not_terminal(self):
        frames = EventStreamCostFrames()
        frames.feed(bedrock_chunk({"type": "content_block_delta"}))
        assert frames.ends_with_done() is False
        frames.feed(event_frame("chunk", {"bytes": "not base64!"}))
        assert frames.ends_with_done() is False

    def test_lost_framing_stops_tracking_and_warns(self, caplog):
        frames = EventStreamCostFrames()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            frames.feed(b"\xff\xff\xff\xff" + b"\x00" * 12)
        frames.feed(EVENTSTREAM_COST_FRAME)
        assert frames.dollar_cost() is None
        assert "Lost AWS event-stream framing" in caplog.text


class TestDollarCostSyncStream:
    def test_full_read_captures_cost_and_passes_chunks_through(self):
        chunks = [*CONTENT_EVENTS, DONE_EVENT, COST_FRAME]
        assert list(_sse_stream(LazyByteStream(chunks))) == chunks
        assert get_captured_dollar_cost() == 0.002145

    def test_close_after_done_drains_the_cost_frame(self):
        """The openai SDK stops at [DONE] and closes without reading the frame after it."""
        inner = LazyByteStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = _sse_stream(inner)
        _read_until_done(stream)
        assert get_captured_dollar_cost() is None
        stream.close()
        assert get_captured_dollar_cost() == 0.002145
        assert inner.pulled == 4
        assert inner.closed

    def test_close_after_done_without_cost_frame_reads_none(self):
        inner = LazyByteStream([*CONTENT_EVENTS, DONE_EVENT])
        stream = _sse_stream(inner)
        _read_until_done(stream)
        stream.close()
        assert get_captured_dollar_cost() is None
        assert inner.closed

    def test_drain_is_bounded(self):
        """Only the cost frame should follow the terminal one; anything else must not be downloaded whole."""
        inner = LazyByteStream([DONE_EVENT, *([b"x" * 1024] * 200)])
        stream = _sse_stream(inner)
        _read_until_done(stream)
        stream.close()
        assert 64 <= inner.pulled - 1 <= 66
        assert get_captured_dollar_cost() is None
        assert inner.closed

    def test_early_close_mid_stream_does_not_drain(self):
        """Cancelling mid-stream must not keep downloading the model's output."""
        inner = LazyByteStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = _sse_stream(inner)
        next(iter(stream))
        stream.close()
        assert inner.pulled == 1
        assert get_captured_dollar_cost() is None
        assert inner.closed

    def test_close_without_reading_only_closes_inner(self):
        inner = LazyByteStream([COST_FRAME])
        _sse_stream(inner).close()
        assert inner.pulled == 0
        assert inner.closed
        assert get_captured_dollar_cost() is None

    def test_drain_failure_still_closes_inner(self):
        class _Failing(LazyByteStream):
            def __iter__(self):
                yield DONE_EVENT
                raise ConnectionError("boom")

        inner = _Failing([])
        stream = _sse_stream(inner)
        _read_until_done(stream)
        stream.close()
        assert inner.closed
        assert get_captured_dollar_cost() is None

    def test_event_stream_close_after_terminal_frame_drains_the_cost_frame(self):
        """langchain-aws stops at the vendor's stop event and closes."""
        inner = LazyByteStream([EVENTSTREAM_STOP_FRAME, EVENTSTREAM_COST_FRAME])
        stream = DollarCostSyncStream(inner, EventStreamCostFrames())
        next(iter(stream))
        stream.close()
        assert get_captured_dollar_cost() == 0.002145
        assert inner.pulled == 2
        assert inner.closed


class TestDollarCostAsyncStream:
    async def test_full_read_captures_cost_and_passes_chunks_through(self):
        chunks = [*CONTENT_EVENTS, DONE_EVENT, COST_FRAME]
        assert [c async for c in _sse_astream(LazyByteStream(chunks))] == chunks
        assert get_captured_dollar_cost() == 0.002145

    async def test_close_after_done_drains_the_cost_frame(self):
        inner = LazyByteStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = _sse_astream(inner)
        async for chunk in stream:
            if DONE_EVENT in chunk:
                break
        await stream.aclose()
        assert get_captured_dollar_cost() == 0.002145
        assert inner.pulled == 4
        assert inner.closed

    async def test_drain_is_bounded(self):
        inner = LazyByteStream([DONE_EVENT, *([b"x" * 1024] * 200)])
        stream = _sse_astream(inner)
        async for _ in stream:
            break
        await stream.aclose()
        assert 64 <= inner.pulled - 1 <= 66
        assert get_captured_dollar_cost() is None

    async def test_early_close_mid_stream_does_not_drain(self):
        inner = LazyByteStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = _sse_astream(inner)
        async for _ in stream:
            break
        await stream.aclose()
        assert inner.pulled == 1
        assert get_captured_dollar_cost() is None

    async def test_close_without_reading_only_closes_inner(self):
        inner = LazyByteStream([COST_FRAME])
        await _sse_astream(inner).aclose()
        assert inner.pulled == 0
        assert inner.closed
        assert get_captured_dollar_cost() is None

    async def test_drain_failure_still_closes_inner(self):
        class _Failing(LazyByteStream):
            async def __aiter__(self):
                yield DONE_EVENT
                raise ConnectionError("boom")

        inner = _Failing([])
        stream = _sse_astream(inner)
        async for chunk in stream:
            if DONE_EVENT in chunk:
                break
        await stream.aclose()
        assert inner.closed
        assert get_captured_dollar_cost() is None


class TestAttachStreamingDollarCostCapture:
    def test_sse_stream_captures_cost(self):
        response = Response(200, headers={"content-type": "text/event-stream; charset=utf-8"})
        response.stream = LazyByteStream([DONE_EVENT, COST_FRAME])
        attach_streaming_dollar_cost_capture(response)
        assert isinstance(response.stream, DollarCostSyncStream)
        list(response.stream)
        assert get_captured_dollar_cost() == 0.002145

    def test_async_sse_stream_is_wrapped(self):
        class _AsyncOnly(AsyncByteStream):
            async def __aiter__(self):
                yield b""

        response = Response(200, headers={"content-type": "text/event-stream"})
        response.stream = _AsyncOnly()
        attach_streaming_dollar_cost_capture(response)
        assert isinstance(response.stream, DollarCostAsyncStream)

    def test_aws_event_stream_captures_cost(self):
        response = Response(200, headers={"content-type": "application/vnd.amazon.eventstream"})
        response.stream = LazyByteStream([EVENTSTREAM_STOP_FRAME, EVENTSTREAM_COST_FRAME])
        attach_streaming_dollar_cost_capture(response)
        list(response.stream)
        assert get_captured_dollar_cost() == 0.002145

    def test_leaves_other_content_types_alone(self):
        inner = LazyByteStream([])
        response = Response(200, headers={"content-type": "application/json"})
        response.stream = inner
        attach_streaming_dollar_cost_capture(response)
        assert response.stream is inner

    def test_leaves_compressed_streams_alone_and_warns(self, caplog):
        """Compressed raw bytes cannot be parsed; warn instead of reading "not priced"."""
        inner = LazyByteStream([])
        response = Response(
            200, headers={"content-type": "text/event-stream", "content-encoding": "gzip"}
        )
        response.stream = inner
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            attach_streaming_dollar_cost_capture(response)
        assert response.stream is inner
        assert "gzip" in caplog.text
