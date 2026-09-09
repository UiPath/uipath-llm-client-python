"""Tests for the opt-in per-request dollar cost helpers."""

from httpx import AsyncByteStream, Request, Response, SyncByteStream

from tests.aws_event_stream import bedrock_chunk, event_frame
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
    set_captured_dollar_cost,
)


def _json_response(content: bytes, content_type: str = "application/json") -> Response:
    return Response(200, headers={"content-type": content_type}, content=content)


class TestRequestsDollarCost:
    def test_true_header_opts_in(self):
        request = Request(
            "POST", "https://example.com", headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "true"}
        )
        assert requests_dollar_cost(request) is True

    def test_header_value_is_parsed_as_bool_like_the_gateway(self):
        request = Request(
            "POST", "https://example.com", headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: " True "}
        )
        assert requests_dollar_cost(request) is True

    def test_false_header_does_not_opt_in(self):
        request = Request(
            "POST", "https://example.com", headers={INCLUDE_ASSOCIATED_DOLLAR_COST_HEADER: "false"}
        )
        assert requests_dollar_cost(request) is False

    def test_missing_header_does_not_opt_in(self):
        assert requests_dollar_cost(Request("POST", "https://example.com")) is False


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

    def test_non_numeric_field_returns_none(self):
        response = _json_response(b'{"associated_dollar_cost": "n/a"}')
        assert extract_associated_dollar_cost(response) is None

    def test_boolean_field_returns_none(self):
        response = _json_response(b'{"associated_dollar_cost": true}')
        assert extract_associated_dollar_cost(response) is None

    def test_non_numeric_field_is_logged(self, caplog):
        """A present but unusable value must not look like "not priced"."""
        import logging

        response = _json_response(b'{"associated_dollar_cost": "n/a"}')
        with caplog.at_level(logging.WARNING, logger="uipath.llm_client.utils.dollar_cost"):
            assert extract_associated_dollar_cost(response) is None
        assert "associated_dollar_cost='n/a'" in caplog.text


class TestDollarCostFromBody:
    def test_reads_numeric_field(self):
        assert dollar_cost_from_body({"associated_dollar_cost": 3}) == 3.0

    def test_non_dict_returns_none(self):
        assert dollar_cost_from_body(["associated_dollar_cost"]) is None


class TestCapturedDollarCostContextVar:
    def test_default_is_none(self):
        assert get_captured_dollar_cost() is None

    def test_set_and_get(self):
        set_captured_dollar_cost(0.002145)
        assert get_captured_dollar_cost() == 0.002145


# ============================================================================
# Streaming capture
# ============================================================================

COST_FRAME = b'data: {"associated_dollar_cost": 0.002145}\n\n'
CONTENT_EVENTS = [
    b'data: {"id": "chatcmpl-1", "choices": [{"delta": {"content": "Hel"}}]}\n\n',
    b'data: {"id": "chatcmpl-1", "choices": [{"delta": {"content": "lo"}}]}\n\n',
]
DONE_EVENT = b"data: [DONE]\n\n"


class _CountingSyncStream(SyncByteStream):
    """Records how many chunks were pulled and whether close() was called."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.pulled = 0
        self.closed = False

    def __iter__(self):
        for chunk in self._chunks:
            self.pulled += 1
            yield chunk

    def close(self):
        self.closed = True


class _CountingAsyncStream(AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.pulled = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            self.pulled += 1
            yield chunk

    async def aclose(self):
        self.closed = True


def _read_until_done(stream: DollarCostSyncStream) -> None:
    """Consume like the openai SDK: stop as soon as the [DONE] event is seen."""
    for chunk in stream:
        if DONE_EVENT in chunk:
            break


def _sse_stream(inner: SyncByteStream) -> DollarCostSyncStream:
    return DollarCostSyncStream(inner, SseCostFrames())


class TestSseCostFrames:
    def test_cost_in_last_event(self):
        frames = SseCostFrames()
        frames.feed(b"".join([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME]))
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

    def test_done_is_not_a_cost(self):
        frames = SseCostFrames()
        frames.feed(DONE_EVENT)
        assert frames.ends_with_done()
        assert frames.dollar_cost() is None


class TestEventStreamCostFrames:
    def test_cost_in_last_frame(self):
        frames = EventStreamCostFrames()
        frames.feed(event_frame("messageStop", {"stopReason": "end_turn"}))
        frames.feed(event_frame("costMetadata", {"associated_dollar_cost": 0.002145}))
        assert frames.dollar_cost() == 0.002145
        assert frames.ends_with_done() is False

    def test_frame_split_across_chunks(self):
        frame = event_frame("costMetadata", {"associated_dollar_cost": 0.002145})
        frames = EventStreamCostFrames()
        for i in range(0, len(frame), 7):
            frames.feed(frame[i : i + 7])
        assert frames.dollar_cost() == 0.002145

    def test_two_frames_in_one_chunk(self):
        frames = EventStreamCostFrames()
        frames.feed(
            event_frame("messageStop", {})
            + event_frame("costMetadata", {"associated_dollar_cost": 0.002145})
        )
        assert frames.dollar_cost() == 0.002145

    def test_last_frame_not_cost_reads_none(self):
        frames = EventStreamCostFrames()
        frames.feed(event_frame("costMetadata", {"associated_dollar_cost": 0.002145}))
        frames.feed(event_frame("messageStop", {"associated_dollar_cost": 0.002145}))
        assert frames.dollar_cost() is None

    def test_converse_metadata_event_is_terminal(self):
        frames = EventStreamCostFrames()
        frames.feed(event_frame("metadata", {"usage": {}}))
        assert frames.ends_with_done() is True

    def test_invoke_chunk_with_invocation_metrics_is_terminal(self):
        frames = EventStreamCostFrames()
        frames.feed(bedrock_chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}))
        assert frames.ends_with_done() is True

    def test_ordinary_frames_are_not_terminal(self):
        frames = EventStreamCostFrames()
        frames.feed(bedrock_chunk({"type": "content_block_delta"}))
        assert frames.ends_with_done() is False
        frames.feed(event_frame("chunk", {"bytes": "not base64!"}))
        assert frames.ends_with_done() is False

    def test_lost_framing_stops_buffering_and_warns(self, caplog):
        import logging

        frames = EventStreamCostFrames()
        with caplog.at_level(logging.WARNING, logger="uipath.llm_client.utils.dollar_cost"):
            frames.feed(b"\xff\xff\xff\xff" + b"\x00" * 12)
        frames.feed(event_frame("costMetadata", {"associated_dollar_cost": 0.002145}))
        assert frames.dollar_cost() is None
        assert frames._buffer == b""
        assert "Lost AWS event-stream framing" in caplog.text


class TestDollarCostSyncStream:
    def test_full_read_captures_trailing_cost_frame(self):
        inner = _CountingSyncStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        assert b"".join(_sse_stream(inner)) == b"".join([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        assert get_captured_dollar_cost() == 0.002145

    def test_bytes_pass_through_unchanged(self):
        chunks = [*CONTENT_EVENTS, DONE_EVENT, COST_FRAME]
        assert list(_sse_stream(_CountingSyncStream(chunks))) == chunks

    def test_close_after_done_drains_the_cost_frame(self):
        """The openai SDK stops at [DONE] and closes without reading the frame after it."""
        inner = _CountingSyncStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = _sse_stream(inner)
        _read_until_done(stream)
        assert get_captured_dollar_cost() is None
        stream.close()
        assert get_captured_dollar_cost() == 0.002145
        assert inner.pulled == 4
        assert inner.closed

    def test_close_after_done_without_cost_frame_reads_none(self):
        inner = _CountingSyncStream([*CONTENT_EVENTS, DONE_EVENT])
        stream = _sse_stream(inner)
        _read_until_done(stream)
        stream.close()
        assert get_captured_dollar_cost() is None
        assert inner.closed

    def test_early_close_mid_stream_does_not_drain(self):
        """Cancelling mid-stream must not keep downloading the model's output."""
        inner = _CountingSyncStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = _sse_stream(inner)
        next(iter(stream))
        stream.close()
        assert inner.pulled == 1
        assert get_captured_dollar_cost() is None
        assert inner.closed

    def test_cost_frame_split_across_chunks(self):
        inner = _CountingSyncStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME[:10], COST_FRAME[10:]])
        list(_sse_stream(inner))
        assert get_captured_dollar_cost() == 0.002145

    def test_unpriced_stream_reads_none(self):
        list(_sse_stream(_CountingSyncStream([*CONTENT_EVENTS, DONE_EVENT])))
        assert get_captured_dollar_cost() is None

    def test_close_without_reading_only_closes_inner(self):
        inner = _CountingSyncStream([COST_FRAME])
        _sse_stream(inner).close()
        assert inner.pulled == 0
        assert inner.closed
        assert get_captured_dollar_cost() is None

    def test_drain_failure_still_closes_inner(self):
        class _Failing(_CountingSyncStream):
            def __iter__(self):
                yield DONE_EVENT
                raise ConnectionError("boom")

        inner = _Failing([])
        stream = _sse_stream(inner)
        _read_until_done(stream)
        stream.close()
        assert inner.closed
        assert get_captured_dollar_cost() is None

    def test_event_stream_full_read_captures_cost(self):
        chunks = [
            event_frame("messageStop", {"stopReason": "end_turn"}),
            event_frame("costMetadata", {"associated_dollar_cost": 0.002145}),
        ]
        inner = _CountingSyncStream(chunks)
        assert list(DollarCostSyncStream(inner, EventStreamCostFrames())) == chunks
        assert get_captured_dollar_cost() == 0.002145

    def test_event_stream_close_after_terminal_frame_drains_the_cost_frame(self):
        """langchain-aws's invoke adapter stops at the vendor's stop event and closes."""
        stop = bedrock_chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {}})
        inner = _CountingSyncStream(
            [stop, event_frame("costMetadata", {"associated_dollar_cost": 0.002145})]
        )
        stream = DollarCostSyncStream(inner, EventStreamCostFrames())
        next(iter(stream))
        stream.close()
        assert get_captured_dollar_cost() == 0.002145
        assert inner.pulled == 2
        assert inner.closed


class TestDollarCostAsyncStream:
    async def test_full_read_captures_trailing_cost_frame(self):
        inner = _CountingAsyncStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        chunks = [chunk async for chunk in DollarCostAsyncStream(inner, SseCostFrames())]
        assert chunks == [*CONTENT_EVENTS, DONE_EVENT, COST_FRAME]
        assert get_captured_dollar_cost() == 0.002145

    async def test_close_after_done_drains_the_cost_frame(self):
        inner = _CountingAsyncStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = DollarCostAsyncStream(inner, SseCostFrames())
        async for chunk in stream:
            if DONE_EVENT in chunk:
                break
        await stream.aclose()
        assert get_captured_dollar_cost() == 0.002145
        assert inner.pulled == 4
        assert inner.closed

    async def test_early_close_mid_stream_does_not_drain(self):
        inner = _CountingAsyncStream([*CONTENT_EVENTS, DONE_EVENT, COST_FRAME])
        stream = DollarCostAsyncStream(inner, SseCostFrames())
        async for _ in stream:
            break
        await stream.aclose()
        assert inner.pulled == 1
        assert get_captured_dollar_cost() is None

    async def test_close_without_reading_only_closes_inner(self):
        inner = _CountingAsyncStream([COST_FRAME])
        await DollarCostAsyncStream(inner, SseCostFrames()).aclose()
        assert inner.pulled == 0
        assert inner.closed
        assert get_captured_dollar_cost() is None

    async def test_drain_failure_still_closes_inner(self):
        class _Failing(_CountingAsyncStream):
            async def __aiter__(self):
                yield DONE_EVENT
                raise ConnectionError("boom")

        inner = _Failing([])
        stream = DollarCostAsyncStream(inner, SseCostFrames())
        async for chunk in stream:
            if DONE_EVENT in chunk:
                break
        await stream.aclose()
        assert inner.closed
        assert get_captured_dollar_cost() is None

    async def test_event_stream_full_read_captures_cost(self):
        chunks = [event_frame("costMetadata", {"associated_dollar_cost": 0.002145})]
        inner = _CountingAsyncStream(chunks)
        assert [c async for c in DollarCostAsyncStream(inner, EventStreamCostFrames())] == chunks
        assert get_captured_dollar_cost() == 0.002145


class TestAttachStreamingDollarCostCapture:
    def test_wraps_sync_sse_stream(self):
        response = Response(200, headers={"content-type": "text/event-stream; charset=utf-8"})
        response.stream = _CountingSyncStream([])
        attach_streaming_dollar_cost_capture(response)
        assert isinstance(response.stream, DollarCostSyncStream)

    def test_wraps_async_sse_stream(self):
        response = Response(200, headers={"content-type": "text/event-stream"})
        response.stream = _CountingAsyncStream([])
        attach_streaming_dollar_cost_capture(response)
        assert isinstance(response.stream, DollarCostAsyncStream)

    def test_wraps_aws_event_stream(self):
        response = Response(200, headers={"content-type": "application/vnd.amazon.eventstream"})
        response.stream = _CountingSyncStream([])
        attach_streaming_dollar_cost_capture(response)
        assert isinstance(response.stream, DollarCostSyncStream)

    def test_leaves_other_content_types_alone(self):
        inner = _CountingSyncStream([])
        response = Response(200, headers={"content-type": "application/json"})
        response.stream = inner
        attach_streaming_dollar_cost_capture(response)
        assert response.stream is inner

    def test_leaves_compressed_streams_alone(self, caplog):
        """Compressed raw bytes cannot be parsed; warn instead of reading "not priced"."""
        import logging

        inner = _CountingSyncStream([])
        response = Response(
            200, headers={"content-type": "text/event-stream", "content-encoding": "gzip"}
        )
        response.stream = inner
        with caplog.at_level(logging.WARNING, logger="uipath.llm_client.utils.dollar_cost"):
            attach_streaming_dollar_cost_capture(response)
        assert response.stream is inner
        assert "gzip" in caplog.text
