"""A byte stream that delivers chunks lazily, like a network transport.

``httpx.Response(content=...)`` pre-buffers the body, so ``iter_bytes()`` never
touches ``response.stream``; tests of stream wrappers need the lazy shape.
"""

from collections.abc import AsyncIterator, Iterator

from httpx import AsyncByteStream, SyncByteStream


class LazyByteStream(SyncByteStream, AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.pulled = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self.pulled += 1
            yield chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self.pulled += 1
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True
