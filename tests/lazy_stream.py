"""A byte stream that delivers chunks lazily, like a network transport.

``httpx.Response(content=...)`` pre-buffers the body, so ``iter_bytes()`` never
touches ``response.stream``; tests of stream wrappers need the lazy shape.
"""

from collections.abc import AsyncIterator, Iterator

from httpx import AsyncByteStream, SyncByteStream


class LazyByteStream(SyncByteStream, AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass
