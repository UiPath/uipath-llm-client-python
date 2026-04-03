"""Embeddings endpoint for the UiPath Normalized API.

Provides synchronous and asynchronous methods for generating text embeddings.
"""

from __future__ import annotations

from typing import Any

from uipath.llm_client.clients.normalized.types import (
    EmbeddingData,
    EmbeddingResponse,
    Usage,
)


def _parse_embedding_response(data: dict[str, Any]) -> EmbeddingResponse:
    """Parse an embedding response from the API."""
    usage_data = data.get("usage", {})
    embeddings = [
        EmbeddingData(
            embedding=item.get("embedding", []),
            index=item.get("index", i),
        )
        for i, item in enumerate(data.get("data", []))
    ]
    return EmbeddingResponse(
        data=embeddings,
        model=data.get("model", ""),
        usage=Usage(**usage_data),
    )


class Embeddings:
    """Embeddings namespace with ``create`` and ``acreate``.

    Handles request building and response parsing for the UiPath normalized
    embeddings API.

    Example:
        >>> response = client.embeddings.create(input=["Hello world"])
        >>> print(response.data[0].embedding[:5])
        >>>
        >>> response = await client.embeddings.acreate(input=["Hello world"])
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(
        self,
        *,
        input: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create embeddings (sync).

        Args:
            input: A string or list of strings to embed.
            **kwargs: Additional parameters for the API.

        Returns:
            EmbeddingResponse with embedding vectors.
        """
        if isinstance(input, str):
            input = [input]

        body: dict[str, Any] = {"input": input, **kwargs}
        response = self._client._embedding_sync_client.request("POST", "/", json=body)
        response.raise_for_status()
        return _parse_embedding_response(response.json())

    async def acreate(
        self,
        *,
        input: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create embeddings (async).

        Args:
            input: A string or list of strings to embed.
            **kwargs: Additional parameters for the API.

        Returns:
            EmbeddingResponse with embedding vectors.
        """
        if isinstance(input, str):
            input = [input]

        body: dict[str, Any] = {"input": input, **kwargs}
        response = await self._client._embedding_async_client.request("POST", "/", json=body)
        response.raise_for_status()
        return _parse_embedding_response(response.json())
