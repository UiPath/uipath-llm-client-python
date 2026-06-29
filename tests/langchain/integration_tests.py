"""Shared base class for the per-provider LangChain integration tests.

`UiPathChatModelIntegrationTests` consolidates the overrides that every
provider's `ChatModelIntegrationTests` subclass previously duplicated:

- `setup_models` autouse fixture (reads the provider-local `completions_config`)
- `chat_model_class` / `chat_model_params` properties
- `supports_*` property defaults (all `True`; override in a subclass to disable)
- The `test_stream` / `test_astream` overrides that parametrize on
  ``output_version`` and assert `chunk_position == "last"`
- The `test_parallel_and_sequential_tool_calling[_async]` methods, with a
  ``_bind_parallel_and_sequential`` hook that providers override to switch
  between the OpenAI dialect (``parallel_tool_calls``) and the Anthropic
  dialect (``disable_parallel_tool_use``)
- The new `test_file_inputs[_async]` matrix, parameterized over every format in
  `tests.langchain.file_fixtures.GENERATORS` (txt/md/csv/html, pdf, docx/xlsx,
  png/jpg/gif/webp). Each test asks the model for structured output and
  asserts the embedded invoice payload round-trips.

Subclasses should still declare a `skip_on_specific_configs` fixture for
provider-specific skips, and override `supports_*` properties or
`_bind_parallel_and_sequential` where the defaults don't apply.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import Runnable
from langchain_tests.integration_tests import ChatModelIntegrationTests

from tests.langchain.file_fixtures import (
    CUSTOMER,
    GENERATORS,
    INVOICE_NUMBER,
    TOTAL_AMOUNT,
    InvoiceInfo,
    build_human_content_block,
    load_fixture,
)
from tests.langchain.utils import search_accommodation, search_attractions, search_flights

_PARALLEL_TOOL_PROMPT = (
    "I want to plan a trip to Paris from New York. "
    "I need to find flights for March 15th, accommodation from March 15th to March 20th, "
    "and things to do there.",
    "Search for accomodations, flights and attractions in parallel. "
    "Don't repeat the same tool call.",
)

_FILE_INPUT_FORMATS = sorted(GENERATORS.keys())


def _assert_invoice(result: Any) -> None:
    assert isinstance(result, InvoiceInfo), f"Unexpected result type: {type(result)!r}"
    assert result.invoice_number.upper().replace(" ", "") == INVOICE_NUMBER, (
        f"Got invoice_number={result.invoice_number!r}, expected {INVOICE_NUMBER!r}"
    )
    assert CUSTOMER.lower() in result.customer.lower(), (
        f"Got customer={result.customer!r}, expected to contain {CUSTOMER!r}"
    )
    assert abs(result.total_amount - TOTAL_AMOUNT) < 0.01, (
        f"Got total_amount={result.total_amount!r}, expected ≈ {TOTAL_AMOUNT}"
    )


class UiPathChatModelIntegrationTests(ChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, completions_config: tuple[type[BaseChatModel], dict[str, Any]]) -> None:
        self._completions_class, self.completions_kwargs = completions_config

    @pytest.fixture(autouse=True)
    def skip_framework_irrelevant(self, request: pytest.FixtureRequest) -> None:
        if request.node.originalname in (
            "test_no_overrides_DO_NOT_OVERRIDE",
            "test_unicode_tool_call_integration",
        ):
            pytest.skip(f"Skipping {request.node.originalname}: not relevant")

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self.completions_kwargs

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_pdf_tool_message(self) -> bool:
        return True

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    def test_stream(self, model: BaseChatModel) -> None:
        chunks: list[AIMessageChunk] = []
        full: AIMessageChunk | None = None
        for chunk in model.stream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            chunks.append(chunk)
            full = chunk if full is None else full + chunk
        assert len(chunks) > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        text_blocks = [block for block in full.content_blocks if block["type"] == "text"]
        assert len(text_blocks) == 1

        last_chunk = chunks[-1]
        assert last_chunk.chunk_position == "last", (
            f"Final chunk must have chunk_position='last', got {last_chunk.chunk_position!r}"
        )

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    async def test_astream(self, model: BaseChatModel) -> None:
        chunks: list[AIMessageChunk] = []
        full: AIMessageChunk | None = None
        async for chunk in model.astream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            chunks.append(chunk)
            full = chunk if full is None else full + chunk
        assert len(chunks) > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        text_blocks = [block for block in full.content_blocks if block["type"] == "text"]
        assert len(text_blocks) == 1

        last_chunk = chunks[-1]
        assert last_chunk.chunk_position == "last", (
            f"Final chunk must have chunk_position='last', got {last_chunk.chunk_position!r}"
        )

    def _bind_parallel_and_sequential(
        self, model: BaseChatModel, tools: Iterable[Any]
    ) -> tuple[Runnable, Runnable]:
        """Return ``(parallel_model, sequential_model)`` for the parallel test.

        Default is the OpenAI dialect (``parallel_tool_calls`` kwarg). Providers
        on the Anthropic dialect override to use ``disable_parallel_tool_use``
        inside ``tool_choice``.
        """
        tools_list = list(tools)
        return (
            model.bind_tools(tools_list, tool_choice="any", parallel_tool_calls=True),
            model.bind_tools(tools_list, tool_choice="any", parallel_tool_calls=False),
        )

    def test_parallel_and_sequential_tool_calling(self, model: BaseChatModel) -> None:
        tools = [search_accommodation, search_flights, search_attractions]
        parallel, sequential = self._bind_parallel_and_sequential(model, tools)

        parallel_response = parallel.invoke(_PARALLEL_TOOL_PROMPT)
        sequential_response = sequential.invoke(_PARALLEL_TOOL_PROMPT)

        assert parallel_response.tool_calls is not None
        assert sequential_response.tool_calls is not None
        assert len(parallel_response.tool_calls) == len(tools), (
            f"Expected multiple different tools to be called in parallel, "
            f"got: {parallel_response.tool_calls}"
        )
        assert len(sequential_response.tool_calls) == 1, (
            f"Expected only one tool to be called in sequential mode, "
            f"got: {sequential_response.tool_calls}"
        )

    async def test_parallel_and_sequential_tool_calling_async(self, model: BaseChatModel) -> None:
        tools = [search_accommodation, search_flights, search_attractions]
        parallel, sequential = self._bind_parallel_and_sequential(model, tools)

        parallel_response = await parallel.ainvoke(_PARALLEL_TOOL_PROMPT)
        sequential_response = await sequential.ainvoke(_PARALLEL_TOOL_PROMPT)

        assert parallel_response.tool_calls is not None
        assert sequential_response.tool_calls is not None
        assert len(parallel_response.tool_calls) == len(tools), (
            f"Expected multiple different tools to be called in parallel, "
            f"got: {parallel_response.tool_calls}"
        )
        assert len(sequential_response.tool_calls) == 1, (
            f"Expected only one tool to be called in sequential mode, "
            f"got: {sequential_response.tool_calls}"
        )

    def _build_file_input_structured_model(self) -> Runnable:
        model = self._completions_class(**self.completions_kwargs)
        return model.with_structured_output(InvoiceInfo)

    def _build_human_message(self, fmt: str) -> HumanMessage:
        # `HumanMessage.content` is typed against `list[str | dict[Unknown, Unknown]]`;
        # langchain happily accepts our concrete `list[dict[str, Any]]` at runtime
        # but pyright can't widen across the invariant `list`, so cast.
        blocks = cast(list[Any], build_human_content_block(fmt, load_fixture(fmt)))
        return HumanMessage(content=blocks)

    @pytest.mark.parametrize("fmt", _FILE_INPUT_FORMATS)
    def test_file_inputs(self, fmt: str) -> None:
        """Round-trip a generated ``fmt`` file through the model via structured output."""
        result = self._build_file_input_structured_model().invoke([self._build_human_message(fmt)])
        _assert_invoice(result)

    @pytest.mark.parametrize("fmt", _FILE_INPUT_FORMATS)
    async def test_file_inputs_async(self, fmt: str) -> None:
        """Async path mirrors `test_file_inputs` for every supported format."""
        result = await self._build_file_input_structured_model().ainvoke(
            [self._build_human_message(fmt)]
        )
        _assert_invoice(result)
