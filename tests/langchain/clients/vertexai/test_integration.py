"""LangChain integration tests for VertexAI provider client."""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests


@pytest.mark.asyncio
@pytest.mark.vcr
class TestVertexAIIntegrationChatModel(ChatModelIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, completions_config: tuple[type[BaseChatModel], dict[str, Any]]):
        self._completions_class, self.completions_kwargs = completions_config

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

    @pytest.fixture(autouse=True)
    def skip_on_specific_configs(
        self,
        request: pytest.FixtureRequest,
        completions_config: tuple[type[BaseChatModel], dict[str, Any]],
    ) -> None:
        model_class, model_kwargs = completions_config
        test_name = request.node.originalname
        has_thinking = "thinking" in model_kwargs

        # Useless framework tests
        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE", "test_unicode_tool_call_integration"]:
            pytest.skip(f"Skipping {test_name}: not relevant")

        # Claude + thinking: tool_choice forces tool use, incompatible with thinking
        if has_thinking and test_name in [
            "test_structured_few_shot_examples",
            "test_tool_calling",
            "test_tool_calling_async",
            "test_tool_calling_with_no_arguments",
            "test_tool_choice",
            "test_tool_message_error_status",
            "test_bind_runnables_as_tools",
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
            "test_parallel_and_sequential_tool_calling",
            "test_parallel_and_sequential_tool_calling_async",
        ]:
            pytest.skip(
                f"Skipping {test_name}: thinking may not be enabled when tool_choice forces tool use"
            )

        # Claude + thinking: extended thinking requires specific conversation history
        if has_thinking and test_name in [
            "test_tool_message_histories_list_content",
            "test_tool_message_histories_string_content",
        ]:
            pytest.skip(
                f"Skipping {test_name}: extended thinking requires a specific conversation history"
            )

        # UiPathChatAnthropicVertex: ls_structured_output_format not implemented
        if test_name in [
            "test_structured_output",
            "test_structured_output_async",
            "test_structured_output_optional_param",
            "test_structured_output_pydantic_2_v1",
        ]:
            pytest.skip(f"Skipping {test_name}: ls_structured_output not supported on this client")

        # UiPathChatAnthropicVertex: content_blocks not populated correctly
        if test_name in [
            "test_tool_calling",
        ]:
            pytest.skip(f"Skipping {test_name}: content_blocks not supported on this client")

        # UiPathChatAnthropicVertex: system message must be at beginning
        if test_name in [
            "test_double_messages_conversation",
        ]:
            pytest.skip(
                f"Skipping {test_name}: system message must be at beginning of message list"
            )

        # UiPathChatAnthropicVertex: agent_loop fails with non_standard content tag
        if test_name in [
            "test_agent_loop",
        ]:
            pytest.skip(f"Skipping {test_name}: fails due to non_standard content tag")

        # Claude via Vertex AI: streaming bugged (502 / empty content)
        if test_name in [
            "test_stream",
            "test_astream",
            "test_stream_time",
            "test_usage_metadata_streaming",
        ]:
            pytest.skip(f"Skipping {test_name}: currently bugged on Vertex AI")

        # Claude via Vertex: image URL sources not supported
        if test_name in [
            "test_image_inputs",
        ]:
            pytest.skip(f"Skipping {test_name}: URL image sources not supported via gateway")

        # Claude via Vertex: PDF/file inputs not supported
        if test_name in [
            "test_pdf_inputs",
        ]:
            pytest.skip(f"Skipping {test_name}: file blocks not supported on this client")

        # Image/pdf tool message not supported
        if test_name in [
            "test_image_tool_message",
            "test_pdf_tool_message",
        ]:
            pytest.skip(f"Skipping {test_name}: not supported on this client")

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self.completions_kwargs

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
