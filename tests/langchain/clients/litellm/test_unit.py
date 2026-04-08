"""LangChain unit tests for LiteLLM provider clients.

Uses the langchain-tests unit test framework.
"""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests


@pytest.mark.vcr
class TestLiteLLMUnitChatModel(ChatModelUnitTests):
    @pytest.fixture(autouse=True)
    def setup_models(self, completions_config: tuple[type[BaseChatModel], dict[str, Any]]):
        self._completions_class, self.completions_kwargs = completions_config

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return self._completions_class

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return self.completions_kwargs

    @pytest.fixture(autouse=True)
    def skip_irrelevant(self, request: pytest.FixtureRequest) -> None:
        test_name = request.node.originalname
        if test_name in ["test_no_overrides_DO_NOT_OVERRIDE"]:
            pytest.skip(f"Skipping {test_name}: not relevant")
