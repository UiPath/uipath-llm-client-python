"""Keep ``temperature`` on gpt-5 variants that default to no reasoning.

langchain-openai strips ``temperature`` for every ``gpt-5*`` model unless reasoning
effort is explicitly the string ``"none"``, at two sites in
``langchain_openai.chat_models.base``: ``validate_temperature`` and
``_construct_responses_api_payload``. Unset effort is ``None``, not ``"none"``, so
the value is dropped by default. That is right for base ``gpt-5`` and the ``pro``
variants (default effort medium) but wrong for the dotted ones (``gpt-5.2``,
``gpt-5.4``), which default to effort ``none`` and do accept it.

Delete once langchain-ai/langchain#35424 ships. Upstream: langchain-ai/langchain#35423.
"""

import re
from collections.abc import Mapping
from typing import Any, cast

from langchain_core.language_models import LanguageModelInput
from pydantic import model_validator

_DOTTED_GPT5 = re.compile(r"gpt-5\.\d+")


def gpt5_keeps_temperature(
    model: str | None,
    *,
    reasoning_effort: str | None = None,
    reasoning: Mapping[str, Any] | None = None,
) -> bool:
    """Whether this model accepts ``temperature`` as currently configured.

    True only for dotted gpt-5 variants with effort unset or ``"none"``. False
    everywhere else, so the caller defers to langchain.
    """
    name = (model or "").lower()
    if "chat" in name or "pro" in name:
        return False
    if not _DOTTED_GPT5.match(name):
        return False
    effort = reasoning_effort or (reasoning or {}).get("effort")
    return effort is None or effort == "none"


class Gpt5TemperatureMixin:
    """Restore ``temperature`` at both sites langchain-openai strips it.

    Mix in ahead of the vendor chat class so the overrides win on the MRO.
    """

    @model_validator(mode="before")
    @classmethod
    def validate_temperature(cls, values: dict[str, Any]) -> Any:
        """Skip langchain's strip when the model does support ``temperature``."""
        if gpt5_keeps_temperature(
            values.get("model_name") or values.get("model"),
            reasoning_effort=values.get("reasoning_effort"),
            reasoning=values.get("reasoning"),
        ):
            return values
        return cast(Any, super()).validate_temperature(values)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Put ``temperature`` back after the Responses payload builder drops it."""
        payload = cast(
            dict[str, Any],
            cast(Any, super())._get_request_payload(input_, stop=stop, **kwargs),
        )
        if "temperature" in payload:
            return payload
        requested = kwargs.get("temperature", getattr(self, "temperature", None))
        if requested is None:
            return payload
        if "temperature" in (getattr(self, "disabled_params", None) or {}):
            return payload
        if gpt5_keeps_temperature(payload.get("model"), reasoning=payload.get("reasoning")):
            payload["temperature"] = requested
        return payload
