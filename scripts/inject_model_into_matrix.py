#!/usr/bin/env python3
"""Add a model to uipath-llm-client-python's langchain integration matrix (workflow-local edit, see .github/workflows/model-onboarding.yml).

The client repo parameterizes its integration tests from per-provider conftest dicts
(COMPLETIONS_MODELS_WITH_CONFIGS). This inserts the model into the normalized-client dict
(UiPathChat, cross-provider) and the matching vendor-client dict, picking the thinking/
reasoning config list by model family the same way the existing entries do.

  inject_model_into_matrix.py --repo . --model <id> --family <ModelFamily> --vendor <VertexAi|AwsBedrock|OpenAi|NativeOpenAi>
"""
import argparse
import re
import sys
from pathlib import Path

CLIENTS = Path("tests/langchain/clients")


def pick(model: str, family: str, vendor: str) -> list[tuple[str, str]]:
    """(conftest dir, CONFIGS list name) pairs to insert into."""
    m = model.lower()
    if family == "GoogleGemini":
        gem = "GEMINI_3_CONFIGS" if re.search(r"gemini-3", m) else "GEMINI_2_5_CONFIGS"
        return [("normalized", gem), ("google", gem)]
    if family == "AnthropicClaude":
        if vendor == "AwsBedrock" or m.startswith("anthropic."):
            return [("normalized", "CLAUDE_BEDROCK_CONFIGS"), ("bedrock", "CLAUDE_BEDROCK_CONFIGS")]
        return [("normalized", "CLAUDE_VERTEXAI_CONFIGS"), ("vertexai", "CLAUDE_VERTEXAI_CONFIGS")]
    if family in ("OpenAi", "OpenAI", "Gpt", "GPT"):
        reasoning = bool(re.match(r"(gpt-5|o\d)", m)) and "chat" not in m
        return [("normalized", "GPT_REASONING_CONFIGS" if reasoning else "GPT_NON_REASONING_CONFIGS"),
                ("openai", "GPT_MODELS_WITH_REASONING_CONFIGS" if reasoning else "GPT_MODELS_NON_REASONING_CONFIGS")]
    return [("normalized", "GPT_NON_REASONING_CONFIGS")]


def inject(conftest: Path, model: str, configs: str) -> bool:
    text = conftest.read_text(encoding="utf-8")
    if f'"{model}"' in text:
        return False
    if configs not in text:
        raise SystemExit(f"{conftest}: config list {configs} not found")
    new, n = re.subn(r"(COMPLETIONS_MODELS_WITH_CONFIGS = \{\n)", rf'\1    "{model}": {configs},\n', text, count=1)
    if n != 1:
        raise SystemExit(f"{conftest}: COMPLETIONS_MODELS_WITH_CONFIGS dict not found")
    conftest.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--vendor", required=True)
    a = ap.parse_args()
    touched = []
    for d, configs in pick(a.model, a.family, a.vendor):
        f = Path(a.repo) / CLIENTS / d / "conftest.py"
        if inject(f, a.model, configs):
            touched.append(f"{d}:{configs}")
    print(f"injected {a.model} into: {' '.join(touched) or '(already present)'}")
    print(" ".join(d for d, _ in pick(a.model, a.family, a.vendor)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
