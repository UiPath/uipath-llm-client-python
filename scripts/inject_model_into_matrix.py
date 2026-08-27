#!/usr/bin/env python3
"""Add a model to the langchain integration matrix for ONE run, selecting clients by the
LLM Gateway's declared SupportedApiFlavors (see .github/workflows/model-onboarding.yml).

Each vendor client speaks one gateway API flavor; a model is only exercised through clients
whose flavor it declares:

  GeminiGenerateContent  -> google (UiPathChatGoogleGenerativeAI), litellm GEMINI_CONFIGS
  AnthropicMessages      -> anthropic (UiPathChatAnthropic, vendor_type by vendor)
                            + vertexai (UiPathChatAnthropicVertex)      [vendor VertexAi]
                            + bedrock  (UiPathChatAnthropicBedrock)     [vendor AwsBedrock]
                            + litellm VERTEX_CLAUDE_CONFIGS               [vendor VertexAi]
  AwsBedrockInvoke       -> bedrock (UiPathChatBedrock), litellm BEDROCK_INVOKE_CONFIGS
  AwsBedrockConverse     -> bedrock (UiPathChatBedrockConverse), litellm BEDROCK_CONVERSE_CONFIGS
  OpenAiChatCompletions  -> openai (UiPathAzureChatOpenAI), litellm OPENAI_CONFIGS
  OpenAiResponses        -> openai (UiPathAzureChatOpenAI, use_responses_api), litellm OPENAI_RESPONSES_CONFIGS
  NormalizedChat (op)    -> normalized (UiPathChat) always, thinking/reasoning config by family

  inject_model_into_matrix.py --repo . --model <id> --family <ModelFamily> --vendor <Vendor> \
      --flavors GeminiGenerateContent[,...] [--operations NormalizedChat]
Prints the conftest dirs touched on the last stdout line (consumed by the workflow).
"""
import argparse
import re
import sys
from pathlib import Path

CLIENTS = Path("tests/langchain/clients")
THINKING = '{"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}}'


def entries(cls: str, extra: str | None = None) -> str:
    """Config entries (plain + optional kwargs variant) as Python source, existing-list style."""
    base = f'{{"model_class": {cls}}}'
    if extra is None:
        return base
    return f'{base}, {{"model_class": {cls}, "model_kwargs": {extra}}}'


def gemini_list(model: str) -> str:
    return "GEMINI_3_CONFIGS" if re.search(r"gemini-3", model.lower()) else "GEMINI_2_5_CONFIGS"


def gpt_reasoning(model: str) -> bool:
    m = model.lower()
    return bool(re.match(r"(gpt-5|o\d)", m)) and "chat" not in m


def plan(model: str, family: str, vendor: str, flavors: set[str], operations: set[str]) -> dict[str, str]:
    """conftest dir -> Python expression for the model's config list."""
    out: dict[str, str] = {}
    lite: list[str] = []
    if "NormalizedChat" in operations:
        if family == "GoogleGemini":
            out["normalized"] = gemini_list(model)
        elif family == "AnthropicClaude":
            out["normalized"] = "CLAUDE_BEDROCK_CONFIGS" if vendor == "AwsBedrock" else "CLAUDE_VERTEXAI_CONFIGS"
        else:
            out["normalized"] = "GPT_REASONING_CONFIGS" if gpt_reasoning(model) else "GPT_NON_REASONING_CONFIGS"
    # Gateway convention: Vertex-hosted Claude entries declare GeminiGenerateContent (the enum has no
    # Vertex-Anthropic flavor); it means "Vertex passthrough", which for Claude is Anthropic Messages.
    if family == "AnthropicClaude" and vendor == "VertexAi" and "GeminiGenerateContent" in flavors:
        flavors = (flavors - {"GeminiGenerateContent"}) | {"AnthropicMessages"}
    if "GeminiGenerateContent" in flavors:
        out["google"] = gemini_list(model)
        lite.append("GEMINI_CONFIGS")
    if "AnthropicMessages" in flavors:
        if vendor == "AwsBedrock":
            out["anthropic"] = "CLAUDE_BEDROCK_CONFIGS"
        else:
            out["anthropic"] = "CLAUDE_VERTEXAI_CONFIGS"
            out["vertexai"] = "CLAUDE_VERTEXAI_CONFIGS"
            lite.append("VERTEX_CLAUDE_CONFIGS")
    bedrock: list[str] = []
    if "AnthropicMessages" in flavors and vendor == "AwsBedrock":
        bedrock.append(entries("UiPathChatAnthropicBedrock", THINKING))
    if "AwsBedrockInvoke" in flavors:
        bedrock.append(entries("UiPathChatBedrock", THINKING))
        lite.append("BEDROCK_INVOKE_CONFIGS")
    if "AwsBedrockConverse" in flavors:
        bedrock.append(entries("UiPathChatBedrockConverse", THINKING))
        lite.append("BEDROCK_CONVERSE_CONFIGS")
    if bedrock:
        out["bedrock"] = "[" + ", ".join(bedrock) + "]"
    openai: list[str] = []
    if "OpenAiChatCompletions" in flavors:
        openai.append(entries("UiPathAzureChatOpenAI", '{"reasoning_effort": "low"}' if gpt_reasoning(model) else None))
        lite.append("OPENAI_CONFIGS")
    if "OpenAiResponses" in flavors:
        if gpt_reasoning(model):
            kw = '{"use_responses_api": True, "reasoning": {"effort": "low", "summary": "auto"}, "verbosity": "low"}'
        else:
            kw = '{"use_responses_api": True}'
        openai.append(f'{{"model_class": UiPathAzureChatOpenAI, "model_kwargs": {kw}}}')
        lite.append("OPENAI_RESPONSES_CONFIGS")
    if openai:
        out["openai"] = "[" + ", ".join(openai) + "]"
    if lite:
        out["litellm"] = " + ".join(lite)
    return out


def inject(conftest: Path, model: str, expr: str) -> bool:
    text = conftest.read_text(encoding="utf-8")
    if f'"{model}"' in text:
        return False
    for name in re.findall(r"\b([A-Z][A-Z0-9_]+_CONFIGS)\b", expr):
        if not re.search(rf"^{name} = \[", text, re.M):
            raise SystemExit(f"{conftest}: config list {name} not found")
    new, n = re.subn(r"(COMPLETIONS_MODELS_WITH_CONFIGS = \{\n)", rf'\1    "{model}": {expr},\n', text, count=1)
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
    ap.add_argument("--flavors", required=True, help="comma-separated SupportedApiFlavors")
    ap.add_argument("--operations", default="NormalizedChat", help="comma-separated SupportedNormalizedOperations")
    a = ap.parse_args()
    flavors = {f.strip() for f in a.flavors.split(",") if f.strip()}
    ops = {o.strip() for o in a.operations.split(",") if o.strip()}
    p = plan(a.model, a.family, a.vendor, flavors, ops)
    if not p:
        raise SystemExit(f"no client covers flavors {sorted(flavors)} / operations {sorted(ops)}")
    touched = [d for d, expr in p.items() if inject(Path(a.repo) / CLIENTS / d / "conftest.py", a.model, expr)]
    for d, expr in p.items():
        print(f"  {d:11s} <- {expr}")
    print(f"injected {a.model} (flavors {','.join(sorted(flavors))}) into: {' '.join(touched) or '(already present)'}")
    print(" ".join(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
