# CLAUDE.md — UiPath LLM Client

## Project Overview

This is a Python monorepo providing a unified client for UiPath's LLM services. It has two published packages:

- **`uipath-llm-client`** (core) — located at `src/uipath/llm_client/`. Core HTTP client with auth, retry, and provider-specific clients (OpenAI, Google, Anthropic).
- **`uipath-langchain-client`** (langchain) — located at `packages/uipath_langchain_client/`. LangChain-compatible chat models and embeddings, depends on the core package.

Supported backends: **AgentHub** (default), **LLMGateway**, and **Orchestrator**.

---

## Development Setup

Uses [uv](https://docs.astral.sh/uv/) with workspace support.

```bash
uv sync --all-extras
```

Run tests:
```bash
pytest tests
```

Lint and format:
```bash
ruff check
ruff format --check
pyright
```

Tests use VCR cassettes (SQLite, `tests/cassettes.db`) via `pytest-recording`. HTTP is recorded and replayed; real network calls are not made in CI.

---

## Repository Structure

```
src/uipath/llm_client/          # Core package source
  clients/                      # Provider clients (openai, google, anthropic)
  settings/                     # Backend settings (platform, llmgateway)
  utils/                        # Shared utilities
  __version__.py                # Core version string
packages/uipath_langchain_client/
  src/uipath_langchain_client/
    clients/                    # LangChain model wrappers per provider
    __version__.py              # LangChain version string
  pyproject.toml                # LangChain package config (declares core dep)
pyproject.toml                  # Root workspace config
CHANGELOG.md                    # Core changelog
packages/uipath_langchain_client/CHANGELOG.md  # LangChain changelog
tests/                          # All tests (core/, langchain/, llamaindex/)
.github/workflows/              # CI/CD pipelines
```

---

## Versioning Rules

Both packages follow semantic versioning. The CD pipelines trigger on version changes in the respective `__version__.py` files.

### When core client changes

Update **all** of the following:

1. `src/uipath/llm_client/__version__.py` — bump core version
2. `packages/uipath_langchain_client/src/uipath_langchain_client/__version__.py` — bump to same version
3. `packages/uipath_langchain_client/pyproject.toml` — update the `uipath-llm-client >= X.Y.Z` dependency to match the new core version
4. `CHANGELOG.md` — add entry under new version
5. `packages/uipath_langchain_client/CHANGELOG.md` — add entry under new version

### When only langchain changes

Update **only**:

1. `packages/uipath_langchain_client/src/uipath_langchain_client/__version__.py` — bump langchain version
2. `packages/uipath_langchain_client/CHANGELOG.md` — add entry under new version

Do **not** touch the core `__version__.py` or root `CHANGELOG.md` for langchain-only changes.

---

## CHANGELOG Format

Follow the existing format — newest version first, grouped by date:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added / Fixed / Changed
- Description of the change
```

---

## Pre-commit / Pre-PR Checklist

Before every commit and before opening a PR, always run:

```bash
ruff check && ruff format . && pyright && pytest tests
```

All four must pass. Fix any lint, format, type, or test failures before committing. This applies when working as an AI assistant too — run the checks, fix failures, then commit and push.

---

## PR Guidelines

### Before Opening a PR

- Run the pre-commit checklist above — all must pass.
- Apply versioning rules above: the CI workflow (`ci_change_version.yml`) enforces that any changed source files must have a corresponding version bump and changelog entry.

### PR Scope

- Keep PRs focused. One logical change per PR.
- If a change touches both core and langchain, that is a single PR — both packages version together.
- Do not mix version bumps with unrelated refactors.

### What to Include in the PR Description

- What changed and why.
- Which package(s) are affected (core, langchain, or both).
- Reference any related issues.

### Dev Builds

Add the `build:dev` label to a PR to trigger a dev package publish to TestPyPI. The PR description will be updated automatically with install instructions.

---

## Code Style

- **Formatter/linter:** ruff (config in root `pyproject.toml`)
- **Type checker:** pyright (strict)
- **Python version:** 3.11+ required, 3.13 used in CI
- StrEnum string comparisons are acceptable — do not replace with enum constants.
- Provider clients live under `clients/<provider>/` in both the core and langchain packages.
- Settings use pydantic-settings with `BaseSettings`; do not add ad-hoc config.

---

## CI/CD Pipelines

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PR to main | Lint, type check, run tests |
| `ci_change_version.yml` | PR to main | Validate version bump + changelog |
| `cd.yml` | Push to main (core version change) | Build + publish core to PyPI |
| `cd-langchain.yml` | Push to main (langchain version change) | Build + publish langchain to PyPI |
| `publish-dev.yml` | PR with `build:dev` label | Publish dev build to TestPyPI |

The langchain CD waits for the core package to appear on PyPI before building, since it depends on it.
