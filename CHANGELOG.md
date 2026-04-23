# UiPath LLM Client Changelog

All notable changes to `uipath_llm_client` (core package) will be documented in this file.

## [1.9.9] - 2026-04-23

### Changed
- Bumped dependency floors to the latest installed versions: `uipath-platform>=0.1.35`.

## [1.9.8] - 2026-04-22

### Changed
- Added upper-bound version constraints (`<next_major`) to all production and dev dependencies to prevent unexpected breaking changes from major-version upgrades.
- `litellm` is now pinned to an exact version (`==1.83.7`) due to its frequent breaking changes between minor releases.

## [1.9.6] - 2026-04-22

### Changed
- Bumped `uipath-platform>=0.1.34` to match the latest installed version.

## [1.9.5] - 2026-04-21

### Added
- `utils.headers.UIPATH_DEFAULT_REQUEST_HEADERS` public constant — the single source of truth for built-in gateway request headers (`X-UiPath-LLMGateway-TimeoutSeconds=295`, `X-UiPath-LLMGateway-AllowFull4xxResponse=false`). `UiPathHttpxClient._default_headers` now references this constant; the langchain base client reuses the same constant for its `class_default_headers`.

## [1.9.4] - 2026-04-21

### Changed
- Bumped dependency floors to the latest released versions: `pydantic-settings>=2.14.0`, `uipath-platform>=0.1.32`, `google-genai>=1.73.1`, `anthropic>=0.96.0`, `litellm>=1.83.7`.

## [1.9.2] - 2026-04-17

### Changed
- `PlatformBaseSettings.build_auth_headers()` now uses the header-name constants from `uipath.platform.common.constants` (lowercase canonical form). HTTP header names are case-insensitive so wire-level behavior is unchanged.
- `UIPATH_PROCESS_KEY` is now URL-encoded (`urllib.parse.quote(..., safe="")`) before being placed in `X-UiPath-ProcessKey`, matching the platform-wide convention.

### Added
- `HEADER_LICENSING_CONTEXT` header populated dynamically from `UiPathConfig.licensing_context` when set.

## [1.9.1] - 2026-04-17

### Added
- `utils.model_family.is_anthropic_model_name()` helper and `ANTHROPIC_MODEL_NAME_KEYWORDS` tuple — name-based Claude detection for BYOM deployments where discovery does not expose `modelFamily`

### Fixed
- `UiPathLiteLLM` now detects Claude-family models by name when `modelFamily` is unavailable (BYOM), correctly routing Bedrock/Vertex provider selection and default flavors

## [1.9.0] - 2026-04-17

### Added
- `ModelFamily` StrEnum constants (`OPENAI`, `GOOGLE_GEMINI`, `ANTHROPIC_CLAUDE`) for model family matching
- `get_model_info()` on `UiPathBaseSettings` — centralized model lookup with filtering by name, vendor, and BYO connection ID
- Discovery cache on `get_available_models()` keyed by settings properties, with `refresh` parameter to bypass

### Changed
- `get_available_models()` is now a concrete cached method on the base class; subclasses implement `_fetch_available_models()` instead
- `validate_byo_model()` is now a default no-op on the base class (only LLMGateway overrides it) and is called automatically inside `get_model_info()`
- LiteLLM client uses `get_model_info()` instead of duplicating model discovery logic

## [1.8.3] - 2026-04-16

### Added
- BYOM API flavor constants for discovery endpoint: `OpenAiChatCompletions`, `OpenAiResponses`, `OpenAiEmbeddings`, `GeminiGenerateContent`, `GeminiEmbeddings`, `AwsBedrockInvoke`, `AwsBedrockConverse`
- `BYOM_TO_ROUTING_FLAVOR` mapping to resolve BYOM discovery flavors to routing-level API flavors
- Extended `API_FLAVOR_TO_VENDOR_TYPE` with BYOM flavor entries for automatic vendor resolution
- LiteLLM client now resolves BYOM discovery flavors to correct routing flavors and litellm providers

## [1.8.2] - 2026-04-13

### Fixed
- Removed top-level import of `UiPathLiteLLM` from `__init__.py` to avoid `ImportError` when the optional `litellm` dependency is not installed

## [1.8.0] - 2026-04-08

### Added
- `UiPathLiteLLM` — provider-agnostic LLM client powered by LiteLLM
  - `completion` / `acompletion` for chat completions across all providers
  - `embedding` / `aembedding` for embeddings
  - Automatic model discovery from the UiPath backend — detects vendor, api_flavor, and model family
  - Optional `vendor_type` and `api_flavor` overrides (same pattern as LangChain factory)
  - Supports OpenAI (chat-completions + responses API), Gemini, Bedrock (invoke + converse), and Vertex AI Claude
  - All HTTP routed through UiPath httpx transport (auth, retry, headers) — no direct calls to Google/AWS/OpenAI
  - Explicit completion parameters with full IDE autocomplete
- `litellm` as an optional dependency (`uv add uipath-llm-client[litellm]`)
- `_strict_response_validation` parameter to all Anthropic client classes

### Changed
- Updated dependency versions: `uipath-platform>=0.1.21`, `anthropic>=0.91.0`, `litellm>=1.83.4`

## [1.7.0] - 2026-04-03

### Added
- `UiPathNormalizedClient` — provider-agnostic LLM client with no optional dependencies
  - `client.completions.create/acreate/stream/astream` for chat completions
  - `client.embeddings.create/acreate` for embeddings
  - Structured output via `response_format` (Pydantic, TypedDict, dict, json_object)
  - Tool calling with dicts, Pydantic models, or callables
  - Streaming with SSE parsing
  - Full vendor parameter coverage: OpenAI (reasoning, logprobs, logit_bias), Anthropic (thinking, top_k), Google (thinking_level/budget, safety_settings, cached_content)
  - Typed response models: `ChatCompletion`, `ChatCompletionChunk`, `EmbeddingResponse`
  - Accepts both dict and Pydantic model messages

## [1.6.0] - 2026-04-03

### Fixed
- Set `api_flavor` to `None` for ANTHROPIC and AZURE vendor types
- Add ANTHROPIC/AZURE cases to validator and remove unused `original_message` parameter
- Fix VertexAI `default_headers` consistency and demo import path
- Fix LLMGateway singleton cache key to include `base_url`

## [1.5.10] - 2026-03-26

### Changed
- Removed `X-UiPath-LLMGateway-AllowFull4xxResponse` from default request headers to avoid PII leakage in logs

## [1.5.9] - 2026-03-26

### Fixed
- Use `availableOperationCodes` field (instead of `operationCodes`) when validating BYOM operation codes

## [1.5.8] - 2026-03-26

### Fixed
- Pass `base_url` to `OpenAI` and `AsyncOpenAI` constructors in `UiPathOpenAI` and `UiPathAsyncOpenAI` to ensure the correct endpoint is forwarded to the underlying SDK clients

## [1.5.7] - 2026-03-23

### Fixed
- Added mapping api_flavor to vendor_type

## [1.5.6] - 2026-03-21

### Feature
- Added `_DYNAMIC_REQUEST_HEADERS` ContextVar and helper functions (`get_dynamic_request_headers`, `set_dynamic_request_headers`) to `utils/headers.py`
- Inject dynamic request headers in httpx `send()` for both sync and async clients

## [1.5.5] - 2026-03-19

### Fixed
- Fix headers for Platform Settings

## [1.5.3] - 2026-03-18

### Fixed
- Factory function fix

## [1.5.2] - 2026-03-18

### Fixed
- Factory function fix

## [1.5.1] - 2026-03-17

### Fixed
- Added error message for normalized embeddings on UiPath Platform (AgentHub/Orchestrator) as there is no supported endpoint
- Fix endpoints for platform to remove api version

## [1.5.0] - 2026-03-16

### Stable Version 1.5.0
- added also backend for orchestrator and renamed AgentHubSettings to PlatformSetting (contains both agenthub and orchestrator)
- removed dependencies from 'uipath', now the repo depends on uipath.platform
- renamed var UIPATH_LLM_BACKEND with UIPATH_LLM_SERVICE in order to match uipath.platform
- More test fixes and documentation update

## [1.4.0] - 2026-03-13

### New client
- Added UiPathChatAnthropicBedrock

### Tests
- added sqlite serializer
- test updates
- new cassettes

### Fixed
- Added constants for VendorType and ApiFlavor

## [1.3.0] - 2026-03-10

### Version Bump
- Stable release
- Fixes

## [1.2.4] - 2026-02-26

### Refactor
- Restructured project such that uipath_llm_client can be included in uipath as submodule.
- imports are now from uipath.llm_client instead of uipath_llm_client

## [1.2.3] - 2026-02-25

### Feature
- Capture LLMGW headers

## [1.2.2] - 2026-02-23

### Fixed
- Fixes to discovery endpoint on LLMGW

## [1.2.1] - 2026-02-18

### Fixed
- Timeout fixes, change typing from int to float
- remove timeout=None from all clients -> caused overriding the default timeout set up on the UiPathHttpxClient

## [1.2.0] - 2026-02-18

### Stable release

### Fixed
- Fixed agenthub auth when token already exists

## [1.1.1] - 2026-02-12

### Fixed
- Small fixes on openai client

## [1.1.0] - 2026-02-11

### Stable release
- Added BYOM validation for settings
- Stable release

## [1.0.13] - 2026-02-05

### Fixed
- Fixed headers on llmgw settings

## [1.0.12] - 2026-02-05

### Fixed
- Added 295 as default llmgateway timeout to avoid problems on the backend side

## [1.0.11] - 2026-02-05

### Feature
- Added retry handler on 429 to include the retry-after header

## [1.0.10] - 2026-02-04

### Fixed
- Import TypedDict from typing_extension

### Fixed
- import @override from typing_extension

## [1.0.8] - 2026-02-04

### Fixed
- Fixed a typing issue of Singleton

## [1.0.7] - 2026-02-04

### Fixed
- Added py.typed to the package

## [1.0.6]

### Fixed
- Fixed model discovery on AgentHub Settings.

## [1.0.5] - 2026-02-03

### Refactor
- Updated documentation to include the new aliases for settings

## [1.0.4] - 2026-02-03

### Fixed
- Adjusted retry logic, now 0 means no retries, 1 means one retry

## [1.0.3] - 2026-02-02

### Refactor
- moved the logic of get_httpx_ssl_client_kwargs from the uipath package to this package; 

## [1.0.2] - 2026-02-02

### Fixed
- Fixed endpoints on AgentHub Settings

## [1.0.1] - 2026-01-30

### Fixed
- Map 400 Bad requests on S2S to 401 Unauthorized for better readability

## [1.0.0] - 2026-01-30

### Official Release
- First stable release of the UiPath LLM Client
- API considered stable; semantic versioning will be followed from this point forward

### Highlights
- Unified client architecture supporting both AgentHub and LLMGateway backends
- Production-ready passthrough clients for OpenAI, Google, Anthropic, AWS Bedrock, VertexAI, and Azure AI
- Normalized API for provider-agnostic LLM access
- Comprehensive authentication support (CLI-based for AgentHub, S2S for LLMGateway)
- Full async/sync support with streaming capabilities
- Robust error handling and retry logic

## [0.3.x] - 2026-01-29

### Release
- First public release accessible to test.pypi of the UiPath LLM Client
- Production-ready for both AgentHub and LLMGateway backends

### Documentation
- Complete rewrite of README.md with architecture overview, installation instructions, and comprehensive usage examples
- Added detailed documentation for `AgentHubSettings` and `LLMGatewaySettings` with all configuration options
- Added module-level docstrings to all major modules

### Features
- Added `get_default_client_settings()` factory function for easy backend selection
- Added `UIPATH_LLM_BACKEND` environment variable for runtime backend switching
- Improved error handling with `UiPathAPIError` hierarchy for specific HTTP status codes

## [0.2.x] - 2026-01-15

### Architecture
- Split monolithic package into two separate packages:
  - `uipath_llm_client` - Core HTTP client with authentication and retry logic
  - `uipath_langchain_client` - LangChain-specific integrations (moved to separate package)
- Merged LLMGateway and AgentHub client implementations into unified architecture
- Introduced `UiPathBaseSettings` as common base for backend-specific settings

### Features
- Added `AgentHubSettings` with automatic CLI-based authentication via `uipath auth`
- Added `LLMGatewaySettings` with S2S (server-to-server) authentication support
- Added support for BYO (Bring Your Own) model connections via `byo_connection_id`
- Unified retry logic with configurable `RetryConfig`

### Breaking Changes
- Package structure changed; imports need to be updated from `uipath_llmgw_client` to `uipath_llm_client`
- Settings classes renamed for consistency

## [0.1.x] - 2025-12-20

### Initial Development Release
- Core HTTP client with authentication and request handling
- Passthrough clients for completions and embeddings:
  - OpenAI/Azure OpenAI
  - Google Gemini
  - Anthropic
  - AWS Bedrock
  - Vertex AI
  - Azure AI
- Normalized API for provider-agnostic requests
- Streaming support (sync and async)
- Retry logic with exponential backoff
- Custom exception hierarchy for API errors
- Wrapped all clients to use httpx for consistent HTTP handling
