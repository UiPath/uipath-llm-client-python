# UiPath LangChain Client Changelog

All notable changes to `uipath_langchain_client` will be documented in this file.

## [1.15.1] - 2026-06-30

### Fixed
- BYO AWS Bedrock backing-model resolution now reads the authoritative `byomDetails.customerModel` field that LLM Gateway discovery exposes for "add your own" (BYOMAdded) connections (LLM-3900). This is the upstream model the customer configured; `get_chat_model` uses it as `base_model_id`/`provider` for capability detection while the connection alias is still sent as `model_id` for gateway routing. Replaces the earlier speculative `byomDetails.baseModel`/`backingModel` lookups, which guessed at a field that did not yet exist. When `customerModel` is absent (UiPath-owned models) it falls back to the discovery `modelName`, which is itself the real model id.

## [1.15.0] - 2026-06-25

### Added
- `UiPathBaseChatModel` now converts provider SDK exceptions into `UiPathError`. When any chat client (`UiPathChatOpenAI`, `UiPathChatAnthropic`, `UiPathChat`, Bedrock/Vertex/Google/LiteLLM/Fireworks, …) raises during `_generate`/`_agenerate`/`_stream`/`_astream`, the error is converted into the matching UiPath semantic subclass (a 429 → `UiPathRateLimitError`, a 400 → `UiPathBadRequestError`, …) — including providers like Google that wrap the response-bearing error one level down — giving provider-agnostic error handling across every client. Errors without an HTTP response (e.g. client-side validation) become the `UiPathError` root. `UiPathError` is re-exported from `uipath_langchain_client`.

### Changed
- Bumped `uipath-llm-client` floor to `>=1.15.0` to pick up `UiPathError` and the `wrap_provider_errors` / `as_uipath_error` helpers.

### Note
- Provider errors are now surfaced as **pure** UiPath types and are **not** catchable as their original vendor type (e.g. `openai.BadRequestError`); the vendor exception is preserved as `__cause__`. Standardise handlers on `UiPathError` and its subclasses.

## [1.14.1] - 2026-06-23

### Fixed
- `WrappedBotoClient` (the httpx-backed Bedrock shim) now calls `raise_for_status()` in `converse`, `invoke_model`, and the streaming generator before reading the response. Previously a non-2xx gateway response (e.g. 403 License-not-available) was parsed as a normal result and handed to `langchain_aws`, which raised a misleading `ValueError("No 'output' key found in the response from the Bedrock Converse API ... misconfiguration of endpoint or region")` — the real status code and `detail` were lost. Gateway HTTP errors now surface as the patched `UiPathAPIError` subclass (e.g. `UiPathPermissionDeniedError`), matching the OpenAI and Vertex paths. For streaming responses the error body is read first so the typed exception retains its `detail`.

## [1.14.0] - 2026-06-15

### Added
- Support for the `AnthropicMessages` API flavor on Bedrock-hosted Claude models. `get_chat_model` now routes discovery's `apiFlavor=AnthropicMessages` (vendor `AwsBedrock`) to `UiPathChatAnthropic` configured with `vendor_type=awsbedrock` and `api_flavor=ApiFlavor.ANTHROPIC_MESSAGES`. The client keeps the Bedrock passthrough URL but uses the native `Anthropic`/`AsyncAnthropic` SDK (model-in-body wire format), which the gateway requires for this flavor, instead of `AnthropicBedrock`.
- `UiPathChatAnthropic` now accepts an explicit `api_flavor`. When set to `ApiFlavor.ANTHROPIC_MESSAGES` it selects the native Anthropic SDK regardless of `vendor_type`; otherwise the flavor and SDK are derived from `vendor_type` exactly as before (`awsbedrock` → `invoke` + `AnthropicBedrock`, unchanged).

### Changed
- Bumped `uipath-llm-client` floor to `>=1.14.0` to pick up `ApiFlavor.ANTHROPIC_MESSAGES`.

## [1.13.1] - 2026-06-09

### Fixed
- Picks up the core `uipath-llm-client` 1.13.1 fix allowing non-JWT access tokens (e.g. opaque UiPath reference tokens) as `UIPATH_ACCESS_TOKEN`, so LangChain clients built on `PlatformSettings` no longer fail validation with "Invalid access token: expected JWT with at least 2 dot-separated parts".

## [1.13.0] - 2026-05-27

### Changed
- **`UiPathBaseLLMClient.max_retries` field default raised from `0` to `3`.** Every LangChain chat and embedding client built on this base (`UiPathChat`, `UiPathChatOpenAI`, `UiPathAzureChatOpenAI`, `UiPathChatAnthropic`, `UiPathChatAnthropicBedrock`, `UiPathChatBedrock`, `UiPathChatBedrockConverse`, `UiPathChatVertexAI`, `UiPathChatFireworks`, `UiPathChatLiteLLM`, plus the matching embeddings classes) now retries failed requests 3 times by default. Pass `max_retries=0` explicitly to disable retries — the opt-out path is unchanged. Combined with the expanded default retry set in `uipath-llm-client` 1.13.0, every LangChain client now retries on HTTP 408, 429, 502, 503, 504, and 529 out of the box.
- Bumped `uipath-llm-client` floor to `>=1.13.0` to pick up the expanded default retry set and the new `UiPathRequestTimeoutError` / `UiPathBadGatewayError` typed exceptions.

## [1.12.2] - 2026-05-24

### Changed
- Bumped the default Azure OpenAI API version from `2025-03-01-preview` to `2025-04-01-preview` on `UiPathChatOpenAI`, `UiPathAzureChatOpenAI`, `UiPathOpenAIEmbeddings`, `UiPathAzureOpenAIEmbeddings`, `UiPathChatFireworks`, and `UiPathFireworksEmbeddings`.
- Bumped `uipath-llm-client` floor to `>=1.12.2` to pick up the matching API version default.

## [1.12.1] - 2026-05-22

### Changed
- `get_chat_model` once again routes to `UiPathChatAnthropicBedrock` when `api_flavor == ApiFlavor.INVOKE` and discovery reports `modelFamily == AnthropicClaude`. Other INVOKE families still use `UiPathChatBedrock`, and `None`/`CONVERSE` continue to use `UiPathChatBedrockConverse`.

## [1.12.0] - 2026-05-21

### Changed
- `get_chat_model` routes the AWSBEDROCK branch purely by `api_flavor`: `ApiFlavor.INVOKE` selects `UiPathChatBedrock`, while `None` or `ApiFlavor.CONVERSE` select `UiPathChatBedrockConverse`. Model family no longer influences the choice.

### Removed
- The AWSBEDROCK branch no longer auto-selects `UiPathChatAnthropicBedrock` for `ANTHROPIC_CLAUDE` models. Callers who want that class can pass it via the `custom_class` kwarg.

## [1.11.3] - 2026-05-21

### Fixed
- `UiPathBaseLLMClient.setup_model_info` now calls `strip_disabled_fields` after merging `disabled_params`, so constructor-set sampling fields (e.g. `UiPathChatAnthropicBedrock(model="anthropic.claude-opus-4-7", temperature=0.7)`) are nulled on the instance once `disabled_params` is resolved. Plugs the init-time leak called out as a known follow-up in 1.10.0 — langchain-anthropic and langchain-aws's Bedrock Converse client read `self.temperature`/`self.top_p`/etc. when serializing the request body, so the existing kwargs-level strip alone wasn't enough. A warning is logged per stripped field with the original value so the caller can see what was dropped.

### Changed
- Bumped `uipath-llm-client` floor to `>=1.11.3` to match the core release exposing `strip_disabled_fields`.

## [1.11.2] - 2026-05-18

### Changed
- Bumped `uipath-llm-client` floor to `>=1.11.2` to pick up the increased default `X-UiPath-LLMGateway-TimeoutSeconds` (295 → 895) in `UIPATH_DEFAULT_REQUEST_HEADERS`.

## [1.11.1] - 2026-05-13

### Fixed
- `UiPathDynamicHeadersCallback` now merges `get_headers()` into the dynamic-headers ContextVar instead of replacing it wholesale. This prevents two stacked callbacks from overwriting each other.

## [1.11.0] - 2026-05-08

### Changed
- Bumped `uipath-llm-client` floor to `>=1.11.0` to match the core release that flips `PlatformSettings.agenthub_config` default from `"agentsruntime"` to `None`.

## [1.10.1] - 2026-05-08

### Added
- `agenthub_config` kwarg on `get_chat_model` and `get_embedding_model`. When set, overrides `client_settings.agenthub_config` for that call via `model_copy` (the supplied settings instance is not mutated).

## [1.10.0] - 2026-04-23

### Added
- `model_details` and `disabled_params` fields on `UiPathBaseLLMClient`, plus a single `@model_validator(mode="after") setup_model_info` that (1) forwards the factory-supplied `model_details` or fetches it from `client_settings.get_model_info`, and (2) sets `disabled_params` to the merge of what the caller passed and what `disabled_params_from_model_details` derives — user keys win on conflicts, so callers can override any derived entry by name.
- `disabled_params` uses the langchain-openai shape (`{name: None | [values]}`), so subclasses inheriting from `ChatOpenAI` / `AzureChatOpenAI` also benefit from the native `_filter_disabled_params` path inside `bind_tools`.
- Runtime stripping in the four `_generate`/`_agenerate`/`_stream`/`_astream` wrappers on `UiPathBaseChatModel` delegates to `uipath.llm_client.utils.sampling.strip_disabled_kwargs`, generic over `disabled_params`. A warning is logged via `self.logger` for each stripped key when a logger is configured. Fixes `anthropic.claude-opus-4-7` rejecting any sampling parameter passed via `.invoke()` / `.ainvoke()` / streams.

### Removed
- The unused `disabled_params` field declaration on `UiPathChat` (now inherited from `UiPathBaseLLMClient`).

### Changed
- Bumped `uipath-llm-client` floor to `>=1.10.0` to match the release that adds `uipath.llm_client.utils.sampling`.

### Known follow-up
- Init-time values set on the instance (`UiPathChat(model="anthropic.claude-opus-4-7", temperature=0.5)`) still flow into the outgoing request body via `_default_params` / the vendor SDK. The runtime invoke-time strip handles `.invoke(..., temperature=...)`; a follow-up will plug the init-time leak using the already-populated `disabled_params`.

## [1.9.9] - 2026-04-23

### Changed
- Bumped dependency floors to the latest installed versions: `langchain-openai>=1.2.0`, `langchain-aws>=1.4.5`.
- Minimum `uipath-llm-client` bumped to 1.9.9 to match the core dependency-floor release.

## [1.9.8] - 2026-04-22

### Changed
- Added upper-bound version constraints (`<next_major`) to all dependencies to prevent unexpected breaking changes from major-version upgrades.
- `langchain-litellm` is pinned to an exact version (`==0.6.4`) alongside `litellm`.
- Minimum `uipath-llm-client` bumped to 1.9.8 to match the core dependency-constraints release.

## [1.9.7] - 2026-04-22

### Changed
- **Behavior change:** `UiPathChat._default_params` now uses pydantic's `model_fields_set` to decide which params to include in the request payload instead of filtering on `v is not None`. Fields that were not explicitly passed are omitted; fields explicitly set to `None` (e.g. `UiPathChat(temperature=None)`) now forward `null` to the API. Previously both "not passed" and "explicitly `None`" were silently dropped.

## [1.9.6] - 2026-04-22

### Added
- `UiPathChat` (normalized) now implements `_identifying_params` (returning `model_name` + `_default_params`) to match the `BaseChatOpenAI` convention, giving traced/cached runs a proper identity key instead of the empty default from `BaseChatModel`.

### Changed
- `UiPathChat._llm_type` now returns `"uipath-chat"` (was `"UiPath-Normalized"`) to align with the lowercase-hyphenated convention used by `openai-chat`, `azure-openai-chat`, etc.

### Changed
- Bumped dependency floors to the latest installed versions: `langchain-openai>=1.1.16`.
- Minimum `uipath-llm-client` bumped to 1.9.6 to match the core dependency-floor release.

## [1.9.5] - 2026-04-21

### Changed
- `UiPathBaseLLMClient.default_headers` is now additive: caller-supplied headers are merged on top of a class-level `class_default_headers` (timeout and `AllowFull4xxResponse` policy) instead of replacing them. User values still win on key collisions. Previously, passing any `default_headers={...}` caused the built-in defaults to be dropped from `self.default_headers` (though the core httpx client's class defaults kept them on the wire).
- `UiPathBaseLLMClient.class_default_headers` now points at the shared `uipath.llm_client.utils.headers.UIPATH_DEFAULT_REQUEST_HEADERS` constant (single source of truth with core's `UiPathHttpxClient._default_headers`).
- Minimum `uipath-llm-client` bumped to 1.9.5 for the shared `UIPATH_DEFAULT_REQUEST_HEADERS` constant.

## [1.9.4] - 2026-04-21

### Changed
- Bumped dependency floors to the latest released versions: `langchain-openai>=1.1.15`, `langchain-google-genai>=4.2.2`, `langchain-anthropic>=1.4.1`, `anthropic[bedrock,vertex]>=0.96.0`, `langchain-aws[anthropic]>=1.4.4`, `langchain-azure-ai>=1.2.2`.
- Minimum `uipath-llm-client` bumped to 1.9.4 to match the core dependency-floor release.

## [1.9.3] - 2026-04-20

### Changed
- `get_chat_model()` now defaults to the OpenAI Responses API (`ApiFlavor.RESPONSES`) when discovery does not specify a flavor for an OpenAI chat model. Explicit `api_flavor=` on the call and BYOM-discovered flavors still take precedence. The LiteLLM client still defaults to chat-completions for OpenAI because LiteLLM 1.83.x drops the injected httpx `client` when its acompletion→aresponses bridge fires, which breaks async auth against the UiPath gateway.

## [1.9.2] - 2026-04-17

### Changed
- **Breaking:** captured gateway headers are now exposed on `AIMessage.response_metadata` under the `headers` key (previously `uipath_llmgateway_headers`). Update any consumers that read this key.
- Minimum `uipath-llm-client` bumped to 1.9.2 for the platform-headers refactor and licensing-context support.

## [1.9.1] - 2026-04-17

### Fixed
- Detect Anthropic-family models by additional name keywords (`anthropic`, `opus`, `sonnet`, `haiku`, `mythos`) alongside `claude` — applies to Bedrock INVOKE factory routing and the normalized client's empty tool-call content workaround. Uses the shared `is_anthropic_model_name()` helper from core 1.9.1.

### Changed
- Minimum `uipath-llm-client` bumped to 1.9.1 for the shared `is_anthropic_model_name()` helper

## [1.9.0] - 2026-04-17

### Changed
- Factory functions use `ModelFamily` constants and `get_model_info()` from core instead of inline discovery logic
- Azure vs non-Azure OpenAI routing now uses `modelFamily` instead of `modelSubscriptionType`

## [1.8.3] - 2026-04-16

### Added
- Factory functions (`get_chat_model`, `get_embedding_model`) now automatically resolve BYOM discovery API flavors to the correct client and routing flavor

## [1.8.2] - 2026-04-13

### Changed
- Version bump to match core package 1.8.2

## [1.8.1] - 2026-04-09

### Changed
- Renamed `aws` optional dependency to `bedrock` for clarity

## [1.8.0] - 2026-04-08

### Added
- `UiPathChatLiteLLM` — LangChain chat model powered by LiteLLM, supporting all UiPath gateway providers
- `langchain-litellm` as an optional dependency for LiteLLM integration

### Changed
- Updated dependency versions: `anthropic[bedrock,vertex]>=0.91.0`
- Version bump to match core package 1.8.0

## [1.7.1] - 2026-04-04

### Added
- `custom_class` parameter in `get_chat_model()` and `get_embedding_model()` factory functions to allow instantiating a user-provided class instead of the auto-detected one

## [1.7.0] - 2026-04-03

### Added
- `UiPathChat.with_structured_output()` — supports `function_calling`, `json_schema`, and `json_mode` methods
- `UiPathChat.bind_tools()` — added `parallel_tool_calls` parameter
- Added vendor-specific parameters to `UiPathChat`: `logit_bias`, `logprobs`, `top_logprobs`, `parallel_tool_calls`, `top_k`, `safety_settings`, `cached_content`, `labels`, `seed`

## [1.6.0] - 2026-04-03

### Fixed
- Version bump to match core package changes

## [1.5.10] - 2026-03-26

### Changed
- Removed `X-UiPath-LLMGateway-AllowFull4xxResponse` from default request headers to avoid PII leakage in logs

## [1.5.9] - 2026-03-26

### Fixed
- Remove `fix_host_header` event hooks from `UiPathChatOpenAI`; host header management is handled by the underlying httpx client

## [1.5.8] - 2026-03-26

### Fixed
- Pass `base_url` to `OpenAI` and `AsyncOpenAI` constructors in `UiPathChatOpenAI` to ensure the correct endpoint is used by the underlying SDK clients

## [1.5.7] - 2026-03-23

### Fixed
- Fix factory for BYO to handle the case where vendor_type is None, but api_flavor is discovered

## [1.5.6] - 2026-03-21

### Feature
- Added `UiPathDynamicHeadersCallback`: extend and implement `get_headers()` to inject custom headers into each LLM gateway request
- Uses `run_inline = True` so `on_chat_model_start`/`on_llm_start` run in the caller's coroutine, ensuring ContextVar mutations propagate to `httpx.send()`
- Cleanup via `on_llm_end`/`on_llm_error`

## [1.5.5] - 2026-03-19

### Fixed
- Fix headers 

## [1.5.4] - 2026-03-19

### Fixed
- Fix bedrock clients with file attachments

## [1.5.3] - 2026-03-18

### Fixed
- Factory function fix

## [1.5.2] - 2026-03-18

### Fixed
- Factory function fix

## [1.5.1] - 2026-03-17

### Fixed
- Fixes to core package, version bump

## [1.5.0] - 2026-03-16

### Stable Version 1.5.0
- Updated package to include the changes in core

## [1.4.0] - 2026-03-13

### New client
- Added UiPathChatAnthropicBedrock
- refactored factory function to use the new client
- brought the enums from the base client


## [1.3.1] - 2026-03-12

### Fixed
- Fix normalized client raise error

## [1.3.0] - 2026-03-10

### Version Bump
- Stable release
- Fixes

## [1.2.7] - 2026-02-26

### Fixed
- Fix Bedrock clients model_id

## [1.2.6] - 2026-02-26

### Refactor
- Restructured project such that uipath_llm_client can be included in uipath as submodule.
- imports are now from uipath.llm_client instead of uipath_llm_client

## [1.2.5] - 2026-02-26

### Fixed
- Parameters on factory fix

## [1.2.4] - 2026-02-26

### Fixed
- Fix typing on factory method

## [1.2.3] - 2026-02-25

### Feature
- Capture headers and inject them in response_metadata.

## [1.2.2] - 2026-02-23

### Fixed
- Fixes to discovery endpoint on LLMGW

## [1.2.1] - 2026-02-18

### Fixed
- Timeout fixes, change typing from int to float
- remove timeout=None from all clients -> caused overriding the default timeout set up on the UiPathHttpxClient

## [1.2.0] - 2026-02-18

### Stable release

## [1.1.9] - 2026-02-13

### Changed
- Updated documentation

## [1.1.8] - 2026-02-13

### Refactor
- Adjust Anthropic factory method to use ChatAnthropic instead of ChatAnthropicVertex

## [1.1.7] - 2026-02-13

### Refactor
- Re-export UiPathChat and UiPathEmbeddings at module level

## [1.1.6] - 2026-02-12

### Fixed
- Added proper type hints for factory method

## [1.1.5] - 2026-02-12

### Fixed
- Fixed bedrock converse api

## [1.1.4] - 2026-02-12

### Fixed
- Fixed anthropic default vendor

## [1.1.3] - 2026-02-12

### Fixed
- Fixes on openai langchain client on responses_api
- Allow the flavor to be set up at request time, not just when instantiating the llm
- Some fixes for the anthropic client

## [1.1.2] - 2026-02-12

### Refactor
- Rename normalized client for better compatibility with other packages

## [1.1.1] - 2026-02-11

### Fixed
- Fix langchain fireworks client for async usage

## [1.1.0] - 2026-02-11

### Features
- Added langchain fireworks client and tested with GLM

### Stable release
- Fixed BYO on passthrough
- Stable release

## [1.0.13] - 2026-02-05

### Fixed
- Bump version

## [1.0.12] - 2026-02-05

### Fixed
- Added 295 as default llmgateway timeout to avoid problems on the backend side

## [1.0.11] - 2026-02-04

### Fixed
- Import TypedDict from typing_extension

## [1.0.10] - 2026-02-04

### Version Upgrade
- Updated version

## [1.0.9] - 2026-02-04

### Fixed
- Fixed typing in core package, updated dependency

## [1.0.8] - 2026-02-04

### Fixed
- Added py.typed to the package

## [1.0.7] - 2026-02-04

### Refactor
- Refactor factory function to include byom models

## [1.0.6] - 2026-02-03

### Refactor
- Updated documentation to include the new aliases for settings
- New alias for settings and request timeout in BaseLLMClient

## [1.0.5] - 2026-02-03

### Fixed
- Fixed retry logic on all clients

## [1.0.4] - 2026-02-03

### Fixed
- Fix some timout issues on langchain_openai from llmgw.

## [1.0.3] - 2026-02-02

### Fixed
- Added better dependencies for langchain-anthropic to include boto and vertex

## [1.0.2] - 2026-02-02

### Fixed
- Removed old fix on Gemini streaming and updated with a new cleaner one

## [1.0.1] - 2026-02-02

### Fixed
- Fixed Api Version on OpenAI Embeddings

## [1.0.0] - 2026-01-30

### Official Release
- First stable release of the UiPath LangChain Client
- API considered stable; semantic versioning will be followed from this point forward

### Highlights
- Production-ready LangChain integrations for all major LLM providers
- Factory functions for automatic vendor detection and model instantiation
- Full compatibility with LangChain agents, tools, and chains
- Comprehensive support for chat completions, embeddings, and streaming
- Seamless integration with both AgentHub and LLMGateway backends

## [0.3.x] - 2026-01-29

### Release
- First public release of the UiPath LangChain Client
- Production-ready integration with LangChain ecosystem

### Documentation
- Complete rewrite of README.md with installation, quick start, and API reference
- Added comprehensive usage examples for all supported providers
- Added module-level and class-level docstrings throughout the codebase

### Features
- Factory functions (`get_chat_model`, `get_embedding_model`) for auto-detecting model vendors
- Normalized API support for provider-agnostic chat completions and embeddings
- Full compatibility with LangChain agents and tools

### Supported Providers
- OpenAI
- Google
- Anthropic
- AWS Bedrock
- Vertex AI
- Azure AI

## [0.2.x] - 2026-01-15

### Architecture
- Extracted from monolithic package into dedicated LangChain integration package
- Now depends on `uipath_llm_client` core package for HTTP client and authentication
- Unified client architecture supporting both AgentHub and LLMGateway backends

### Chat Model Classes
- `UiPathChatOpenAI` - OpenAI models via direct API
- `UiPathAzureChatOpenAI` - OpenAI models via Azure
- `UiPathChatGoogleGenerativeAI` - Google Gemini models
- `UiPathChatAnthropic` - Anthropic Claude models
- `UiPathChatAnthropicVertex` - Claude models via Google VertexAI
- `UiPathChatBedrock` - AWS Bedrock models
- `UiPathChatBedrockConverse` - AWS Bedrock Converse API
- `UiPathAzureAIChatCompletionsModel` - Azure AI models (non-OpenAI)
- `UiPathChat` - Provider-agnostic normalized API

### Embeddings Classes
- `UiPathOpenAIEmbeddings` - OpenAI embeddings via direct API
- `UiPathAzureOpenAIEmbeddings` - OpenAI embeddings via Azure
- `UiPathGoogleGenerativeAIEmbeddings` - Google embeddings
- `UiPathBedrockEmbeddings` - AWS Bedrock embeddings
- `UiPathAzureAIEmbeddingsModel` - Azure AI embeddings
- `UiPathEmbeddings` - Provider-agnostic normalized API

### Features
- Support for BYO (Bring Your Own) model connections

### Breaking Changes
- Package renamed from internal module to `uipath_langchain_client`
- Import paths changed; update imports accordingly

## [0.1.x] - 2025-12-20

### Initial Development Release
- LangChain-compatible chat models wrapping UiPath LLM services
- Passthrough clients for:
  - OpenAI
  - Google Gemini
  - Anthropic
  - AWS Bedrock
  - Vertex AI
  - Azure AI
- Embeddings support for text-embedding models
- Streaming support (sync and async)
- Tool/function calling support
- Full compatibility with LangChain's `BaseChatModel` interface
- httpx-based HTTP handling for consistent behavior
