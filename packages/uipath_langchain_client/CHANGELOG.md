# UiPath LangChain Client Changelog

All notable changes to `uipath_langchain_client` will be documented in this file.

## [1.9.9] - 2026-04-23

### Added
- `UiPathBaseLLMClient.model_details` — optional constructor field carrying the discovery `modelDetails` dict (`shouldSkipTemperature`, `maxOutputTokens`, etc.). If omitted, resolved lazily via `get_model_info()`. Pass explicitly to skip the discovery call at first use.
- `UiPathBaseLLMClient.resolved_model_details` / `_should_skip_sampling_params` cached properties — read `modelDetails.shouldSkipTemperature` with a name-based fallback (`is_claude_opus_4_or_above`) for models not found in discovery.
- `get_chat_model` / `get_embedding_model` now forward `modelDetails` from the discovery lookup into the constructed client's `model_details` kwarg (via `setdefault`, so an explicit user-provided override still wins). Eliminates a redundant discovery fetch inside `resolved_model_details`.

### Fixed
- Sampling params (`temperature`, `top_k`, `top_p`) are now stripped centrally in `UiPathBaseChatModel` for models whose discovery marks `shouldSkipTemperature: True` (e.g. `anthropic.claude-opus-4-7`). Stripping happens at both sites: (1) a `model_validator` nulls the instance fields and discards them from `__pydantic_fields_set__` at construction time — so `UiPathChat(model="...", temperature=0.6)` no longer forwards the value; and (2) the `_generate` / `_agenerate` / `_stream` / `_astream` wrappers pop the same keys from invocation kwargs — so `llm.invoke("msg", temperature=0.7)` and `.bind(temperature=0.7)` are also handled. Previously each client had its own ad-hoc stripping path.
- `UiPathChatBedrockConverse._converse_params` defensively drops `None` `temperature` / `topP` from `inferenceConfig` so boto3 doesn't serialize them as explicit nulls to the wire.
- `UiPathChat._default_params` still drops `temperature` when `thinking` is set (Anthropic's extended thinking API requires `temperature=1`).

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

