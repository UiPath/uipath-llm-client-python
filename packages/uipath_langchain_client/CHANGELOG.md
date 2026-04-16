# UiPath LangChain Client Changelog

All notable changes to `uipath_langchain_client` will be documented in this file.

## [1.8.4] - 2026-04-16

### Changed
- Factory functions (`get_chat_model`, `get_embedding_model`) now use the shared `get_model_info()` utility from the core package instead of an inline implementation

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
