# UiPath LangChain Client Changelog

All notable changes to `uipath_langchain_client` will be documented in this file.

## [1.3.0] - 2026-03-10

### Version Bump
- Stable release
- Fixes

## [1.2.7] - 2026-02-26

### Fix
- Fix Bedrock clients model_id

## [1.2.6] - 2026-02-26

### Refactor
- Restructured project such that uipath_llm_client can be included in uipath as submodule.
- imports are now from uipath.llm_client instead of uipath_llm_client

## [1.2.5] - 2026-02-26

### Fix
- Parameters on factory fix

## [1.2.4] - 2026-02-26

### Fix
- Fix typing on factory method

## [1.2.3] - 2026-02-25

### Feature
- Capture headers and inject them in response_metadata.

## [1.2.2] - 2026-02-23

### Fix
- Fixes to discovery endpoint on LLMGW

## [1.2.1] - 2026-02-18

### Fix
- TImeout fixes, change typing from int to float
- remove timeout=None from all clients -> caused overriding the default timeout set up on the UiPathHttpxClient

## [1.2.0] - 2026-02-18

### Stable release

## [1.1.9] - 2026-02-13

### Docs
- Updated documentation

## [1.1.8] - 2026-02-13

### Refactor
- Adjust Anthropic factory method to use ChatAnthropic instead of ChatAnthropicVertex

## [1.1.7] - 2026-02-13

### Refactor
- Re-export UiPathChat and UiPathEmbeddings at module level

## [1.1.6] - 2026-02-12

### Fixes
- Added proper type hints for factory method

## [1.1.5] - 2026-02-12

### Fixes
- Fixed bedrock converse api

## [1.1.4] - 2026-02-12

### Fixes
- Fixed anthropic default vendor

## [1.1.3] - 2026-02-12

### Fixes
- Fixes on openai langchain client on resposes_api
- Allow the flavor to be set up at requst time, not just when instantiating the llm
- Some fixes for the anthropic client

## [1.1.2] - 2026-02-12

### Refactor
- Rename normalized client for better comaptibility with other packages

## [1.1.1] - 2026-02-11

### Fixes
- Fix langchain fireworks client for async usage

## [1.1.0] - 2026-02-11

### Features
- Added langchain fireworks client and tested with GLM

### Stable release
- Fixed BYO on passthrough
- Stable release

## [1.0.13] - 2026-02-05

### Fix
- Bump version

## [1.0.12] - 2026-02-05

### Fix
- Added 295 as default llmgateway timeout to avoid problems on the backend side

## [1.0.11] - 2026-02-04

### Type fix
- Import TypedDict from typing_extension

## [1.0.10] - 2026-02-04

### Version Upgrade
- Updated version

## [1.0.9] - 2026-02-04

### Fix
- Fixed typing in core package, updated dependency

## [1.0.8] - 2026-02-04

### Fix
- Added py.typed to the package

## [1.0.7] - 2026-02-04

### Refactor
- Refactor factory function to include byom models

## [1.0.6] - 2026-02-03

### Refactor
- Updated documentation to include the new aliases for settings
- New alias for settings and request timeout in BaseLLMClient

## [1.0.5] - 2026-02-03

### Bug Fix
- Fixed retry logic on all clients

## [1.0.4] - 2026-02-03

### Bug Fix
- Fix some timout issues on langchain_openai from llmgw.

## [1.0.3] - 2026-02-02

### Bug Fix
- Added better dependencies for langchain-anthropic to include boto and vertex

## [1.0.2] - 2026-02-02

### Bug Fix
- Removed old fix on Gemini streaming and updated with a new cleaner one

## [1.0.1] - 2026-02-02

### Bug Fix
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
