# UiPath LangChain Client Changelog

All notable changes to `uipath_langchain_client` will be documented in this file.

## [1.0.0] - 2025-01-30

### Official Release
- First stable release of the UiPath LangChain Client
- API considered stable; semantic versioning will be followed from this point forward

### Highlights
- Production-ready LangChain integrations for all major LLM providers
- Factory functions for automatic vendor detection and model instantiation
- Full compatibility with LangChain agents, tools, and chains
- Comprehensive support for chat completions, embeddings, and streaming
- Seamless integration with both AgentHub and LLMGateway backends

## [0.3.x] - 2025-01-29

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

## [0.2.x] - 2025-01-15

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
- `UiPathNormalizedChatModel` - Provider-agnostic normalized API

### Embeddings Classes
- `UiPathOpenAIEmbeddings` - OpenAI embeddings via direct API
- `UiPathAzureOpenAIEmbeddings` - OpenAI embeddings via Azure
- `UiPathGoogleGenerativeAIEmbeddings` - Google embeddings
- `UiPathBedrockEmbeddings` - AWS Bedrock embeddings
- `UiPathAzureAIEmbeddingsModel` - Azure AI embeddings
- `UiPathNormalizedEmbeddings` - Provider-agnostic normalized API

### Features
- Support for BYO (Bring Your Own) model connections

### Breaking Changes
- Package renamed from internal module to `uipath_langchain_client`
- Import paths changed; update imports accordingly

## [0.1.x] - 2024-12-20

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
