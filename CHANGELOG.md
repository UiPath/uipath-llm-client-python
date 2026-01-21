# UiPath LLM Client Changelog

All notable changes to `uipath_llm_client` (core package) will be documented in this file.

## [1.0.0] - 2025-01-30

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

## [0.3.x] - 2025-01-29

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

## [0.2.x] - 2025-01-15

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

## [0.1.x] - 2024-12-20

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
