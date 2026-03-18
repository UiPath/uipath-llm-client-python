# UiPath LLM Client Changelog

All notable changes to `uipath_llm_client` (core package) will be documented in this file.

## [1.5.3] - 2026-03-18

### Fix
- Factory function fix

## [1.5.2] - 2026-03-18

### Fix
- Factory function fix

## [1.5.1] - 2026-03-17

### Fix
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

### Fixes
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

### Fix
- Fixes to discovery endpoint on LLMGW

## [1.2.1] - 2026-02-18

### Fix
- TImeout fixes, change typing from int to float
- remove timeout=None from all clients -> caused overriding the default timeout set up on the UiPathHttpxClient

## [1.2.0] - 2026-02-18

### Stable release

### Fix
- Fixed agenhub auth when token already exists

## [1.1.1] - 2026-02-12

### Fix
- Small fixes on openai client

## [1.1.0] - 2026-02-11

### Stable release
- Adeed BYOM validation for settings
- Stable release

## [1.0.13] - 2026-02-05

### Fix
- Fixed headers on llmgw settings

## [1.0.12] - 2026-02-05

### Fix
- Added 295 as default llmgateway timeout to avoid problems on the backend side

## [1.0.11] - 2026-02-05

### Feature
- Added retry handler on 429 to include the retry-after header

## [1.0.10] - 2026-02-04

### Type fix
- Import TypedDict from typing_extension

### Typing Fix
- import @override from typing_extension

## [1.0.8] - 2026-02-04

### Bug Fix
- Fixed a typing issue of Singleton

## [1.0.7] - 2026-02-04

### Fix
- Added py.typed to the package

## [1.0.6]

### Bug Fix
- Fixed model discovery on AgentHub Settings.

## [1.0.5] - 2026-02-03

### Refactor
- Updated documentation to include the new aliases for settings

## [1.0.4] - 2026-02-03

### Bug Fix
- Adjusted retry logic, now 0 means no retries, 1 means one retry

## [1.0.3] - 2026-02-02

### Refactor
- moved the logic of get_httpx_ssl_client_kwargs from the uipath package to this package; 

## [1.0.2] - 2026-02-02

### Bug Fixes
- Fixed endpoints on AgentHub Settings

## [1.0.1] - 2026-01-30

### Bug Fixes
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
