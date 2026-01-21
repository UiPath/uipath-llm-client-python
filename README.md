# UiPath LLM Client

A Python client for interacting with UiPath's LLM services. This package provides both a low-level HTTP client and framework-specific integrations (LangChain, LlamaIndex) for accessing LLMs through UiPath's infrastructure.

## Architecture Overview

This repository is organized as a monorepo with the following packages:

- **`uipath_llm_client`** (root): Core HTTP client with authentication, retry logic, and request handling
- **`uipath_langchain_client`** (packages/): LangChain-compatible chat models and embeddings
- **`uipath_llamaindex_client`** (packages/): LlamaIndex-compatible integrations

### Supported Backends

The client supports two UiPath backends:

| Backend | Description | Default |
|---------|-------------|---------|
| **AgentHub** | UiPath's AgentHub infrastructure with automatic CLI-based authentication | Yes |
| **LLMGateway** | UiPath's LLM Gateway with S2S authentication | No |

### Supported Providers

| Provider | Chat Models | Embeddings | Vendor Type |
|----------|-------------|------------|-------------|
| OpenAI/Azure | GPT-4o, GPT-4, etc. | text-embedding-3-large/small | `openai` |
| Google | Gemini 2.5, Gemini 2.0, etc. | text-embedding-004 | `vertexai` |
| Anthropic | Claude Sonnet 4.5, etc. | - | `awsbedrock`, `vertexai` |
| AWS Bedrock | Claude models | None currently available | `awsbedrock` |

## Installation

### Using `pip`

```bash
# Base installation (core client only)
pip install uipath-llm-client

# With LangChain support
pip install uipath-langchain-client

# With specific provider extras for passthrough mode
pip install "uipath-langchain-client[openai]"      # OpenAI/Azure models
pip install "uipath-langchain-client[google]"      # Google Gemini models
pip install "uipath-langchain-client[anthropic]"   # Anthropic Claude models
pip install "uipath-langchain-client[all]"         # All providers
```

### Using `uv`

1. Add the custom index to your `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "uipath"
url = "https://uipath.pkgs.visualstudio.com/_packaging/ml-packages/pypi/simple/"
publish-url = "https://uipath.pkgs.visualstudio.com/_packaging/ml-packages/pypi/upload/"
```

2. Install the packages:

```bash
# Core client
uv add uipath-llm-client

# LangChain integration with all providers
uv add "uipath-langchain-client[all]"
```

## Configuration

### AgentHub Backend (Default)

The AgentHub backend uses the UiPath CLI for authentication. On first use, it will prompt you to log in via browser.

```bash
# Optional: Pre-authenticate via CLI
uv run uipath auth login

# Or set environment variables directly
export UIPATH_ENVIRONMENT="cloud"              # Environment: "cloud", "staging", or "alpha" (default: "cloud")
export UIPATH_URL="https://cloud.uipath.com"
export UIPATH_ORGANIZATION_ID="your-org-id"
export UIPATH_TENANT_ID="your-tenant-id"
export UIPATH_ACCESS_TOKEN="your-access-token"  # Optional if using CLI auth

# For S2S authentication (alternative to CLI)
export UIPATH_CLIENT_ID="your-client-id"
export UIPATH_CLIENT_SECRET="your-client-secret"
export UIPATH_CLIENT_SCOPE="your-scope"         # Optional: custom OAuth scope
```

### LLMGateway Backend

To use the LLMGateway backend, set the following environment variables:

```bash
# Select the backend
export UIPATH_LLM_BACKEND="llmgateway"

# Required configuration
export LLMGW_URL="https://your-llmgw-url.com"
export LLMGW_SEMANTIC_ORG_ID="your-org-id"
export LLMGW_SEMANTIC_TENANT_ID="your-tenant-id"
export LLMGW_REQUESTING_PRODUCT="your-product-name"
export LLMGW_REQUESTING_FEATURE="your-feature-name"

# Authentication (choose one)
export LLMGW_ACCESS_TOKEN="your-access-token"
# OR for S2S authentication:
export LLMGW_CLIENT_ID="your-client-id"
export LLMGW_CLIENT_SECRET="your-client-secret"

# Optional tracking
export LLMGW_SEMANTIC_USER_ID="your-user-id"
```

## Settings Reference

### AgentHubSettings

Configuration settings for UiPath AgentHub client requests. These settings control routing, authentication, and tracking for requests to AgentHub.

```python
from uipath_llm_client.settings import AgentHubSettings

settings = AgentHubSettings(
    environment="cloud",        # UiPath environment
    access_token="...",         # Optional: pre-set access token
    base_url="...",             # Optional: custom base URL
    tenant_id="...",            # Optional: tenant ID
    organization_id="...",      # Optional: organization ID
)
```

| Attribute | Environment Variable | Type | Default | Description |
|-----------|---------------------|------|---------|-------------|
| `environment` | `UIPATH_ENVIRONMENT` | `"cloud"` \| `"staging"` \| `"alpha"` | `"cloud"` | The UiPath environment to connect to |
| `access_token` | `UIPATH_ACCESS_TOKEN` | `SecretStr \| None` | `None` | Access token for authentication (auto-populated via CLI if not set) |
| `base_url` | `UIPATH_URL` | `str \| None` | `None` | Base URL of the AgentHub API (auto-populated via CLI if not set) |
| `tenant_id` | `UIPATH_TENANT_ID` | `str \| None` | `None` | Tenant ID for request routing (auto-populated via CLI if not set) |
| `organization_id` | `UIPATH_ORGANIZATION_ID` | `str \| None` | `None` | Organization ID for request routing (auto-populated via CLI if not set) |
| `client_id` | `UIPATH_CLIENT_ID` | `SecretStr \| None` | `None` | Client ID for OAuth/S2S authentication |
| `client_secret` | `UIPATH_CLIENT_SECRET` | `SecretStr \| None` | `None` | Client secret for OAuth/S2S authentication |
| `client_scope` | `UIPATH_CLIENT_SCOPE` | `str \| None` | `None` | Custom OAuth scope for authentication |
| `agenthub_config` | `UIPATH_AGENTHUB_CONFIG` | `str \| None` | `None` | AgentHub configuration for tracing |
| `process_key` | `UIPATH_PROCESS_KEY` | `str \| None` | `None` | Process key for tracing |
| `job_key` | `UIPATH_JOB_KEY` | `str \| None` | `None` | Job key for tracing |

**Authentication behavior:**
- If `access_token`, `base_url`, `tenant_id`, and `organization_id` are all provided, they are used directly
- Otherwise, the client uses the UiPath CLI (`uipath auth`) to authenticate automatically
- For S2S authentication, provide `client_id` and `client_secret`

### LLMGatewaySettings

Configuration settings for LLM Gateway client requests. These settings control routing, authentication, and tracking for requests to LLM Gateway.

```python
from uipath_llm_client.settings import LLMGatewaySettings

settings = LLMGatewaySettings(
    base_url="https://your-llmgw-url.com",
    org_id="your-org-id",
    tenant_id="your-tenant-id",
    requesting_product="your-product",
    requesting_feature="your-feature",
    client_id="your-client-id",           # For S2S auth
    client_secret="your-client-secret",   # For S2S auth
)
```

| Attribute | Environment Variable | Type | Required | Description |
|-----------|---------------------|------|----------|-------------|
| `base_url` | `LLMGW_URL` | `str` | Yes | Base URL of the LLM Gateway |
| `org_id` | `LLMGW_SEMANTIC_ORG_ID` | `str` | Yes | Organization ID for request routing |
| `tenant_id` | `LLMGW_SEMANTIC_TENANT_ID` | `str` | Yes | Tenant ID for request routing |
| `requesting_product` | `LLMGW_REQUESTING_PRODUCT` | `str` | Yes | Product name making the request (for tracking) |
| `requesting_feature` | `LLMGW_REQUESTING_FEATURE` | `str` | Yes | Feature name making the request (for tracking) |
| `access_token` | `LLMGW_ACCESS_TOKEN` | `SecretStr \| None` | Conditional | Access token for authentication |
| `client_id` | `LLMGW_CLIENT_ID` | `SecretStr \| None` | Conditional | Client ID for S2S authentication |
| `client_secret` | `LLMGW_CLIENT_SECRET` | `SecretStr \| None` | Conditional | Client secret for S2S authentication |
| `user_id` | `LLMGW_SEMANTIC_USER_ID` | `str \| None` | No | User ID for tracking and billing |
| `action_id` | `LLMGW_ACTION_ID` | `str \| None` | No | Action ID for tracking |
| `additional_headers` | `LLMGW_ADDITIONAL_HEADERS` | `Mapping[str, str]` | No | Additional custom headers to include in requests |

**Authentication behavior:**
- Either `access_token` OR both `client_id` and `client_secret` must be provided
- S2S authentication uses `client_id`/`client_secret` to obtain tokens automatically

## Usage Examples

### Quick Start with Direct Client Classes

The simplest way to get started - settings are automatically loaded from environment variables:

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI

# No settings needed - uses defaults from environment (AgentHub backend)
chat = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")
response = chat.invoke("What is the capital of France?")
print(response.content)
```

### Using Different Providers

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI
from uipath_langchain_client.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.openai.embeddings import UiPathAzureOpenAIEmbeddings

# OpenAI/Azure models
openai_chat = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")
response = openai_chat.invoke("Hello!")
print(response.content)

# Google Gemini models
gemini_chat = UiPathChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = gemini_chat.invoke("Hello!")
print(response.content)

# Anthropic Claude models (via AWS Bedrock)
claude_chat = UiPathChatAnthropic(model="anthropic.claude-sonnet-4-5-20250929-v1:0", vendor_type="awsbedrock")
response = claude_chat.invoke("Hello!")
print(response.content)

# Embeddings
embeddings = UiPathAzureOpenAIEmbeddings(model="text-embedding-3-large")
vectors = embeddings.embed_documents(["Hello world", "How are you?"])
print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}")
```

### Using Factory Functions (Auto-Detect Vendor)

Factory functions automatically detect the model vendor but require settings to be passed:

```python
from uipath_langchain_client import get_chat_model, get_embedding_model
from uipath_llm_client.settings import get_default_client_settings

settings = get_default_client_settings()

# Create a chat model - vendor is auto-detected from model name
chat_model = get_chat_model(model_name="gpt-4o-2024-11-20", client_settings=settings)
response = chat_model.invoke("What is the capital of France?")
print(response.content)

# Create an embeddings model
embeddings_model = get_embedding_model(model="text-embedding-3-large", client_settings=settings)
vectors = embeddings_model.embed_documents(["Hello world", "How are you?"])
```

### Using the Normalized API (Provider-Agnostic)

The normalized API provides a consistent interface across all LLM providers:

```python
from uipath_langchain_client import get_chat_model
from uipath_llm_client.settings import get_default_client_settings

settings = get_default_client_settings()

# Use normalized API for provider-agnostic calls
chat_model = get_chat_model(
    model_name="gpt-4o-2024-11-20",
    client_settings=settings,
    client_type="normalized",
)

# Works the same way regardless of the underlying provider
response = chat_model.invoke("Explain quantum computing in simple terms.")
print(response.content)
```

### Streaming Responses

All chat models support streaming for real-time output:

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI

chat_model = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")

for chunk in chat_model.stream("Write a short poem about coding."):
    print(chunk.content, end="", flush=True)
print()
```

### Async Operations

For async/await support:

```python
import asyncio
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI

async def main():
    chat_model = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")
    
    # Async invoke
    response = await chat_model.ainvoke("What is 2 + 2?")
    print(response.content)
    
    # Async streaming
    async for chunk in chat_model.astream("Tell me a joke."):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(main())
```

### Tool/Function Calling

Use tools with LangChain's standard interface:

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72°F."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

chat_model = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")

# Bind tools to the model
model_with_tools = chat_model.bind_tools([get_weather, calculate])

# The model can now use tools
response = model_with_tools.invoke("What's the weather in Paris?")
print(response.tool_calls)
```

### Using with LangChain Agents

Integrate with LangChain's agent framework:

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

chat_model = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")
agent = create_react_agent(chat_model, [search])

# Run the agent
result = agent.invoke({"messages": [("user", "Search for Python tutorials")]})
print(result["messages"][-1].content)
```

### Low-Level HTTP Client

For advanced use cases, use the low-level client directly:

```python
from uipath_llm_client import UiPathBaseLLMClient, UiPathAPIConfig

# Create a low-level client (settings auto-loaded from environment)
client = UiPathBaseLLMClient(
    model="gpt-4o-2024-11-20",
    api_config=UiPathAPIConfig(
        api_type="completions",
        client_type="passthrough",
        vendor_type="openai",
    ),
)

# Make a raw request
response = client.uipath_request(
    request_body={
        "model": "gpt-4o-2024-11-20",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
    }
)
print(response.json())
```

### Custom Configuration

Pass custom settings when you need more control:

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI
from uipath_llm_client.settings import AgentHubSettings
from uipath_llm_client.utils.retry import RetryConfig

# Custom settings for AgentHub
settings = AgentHubSettings(environment="cloud")  # or "staging", "alpha"

# With retry configuration
retry_config: RetryConfig = {
    "initial_delay": 2.0,
    "max_delay": 60.0,
    "exp_base": 2.0,
    "jitter": 1.0,
}

chat_model = UiPathAzureChatOpenAI(
    model="gpt-4o-2024-11-20",
    client_settings=settings,
    max_retries=3,
    retry_config=retry_config,
)
```

### Switching Between Backends

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI
from uipath_llm_client.settings import get_default_client_settings

# Explicitly specify the backend
agenthub_settings = get_default_client_settings(backend="agenthub")
llmgw_settings = get_default_client_settings(backend="llmgateway")

chat = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20", client_settings=llmgw_settings)

# Or use environment variable (no code changes needed)
# export UIPATH_LLM_BACKEND="llmgateway"
```

### Using LLMGatewaySettings Directly

You can instantiate `LLMGatewaySettings` directly for full control over configuration:

**With Direct Client Classes:**

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI
from uipath_langchain_client.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.openai.embeddings import UiPathAzureOpenAIEmbeddings
from uipath_llm_client.settings import LLMGatewaySettings

# Create LLMGatewaySettings with explicit configuration
settings = LLMGatewaySettings(
    base_url="https://your-llmgw-url.com",
    org_id="your-org-id",
    tenant_id="your-tenant-id",
    requesting_product="my-product",
    requesting_feature="my-feature",
    client_id="your-client-id",
    client_secret="your-client-secret",
    user_id="optional-user-id",  # Optional: for tracking
)

# Use with OpenAI/Azure chat model
openai_chat = UiPathAzureChatOpenAI(
    model="gpt-4o-2024-11-20",
    client_settings=settings,
)
response = openai_chat.invoke("Hello!")
print(response.content)

# Use with Google Gemini
gemini_chat = UiPathChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    client_settings=settings,
)
response = gemini_chat.invoke("Hello!")
print(response.content)

# Use with embeddings
embeddings = UiPathAzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    client_settings=settings,
)
vectors = embeddings.embed_documents(["Hello world"])
```

**With Factory Methods:**

```python
from uipath_langchain_client import get_chat_model, get_embedding_model
from uipath_llm_client.settings import LLMGatewaySettings

# Create LLMGatewaySettings
settings = LLMGatewaySettings(
    base_url="https://your-llmgw-url.com",
    org_id="your-org-id",
    tenant_id="your-tenant-id",
    requesting_product="my-product",
    requesting_feature="my-feature",
    client_id="your-client-id",
    client_secret="your-client-secret",
)

# Factory auto-detects vendor from model name
chat_model = get_chat_model(
    model_name="gpt-4o-2024-11-20",
    client_settings=settings,
)
response = chat_model.invoke("What is the capital of France?")
print(response.content)

# Use normalized API for provider-agnostic interface
normalized_chat = get_chat_model(
    model_name="gemini-2.5-flash",
    client_settings=settings,
    client_type="normalized",
)
response = normalized_chat.invoke("Explain quantum computing.")
print(response.content)

# Embeddings with factory
embeddings = get_embedding_model(
    model="text-embedding-3-large",
    client_settings=settings,
)
vectors = embeddings.embed_documents(["Hello", "World"])
```

### Bring Your Own (BYO) Model Connections

If you have enrolled your own model deployment into UiPath's LLMGateway, you can use it by providing your BYO connection ID. This allows you to route requests through LLMGateway to your custom-enrolled models.

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI

# Use your BYO connection ID from LLMGateway enrollment
chat = UiPathAzureChatOpenAI(
    model="your-custom-model-name",
    byo_connection_id="your-byo-connection-id",  # UUID from LLMGateway enrollment
)

response = chat.invoke("Hello from my custom model!")
print(response.content)
```

This works with any client class:

```python
from uipath_langchain_client.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.openai.embeddings import UiPathAzureOpenAIEmbeddings

# BYO chat model
byo_chat = UiPathChatGoogleGenerativeAI(
    model="my-custom-gemini",
    byo_connection_id="f1d29b49-0c7b-4c01-8bc4-fc1b7d918a87",
)

# BYO embeddings model
byo_embeddings = UiPathAzureOpenAIEmbeddings(
    model="my-custom-embeddings",
    byo_connection_id="a2e38c51-1d8a-5e02-9cd5-ge2c8e029b98",
)
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/UiPath/uipath-llm-client.git
cd uipath-llm-client
uv sync --dev

# Run tests
uv run pytest

# Format and lint
uv run ruff format .
uv run ruff check .
uv run pyright
```

### Testing

Tests use [VCR.py](https://vcrpy.readthedocs.io/) to record and replay HTTP interactions. Cassettes (recorded responses) are stored in `tests/cassettes/` using Git LFS.

**Important:** Tests must pass locally before submitting a PR. The CI pipeline does not make any real API requests—it only runs tests using the pre-recorded cassettes.

**Prerequisites:**
- Install [Git LFS](https://git-lfs.com/): `brew install git-lfs` (macOS) or `apt install git-lfs` (Ubuntu)
- Initialize Git LFS: `git lfs install`
- Pull cassettes: `git lfs pull`

**Running tests locally:**

```bash
# Run all tests using cassettes (no API credentials required)
uv run pytest

# Run specific test files
uv run pytest tests/langchain/
uv run pytest tests/core/
```

**Updating cassettes:**

When adding new tests or modifying existing ones that require new API interactions:

1. Set up your environment with valid credentials (see [Configuration](#configuration))
2. Run the tests—VCR will record new interactions automatically
3. Commit the updated cassettes along with your code changes

**Note:** The CI pipeline validates that all tests pass using the committed cassettes. If your tests require new API calls, you must record and commit the corresponding cassettes for the pipeline to pass.

## Project Structure

```
uipath-llm-client/
├── src/uipath_llm_client/          # Core HTTP client
│   ├── client.py                   # UiPathBaseLLMClient base class
│   ├── settings/                   # Backend-specific settings
│   │   ├── agenthub/              # AgentHub authentication
│   │   └── llmgateway/            # LLMGateway authentication
│   └── utils/                      # Error handling, retry, logging
├── packages/
│   ├── uipath_langchain_client/   # LangChain integration
│   │   └── src/uipath_langchain_client/
│   │       ├── factory.py         # Auto-detection factory functions
│   │       ├── normalized/        # Provider-agnostic API
│   │       ├── openai/            # OpenAI/Azure passthrough
│   │       ├── google/            # Google Gemini passthrough
│   │       ├── anthropic/         # Anthropic passthrough
│   │       └── ...
│   └── uipath_llamaindex_client/  # LlamaIndex integration
└── tests/                          # Test suite with VCR cassettes
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact the maintainers at [UiPath GitHub Repository](https://github.com/UiPath/uipath-llm-client).
