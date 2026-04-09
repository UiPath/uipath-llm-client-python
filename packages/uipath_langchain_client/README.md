# UiPath LangChain Client

LangChain-compatible chat models and embeddings for accessing LLMs through UiPath's infrastructure.

## Installation

```bash
# Base installation (normalized API only)
pip install uipath-langchain-client

# With specific provider extras for passthrough mode
pip install "uipath-langchain-client[openai]"      # OpenAI/Azure models
pip install "uipath-langchain-client[google]"      # Google Gemini models
pip install "uipath-langchain-client[anthropic]"   # Anthropic Claude models
pip install "uipath-langchain-client[azure]"       # Azure AI models
pip install "uipath-langchain-client[bedrock]"      # AWS Bedrock models
pip install "uipath-langchain-client[vertexai]"    # Google VertexAI models
pip install "uipath-langchain-client[fireworks]"   # Fireworks AI models
pip install "uipath-langchain-client[all]"         # All providers
```

## Quick Start

### Using Factory Functions (Recommended)

The factory functions automatically detect the model vendor and return the appropriate client:

```python
from uipath_langchain_client import get_chat_model, get_embedding_model
from uipath_langchain_client.settings import get_default_client_settings

# Get default settings (uses UIPATH_LLM_SERVICE env var or defaults to AgentHub)
settings = get_default_client_settings()

# Chat model - vendor auto-detected from model name
chat_model = get_chat_model(
    model_name="gpt-4o-2024-11-20",
    client_settings=settings,
)
response = chat_model.invoke("Hello, how are you?")
print(response.content)

# Embeddings model
embeddings = get_embedding_model(
    model_name="text-embedding-3-large",
    client_settings=settings,
)
vectors = embeddings.embed_documents(["Hello world"])
print(f"Embedding dimension: {len(vectors[0])}")
```

### Using Direct Client Classes

For more control, instantiate provider-specific classes directly:

```python
from uipath_langchain_client.clients.openai.chat_models import UiPathAzureChatOpenAI
from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.settings import get_default_client_settings

settings = get_default_client_settings()

# OpenAI/Azure
openai_chat = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20", settings=settings)

# Google Gemini
gemini_chat = UiPathChatGoogleGenerativeAI(model="gemini-2.5-flash", settings=settings)

# Anthropic Claude (via AWS Bedrock)
claude_chat = UiPathChatAnthropic(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    settings=settings,
    vendor_type="awsbedrock",
)

# Normalized (provider-agnostic)
normalized_chat = UiPathChat(model="gpt-4o-2024-11-20", settings=settings)
```

## Available Client Types

### Passthrough Mode (Default)

Uses vendor-specific APIs through UiPath's gateway. Full feature parity with native SDKs.

**Chat Models:**

| Class | Provider | Extra | Models |
|-------|----------|-------|--------|
| `UiPathAzureChatOpenAI` | OpenAI/Azure (UiPath-owned) | `[openai]` | GPT-4o, GPT-4, o1, o3, etc. |
| `UiPathChatOpenAI` | OpenAI (BYO) | `[openai]` | GPT-4o, GPT-4, etc. |
| `UiPathChatGoogleGenerativeAI` | Google | `[google]` | Gemini 2.5, 2.0, 1.5 |
| `UiPathChatAnthropic` | Anthropic (via Bedrock) | `[anthropic]` | Claude Sonnet 4.5, Opus, etc. |
| `UiPathChatAnthropicVertex` | Anthropic (via VertexAI) | `[vertexai]` | Claude models |
| `UiPathChatBedrock` | AWS Bedrock (invoke API) | `[bedrock]` | Bedrock-hosted models |
| `UiPathChatBedrockConverse` | AWS Bedrock (Converse API) | `[bedrock]` | Bedrock-hosted models |
| `UiPathChatFireworks` | Fireworks AI | `[fireworks]` | Various open-source models |
| `UiPathAzureAIChatCompletionsModel` | Azure AI | `[azure]` | Various Azure AI models |

**Embeddings:**

| Class | Provider | Extra | Models |
|-------|----------|-------|--------|
| `UiPathAzureOpenAIEmbeddings` | OpenAI/Azure (UiPath-owned) | `[openai]` | text-embedding-3-large/small |
| `UiPathOpenAIEmbeddings` | OpenAI (BYO) | `[openai]` | text-embedding-3-large/small |
| `UiPathGoogleGenerativeAIEmbeddings` | Google | `[google]` | text-embedding-004 |
| `UiPathBedrockEmbeddings` | AWS Bedrock | `[bedrock]` | Titan Embeddings, etc. |
| `UiPathFireworksEmbeddings` | Fireworks AI | `[fireworks]` | Various |
| `UiPathAzureAIEmbeddingsModel` | Azure AI | `[azure]` | Various Azure AI models |

### Normalized Mode

Uses UiPath's normalized API for a consistent interface across all providers. No extra dependencies required.

| Class | Type | Description |
|-------|------|-------------|
| `UiPathChat` | Chat | Provider-agnostic chat completions |
| `UiPathEmbeddings` | Embeddings | Provider-agnostic embeddings |

## Features

### Streaming

```python
from uipath_langchain_client import get_chat_model
from uipath_langchain_client.settings import get_default_client_settings

settings = get_default_client_settings()
chat_model = get_chat_model(model_name="gpt-4o-2024-11-20", client_settings=settings)

# Sync streaming
for chunk in chat_model.stream("Write a haiku about Python"):
    print(chunk.content, end="", flush=True)

# Async streaming
async for chunk in chat_model.astream("Write a haiku about Python"):
    print(chunk.content, end="", flush=True)
```

### Tool Calling

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72°F in {city}"

chat_model = get_chat_model(model_name="gpt-4o-2024-11-20", client_settings=settings)
model_with_tools = chat_model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in Tokyo?")
print(response.tool_calls)
```

### LangGraph Agents

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

chat_model = get_chat_model(model_name="gpt-4o-2024-11-20", client_settings=settings)
agent = create_react_agent(chat_model, [search])

result = agent.invoke({"messages": [("user", "Search for UiPath documentation")]})
```

### Extended Thinking (Model-Specific)

```python
# OpenAI o1/o3 reasoning
chat_model = get_chat_model(
    model_name="o3-mini",
    client_settings=settings,
    client_type="normalized",
    reasoning_effort="medium",  # "low", "medium", "high"
)

# Anthropic Claude thinking
chat_model = get_chat_model(
    model_name="claude-sonnet-4-5",
    client_settings=settings,
    client_type="normalized",
    thinking={"type": "enabled", "budget_tokens": 10000},
)

# Gemini thinking
chat_model = get_chat_model(
    model_name="gemini-2.5-pro",
    client_settings=settings,
    client_type="normalized",
    thinking_level="medium",
    include_thoughts=True,
)
```

## Configuration

### Retry Configuration

```python
# RetryConfig is a TypedDict - all fields are optional with sensible defaults
retry_config = {
    "initial_delay": 2.0,   # Initial delay before first retry
    "max_delay": 60.0,      # Maximum delay between retries
    "exp_base": 2.0,        # Exponential backoff base
    "jitter": 1.0,          # Random jitter to add
}

chat_model = get_chat_model(
    model_name="gpt-4o-2024-11-20",
    client_settings=settings,
    max_retries=3,
    retry_config=retry_config,
)
```

### Request Timeout

```python
chat_model = get_chat_model(
    model_name="gpt-4o-2024-11-20",
    client_settings=settings,
    request_timeout=120,  # Client-side timeout in seconds
)
```

## API Reference

### `get_chat_model()`

Factory function to create a chat model. Automatically detects the model vendor by querying UiPath's discovery endpoint and returns the appropriate LangChain model class.

**Parameters:**
- `model_name` (str): Name of the model (e.g., "gpt-4o-2024-11-20")
- `byo_connection_id` (str | None): Optional BYO connection ID for custom-enrolled models (default: None)
- `client_settings` (UiPathBaseSettings | None): Client settings for authentication (default: auto-detected)
- `client_type` (Literal["passthrough", "normalized"]): API mode (default: "passthrough")
- `**model_kwargs`: Additional arguments passed to the model constructor (e.g., `max_retries`, `retry_config`, `request_timeout`)

**Returns:** `UiPathBaseChatModel` - A LangChain-compatible chat model

**Raises:** `ValueError` - If the model is not found in available models or vendor is not supported

### `get_embedding_model()`

Factory function to create an embeddings model. Automatically detects the model vendor by querying UiPath's discovery endpoint and returns the appropriate LangChain embeddings class.

**Parameters:**
- `model_name` (str): Name of the embeddings model (e.g., "text-embedding-3-large")
- `byo_connection_id` (str | None): Optional BYO connection ID for custom-enrolled models (default: None)
- `client_settings` (UiPathBaseSettings | None): Client settings for authentication (default: auto-detected)
- `client_type` (Literal["passthrough", "normalized"]): API mode (default: "passthrough")
- `**model_kwargs`: Additional arguments passed to the embeddings constructor (e.g., `max_retries`, `retry_config`, `request_timeout`)

**Returns:** `UiPathBaseEmbeddings` - A LangChain-compatible embeddings model

**Raises:** `ValueError` - If the model is not found or the vendor is not supported

## UiPathChat Parameter Reference

The normalized `UiPathChat` model supports the following parameters:

### Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` (alias: `model_name`) | `str` | Required | Model identifier (e.g., `"gpt-4o-2024-11-20"`, `"gemini-2.5-flash"`) |
| `max_tokens` | `int \| None` | `None` | Maximum number of tokens in the response |
| `temperature` | `float \| None` | `None` | Sampling temperature (0.0 to 2.0) |
| `stop` (alias: `stop_sequences`) | `list[str] \| str \| None` | `None` | Stop sequences to end generation |
| `n` | `int \| None` | `None` | Number of completions to generate |
| `top_p` | `float \| None` | `None` | Nucleus sampling probability mass |
| `presence_penalty` | `float \| None` | `None` | Penalty for repeated tokens (-2.0 to 2.0) |
| `frequency_penalty` | `float \| None` | `None` | Frequency-based repetition penalty (-2.0 to 2.0) |
| `verbosity` | `str \| None` | `None` | Response verbosity: `"low"`, `"medium"`, or `"high"` |
| `model_kwargs` | `dict[str, Any]` | `{}` | Additional model-specific parameters |
| `disabled_params` | `dict[str, Any] \| None` | `None` | Parameters to exclude from requests |

### Extended Thinking Parameters

| Parameter | Provider | Type | Description |
|-----------|----------|------|-------------|
| `reasoning` | OpenAI (o1/o3) | `dict[str, Any] \| None` | Reasoning config, e.g., `{"effort": "medium", "summary": "auto"}` |
| `reasoning_effort` | OpenAI (o1/o3) | `str \| None` | Shorthand: `"minimal"`, `"low"`, `"medium"`, or `"high"` |
| `thinking` | Anthropic Claude | `dict[str, Any] \| None` | Thinking config, e.g., `{"type": "enabled", "budget_tokens": 10000}` |
| `thinking_level` | Google Gemini | `str \| None` | Thinking depth level |
| `thinking_budget` | Google Gemini | `int \| None` | Token budget for thinking |
| `include_thoughts` | Google Gemini | `bool \| None` | Whether to include thinking in responses |

### Base Client Parameters (All Models)

All LangChain model classes (`UiPathChat`, `UiPathAzureChatOpenAI`, etc.) inherit these from `UiPathBaseLLMClient`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` (alias: `model_name`) | `str` | Required | Model identifier |
| `settings` (alias: `client_settings`) | `UiPathBaseSettings` | Auto-detected | Client settings for auth and routing |
| `byo_connection_id` | `str \| None` | `None` | BYO connection ID for custom-enrolled models |
| `request_timeout` (aliases: `timeout`, `default_request_timeout`) | `int \| None` | `None` | Client-side request timeout in seconds |
| `max_retries` | `int` | `0` | Maximum number of retries for failed requests |
| `retry_config` | `RetryConfig \| None` | `None` | Retry configuration for failed requests |
| `logger` | `logging.Logger \| None` | `None` | Logger instance for request/response logging |
| `default_headers` | `Mapping[str, str] \| None` | See note | Additional request headers (see [Default Headers](../../README.md#default-headers)) |

### Low-Level Methods

`UiPathBaseLLMClient` also exposes these methods for advanced use cases:

| Method | Description |
|--------|-------------|
| `uipath_request(method, url, *, request_body, **kwargs)` | Synchronous HTTP request, returns `httpx.Response` |
| `uipath_arequest(method, url, *, request_body, **kwargs)` | Asynchronous HTTP request, returns `httpx.Response` |
| `uipath_stream(method, url, *, request_body, stream_type, **kwargs)` | Synchronous streaming, yields `str \| bytes` |
| `uipath_astream(method, url, *, request_body, stream_type, **kwargs)` | Asynchronous streaming, yields `str \| bytes` |

The `stream_type` parameter controls iteration: `"lines"` (default, best for SSE), `"text"`, `"bytes"`, or `"raw"`.

## See Also

- [Main README](../../README.md) - Overview and core client documentation
- [UiPath LLM Client](../../src/uipath/llm_client//) - Low-level HTTP client
