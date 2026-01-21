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
pip install "uipath-langchain-client[aws]"         # AWS Bedrock models
pip install "uipath-langchain-client[vertexai]"    # Google VertexAI models
pip install "uipath-langchain-client[all]"         # All providers
```

## Quick Start

### Using Factory Functions (Recommended)

The factory functions automatically detect the model vendor and return the appropriate client:

```python
from uipath_langchain_client import get_chat_model, get_embedding_model
from uipath_langchain_client.settings import get_default_client_settings

# Get default settings (uses UIPATH_LLM_BACKEND env var or defaults to AgentHub)
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
    model="text-embedding-3-large",
    client_settings=settings,
)
vectors = embeddings.embed_documents(["Hello world"])
print(f"Embedding dimension: {len(vectors[0])}")
```

### Using Direct Client Classes

For more control, instantiate provider-specific classes directly:

```python
from uipath_langchain_client.openai.chat_models import UiPathAzureChatOpenAI
from uipath_langchain_client.google.chat_models import UiPathChatGoogleGenerativeAI
from uipath_langchain_client.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.normalized.chat_models import UiPathNormalizedChatModel
from uipath_langchain_client.settings import get_default_client_settings

settings = get_default_client_settings()

# OpenAI/Azure
openai_chat = UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20", client_settings=settings)

# Google Gemini
gemini_chat = UiPathChatGoogleGenerativeAI(model="gemini-2.5-flash", client_settings=settings)

# Anthropic Claude (via AWS Bedrock)
claude_chat = UiPathChatAnthropic(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    client_settings=settings,
    vendor_type="awsbedrock",
)

# Normalized (provider-agnostic)
normalized_chat = UiPathNormalizedChatModel(model="gpt-4o-2024-11-20", client_settings=settings)
```

## Available Client Types

### Passthrough Mode (Default)

Uses vendor-specific APIs through UiPath's gateway. Full feature parity with native SDKs.

| Class | Provider | Models |
|-------|----------|--------|
| `UiPathAzureChatOpenAI` | OpenAI/Azure | GPT-4o, GPT-4, GPT-3.5 |
| `UiPathChatOpenAI` | OpenAI | GPT-4o, GPT-4, GPT-3.5 |
| `UiPathChatGoogleGenerativeAI` | Google | Gemini 2.5, 2.0, 1.5 |
| `UiPathChatAnthropic` | Anthropic | Claude Sonnet 4.5, Opus, etc. |
| `UiPathChatAnthropicVertex` | Anthropic (via VertexAI) | Claude models |
| `UiPathAzureAIChatCompletionsModel` | Azure AI | Various |

### Normalized Mode

Uses UiPath's normalized API for a consistent interface across all providers.

| Class | Description |
|-------|-------------|
| `UiPathNormalizedChatModel` | Provider-agnostic chat completions |
| `UiPathNormalizedEmbeddings` | Provider-agnostic embeddings |

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

Factory function to create a chat model.

**Parameters:**
- `model_name` (str): Name of the model (e.g., "gpt-4o-2024-11-20")
- `client_settings` (UiPathBaseSettings): Client settings for authentication
- `client_type` (Literal["passthrough", "normalized"]): API mode (default: "passthrough")
- `**model_kwargs`: Additional arguments passed to the model constructor

**Returns:** `BaseChatModel` - A LangChain-compatible chat model

### `get_embedding_model()`

Factory function to create an embeddings model.

**Parameters:**
- `model` (str): Name of the model (e.g., "text-embedding-3-large")
- `client_settings` (UiPathBaseSettings): Client settings for authentication
- `client_type` (Literal["passthrough", "normalized"]): API mode (default: "passthrough")
- `**model_kwargs`: Additional arguments passed to the model constructor

**Returns:** `Embeddings` - A LangChain-compatible embeddings model

## See Also

- [Main README](../../README.md) - Overview and core client documentation
- [UiPath LLM Client](../../src/uipath_llm_client/) - Low-level HTTP client
