# UiPath LlamaIndex Client

LlamaIndex-compatible LLMs and embeddings for accessing models through UiPath's infrastructure.

> **Note:** This package is currently under development. Full LlamaIndex integration is coming soon.

## Planned Features

- LlamaIndex-compatible chat models (CustomLLM)
- LlamaIndex-compatible embeddings (BaseEmbedding)
- Support for all UiPath-supported providers (OpenAI, Google, Anthropic)
- Both AgentHub and LLMGateway backend support

## Installation

```bash
# Coming soon
pip install uipath-llamaindex-client
```

## Usage (Coming Soon)

```python
from uipath_llamaindex_client import get_llm, get_embeddings
from uipath_langchain_client.settings import get_default_client_settings

settings = get_default_client_settings()

# Create a LlamaIndex LLM
llm = get_llm(
    model_name="gpt-4o-2024-11-20",
    client_settings=settings,
)

# Create embeddings
embeddings = get_embeddings(
    model_name="text-embedding-3-large",
    client_settings=settings,
)

# Use with LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embeddings,
)

query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(response)
```

## See Also

- [Main README](../../README.md) - Overview and core client documentation
- [UiPath LangChain Client](../uipath_langchain_client/) - LangChain integration (available now)
