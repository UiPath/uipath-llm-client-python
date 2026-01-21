"""
UiPath LangChain Client Demo

This script demonstrates various ways to use the UiPath LangChain integration
for chat completions, embeddings, streaming, tool calling, and agents.

Prerequisites:
    - Set up authentication (either AgentHub CLI or environment variables)
    - Install required extras: uv add "uipath-langchain-client[all]"

Usage:
    uv run python demo.py
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from uipath_langchain_client import get_chat_model, get_embedding_model
from uipath_langchain_client.settings import get_default_client_settings


def demo_basic_chat():
    """Demonstrate basic chat completion using the factory function."""
    print("\n" + "=" * 60)
    print("Demo: Basic Chat Completion")
    print("=" * 60)

    # Get default settings (uses AgentHub by default)
    settings = get_default_client_settings()

    # Create a chat model using the factory function
    # The vendor is auto-detected from the model name
    chat_model = get_chat_model(
        model_name="gpt-4o-2024-11-20",
        client_settings=settings,
    )

    # Simple invocation
    response = chat_model.invoke("What is the capital of France?")
    print(f"Response: {response.content}")
    print(f"Token usage: {response.usage_metadata}")


def demo_chat_with_messages():
    """Demonstrate chat with system and human messages."""
    print("\n" + "=" * 60)
    print("Demo: Chat with System Message")
    print("=" * 60)

    settings = get_default_client_settings()
    chat_model = get_chat_model(
        model_name="gpt-4o-2024-11-20",
        client_settings=settings,
    )

    # Use structured messages
    messages = [
        SystemMessage(content="You are a helpful assistant that speaks like a pirate."),
        HumanMessage(content="Tell me about Python programming."),
    ]

    response = chat_model.invoke(messages)
    print(f"Response: {response.content}")


def demo_streaming():
    """Demonstrate streaming responses."""
    print("\n" + "=" * 60)
    print("Demo: Streaming Response")
    print("=" * 60)

    settings = get_default_client_settings()
    chat_model = get_chat_model(
        model_name="gpt-4o-2024-11-20",
        client_settings=settings,
    )

    print("Streaming: ", end="")
    for chunk in chat_model.stream("Write a short haiku about coding."):
        print(chunk.content, end="", flush=True)
    print()  # Newline at the end


async def demo_async_operations():
    """Demonstrate async invocation and streaming."""
    print("\n" + "=" * 60)
    print("Demo: Async Operations")
    print("=" * 60)

    settings = get_default_client_settings()
    chat_model = get_chat_model(
        model_name="gpt-4o-2024-11-20",
        client_settings=settings,
    )

    # Async invoke
    response = await chat_model.ainvoke("What is 2 + 2?")
    print(f"Async response: {response.content}")

    # Async streaming
    print("Async streaming: ", end="")
    async for chunk in chat_model.astream("Tell me a very short joke."):
        print(chunk.content, end="", flush=True)
    print()


def demo_tool_calling():
    """Demonstrate tool/function calling."""
    print("\n" + "=" * 60)
    print("Demo: Tool Calling")
    print("=" * 60)

    # Define tools using LangChain's @tool decorator
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city.

        Args:
            city: The name of the city to get weather for.
        """
        # In a real app, this would call a weather API
        weather_data = {
            "Paris": "Sunny, 22°C",
            "Tokyo": "Cloudy, 18°C",
            "New York": "Rainy, 15°C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: A mathematical expression to evaluate (e.g., "2 + 2").
        """
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    settings = get_default_client_settings()
    chat_model = get_chat_model(
        model_name="gpt-4o-2024-11-20",
        client_settings=settings,
    )

    # Bind tools to the model
    model_with_tools = chat_model.bind_tools([get_weather, calculate])

    # Ask a question that requires a tool
    response = model_with_tools.invoke("What's the weather like in Paris?")
    print(f"Tool calls: {response.tool_calls}")

    # If there are tool calls, execute them and continue the conversation
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"  - Calling {tool_call['name']} with args: {tool_call['args']}")


def demo_embeddings():
    """Demonstrate embeddings generation."""
    print("\n" + "=" * 60)
    print("Demo: Embeddings")
    print("=" * 60)

    settings = get_default_client_settings()
    embeddings = get_embedding_model(
        model="text-embedding-3-large",
        client_settings=settings,
    )

    # Embed documents
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above an idle hound.",
        "Python is a programming language.",
    ]

    vectors = embeddings.embed_documents(texts)
    print(f"Generated {len(vectors)} embeddings")
    print(f"Embedding dimension: {len(vectors[0])}")

    # Embed a single query
    query_vector = embeddings.embed_query("What animal is mentioned?")
    print(f"Query embedding dimension: {len(query_vector)}")


def demo_normalized_api():
    """Demonstrate using the normalized (provider-agnostic) API."""
    print("\n" + "=" * 60)
    print("Demo: Normalized API")
    print("=" * 60)

    settings = get_default_client_settings()

    # Use normalized API for provider-agnostic calls
    chat_model = get_chat_model(
        model_name="gpt-4o-2024-11-20",
        client_settings=settings,
        client_type="normalized",  # Use normalized API
    )

    response = chat_model.invoke("What is machine learning in one sentence?")
    print(f"Response: {response.content}")


def demo_different_providers():
    """Demonstrate using different LLM providers."""
    print("\n" + "=" * 60)
    print("Demo: Different Providers")
    print("=" * 60)

    settings = get_default_client_settings()

    # OpenAI
    print("\n--- OpenAI GPT-4o ---")
    openai_chat = get_chat_model(
        model_name="gpt-4o-2024-11-20",
        client_settings=settings,
    )
    response = openai_chat.invoke("Say hello in 5 words or less.")
    print(f"GPT-4o: {response.content}")

    # You can also try other providers if they're available in your UiPath setup:
    # - Gemini: model_name="gemini-2.5-flash"
    # - Claude: model_name="anthropic.claude-sonnet-4-5-20250929-v1:0"


def demo_direct_client_usage():
    """Demonstrate using provider-specific client classes directly."""
    print("\n" + "=" * 60)
    print("Demo: Direct Client Usage")
    print("=" * 60)

    from uipath_langchain_client.clients.openai.chat_models import UiPathAzureChatOpenAI

    settings = get_default_client_settings()

    # Use the Azure OpenAI client directly for more control
    chat_model = UiPathAzureChatOpenAI(
        model="gpt-4o-2024-11-20",
        client_settings=settings,
        temperature=0.7,
    )

    response = chat_model.invoke("Tell me a fun fact about space.")
    print(f"Response: {response.content}")


def main():
    """Run all demos."""
    print("UiPath LangChain Client Demo")
    print("============================")

    # Basic demos
    demo_basic_chat()
    demo_chat_with_messages()
    demo_streaming()

    # Async demo
    asyncio.run(demo_async_operations())

    # Advanced demos
    demo_tool_calling()
    demo_embeddings()
    demo_normalized_api()
    demo_different_providers()
    demo_direct_client_usage()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
