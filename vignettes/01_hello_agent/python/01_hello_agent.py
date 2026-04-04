"""
Hello Agent — Simplest possible agent (Python)

This sample creates a minimal agent using the Ollama chat client,
sends a single prompt, and prints the response. It mirrors the Julia
vignette 01_hello_agent.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio

from agent_framework.ollama import OllamaChatClient


async def main() -> None:
    # Create a chat client pointing at the local Ollama instance.
    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    # Build an agent with a name and system instructions.
    agent = client.as_agent(
        name="HelloAgent",
        instructions="You are a friendly assistant. Keep your answers brief.",
    )

    # Run the agent with a simple prompt (non-streaming).
    result = await agent.run("What is the capital of France?")
    print(f"Agent: {result}")

    # Streaming: receive tokens as they are generated.
    print("Agent (streaming): ", end="", flush=True)
    async for chunk in agent.run("Tell me a one-sentence fun fact.", stream=True):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
