"""
Streaming Responses — Real-time token delivery (Python)

This sample demonstrates how to stream responses from an agent,
receiving tokens as they are generated rather than waiting for the
full response. It mirrors the Julia vignette 04_streaming.

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

    # Build an agent with instructions.
    agent = client.as_agent(
        name="StreamingAgent",
        instructions="You are a helpful assistant. Provide detailed answers.",
    )

    # ── Non-streaming: wait for the full response ────────────────────────
    print("=== Non-streaming ===")
    result = await agent.run("Explain the water cycle in one paragraph.")
    print(f"Agent: {result}")
    print()

    # ── Streaming: receive tokens as they arrive ─────────────────────────
    print("=== Streaming ===")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run(
        "Explain the water cycle in one paragraph.", stream=True
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()
    print()

    # ── Streaming with token inspection ──────────────────────────────────
    print("=== Token inspection ===")
    token_count = 0
    async for chunk in agent.run("What is 2 + 2?", stream=True):
        if chunk.text:
            token_count += 1
            print(f"  Fragment {token_count}: {chunk.text!r}")
    print(f"Total fragments: {token_count}")


if __name__ == "__main__":
    asyncio.run(main())
