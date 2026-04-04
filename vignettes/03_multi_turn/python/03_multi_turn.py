"""
Multi-Turn Conversations (Python)

This sample shows how to maintain conversation context across multiple
agent calls by reusing a session object. It mirrors the Julia vignette
03_multi_turn.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio

from agent_framework.ollama import OllamaChatClient


async def main() -> None:
    # Create a chat client and agent.
    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    agent = client.as_agent(
        name="ConversationAgent",
        instructions="You are a friendly assistant. Keep your answers brief.",
    )

    # Create a session to maintain conversation history across turns.
    session = agent.create_session()

    # Turn 1 — introduce yourself.
    result = await agent.run("My name is Alice and I love hiking.", session=session)
    print(f"Turn 1: {result}\n")

    # Turn 2 — the agent should remember the user's name and hobby.
    result = await agent.run("What do you remember about me?", session=session)
    print(f"Turn 2: {result}\n")

    # Turn 3 — continue building on the conversation.
    result = await agent.run("Suggest a hiking trail for me.", session=session)
    print(f"Turn 3: {result}\n")

    # Demonstrate multiple independent sessions.
    session_a = agent.create_session()
    session_b = agent.create_session()

    await agent.run("My name is Alice.", session=session_a)
    await agent.run("My name is Bob.", session=session_b)

    resp_a = await agent.run("What is my name?", session=session_a)
    resp_b = await agent.run("What is my name?", session=session_b)

    print(f"Session A: {resp_a}")
    print(f"Session B: {resp_b}")


if __name__ == "__main__":
    asyncio.run(main())
