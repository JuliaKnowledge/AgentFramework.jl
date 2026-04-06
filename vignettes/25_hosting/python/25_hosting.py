"""
Hosting and Agent-to-Agent Protocol — Python

This sample demonstrates two related capabilities:
  1. HostedRuntime — manage agent and workflow lifecycles with persistence
     and HTTP serving.
  2. A2A Protocol — connect to remote agents using the Agent-to-Agent
     standard (JSON-RPC 2.0 over HTTP).

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama agent-framework-a2a
"""

import asyncio

from agent_framework import Agent, Message, WorkflowBuilder
from agent_framework.ollama import OllamaChatClient


# --------------------------------------------------------------------------- #
# Part 1 — Hosted Runtime                                                     #
# --------------------------------------------------------------------------- #


async def hosting_demo() -> None:
    """Demonstrate the hosted runtime for agents and workflows."""

    # -- The Python SDK does not ship a single "HostedRuntime" object like the
    # Julia port.  Instead, hosting is achieved via platform-specific
    # integrations (Durable Task, Azure Functions, Starlette A2A server).
    # The pattern below shows the conceptual equivalent using Python idioms. --

    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    # ── 1. Create agents ──────────────────────────────────────────────────
    helper = client.as_agent(
        name="Helper",
        instructions="You are a helpful assistant.",
    )

    coder = client.as_agent(
        name="Coder",
        instructions="You write Python code. Always include docstrings.",
    )

    # ── 2. Run an agent ───────────────────────────────────────────────────
    print("=== Run Helper Agent ===\n")
    response = await helper.run("What is 2+2?")
    print(f"Response: {response.text}\n")

    # ── 3. Multi-turn with session persistence ────────────────────────────
    print("=== Multi-Turn Session ===\n")
    session = helper.create_session()

    r1 = await helper.run("My name is Alice", session=session)
    print(f"Turn 1: {r1.text}")

    r2 = await helper.run("What's my name?", session=session)
    print(f"Turn 2: {r2.text}\n")


# --------------------------------------------------------------------------- #
# Part 2 — A2A Protocol                                                       #
# --------------------------------------------------------------------------- #


async def a2a_demo() -> None:
    """Demonstrate connecting to a remote agent via the A2A protocol."""

    from agent_framework.a2a import A2AAgent

    # ── 1. Create an A2A agent pointing to a remote endpoint ──────────────
    remote = A2AAgent(
        url="http://localhost:8080",
        name="RemoteHelper",
        description="A remote assistant agent",
    )

    # ── 2. Discover capabilities via the agent card ───────────────────────
    print("=== A2A Agent Card ===\n")
    # The A2A agent fetches the agent card from /.well-known/agent.json
    # during connection.  Access it after first interaction or manually.

    # ── 3. Run the remote agent (non-streaming) ──────────────────────────
    print("=== A2A Non-Streaming ===\n")
    response = await remote.run("What is 2+2?")
    print(f"Response: {response.text}\n")

    # ── 4. Run the remote agent (streaming) ───────────────────────────────
    print("=== A2A Streaming ===\n")
    async for update in remote.run(
        "Tell me a story about Python",
        stream=True,
    ):
        if update.text:
            print(update.text, end="", flush=True)
    print("\n")

    # ── 5. Multi-turn with session persistence ────────────────────────────
    print("=== A2A Multi-Turn ===\n")
    session = await remote.create_session()

    r1 = await remote.run("My name is Alice", session=session)
    print(f"Turn 1: {r1.text}")

    r2 = await remote.run("What's my name?", session=session)
    print(f"Turn 2: {r2.text}\n")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


async def main() -> None:
    print("=" * 60)
    print("Part 1 — Hosted Runtime")
    print("=" * 60 + "\n")
    await hosting_demo()

    print("=" * 60)
    print("Part 2 — A2A Protocol")
    print("=" * 60 + "\n")
    # NOTE: A2A demo requires a running A2A server.  Start one with:
    #   uv run python a2a_server.py --port 8080
    # Uncomment the line below when a server is available:
    # await a2a_demo()
    print("(Skipped — start an A2A server first, then uncomment a2a_demo())\n")


if __name__ == "__main__":
    asyncio.run(main())
