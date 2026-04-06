"""
Mem0 Persistent Memory — Python

This sample demonstrates Mem0 integration for persistent agent memory:
  1. Connecting to Mem0 (platform and OSS).
  2. Using Mem0ContextProvider for automatic memory injection.
  3. Scoping memories by user, agent, and application.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama agent-framework-mem0
  - A Mem0 API key (set MEM0_API_KEY env var) or local Mem0 OSS instance
"""

import asyncio
import os
from typing import Any

from agent_framework import Agent, AgentSession
from agent_framework.mem0 import Mem0ContextProvider
from agent_framework.ollama import OllamaChatClient


# --------------------------------------------------------------------------- #
# Helper: print a divider                                                     #
# --------------------------------------------------------------------------- #

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


async def main() -> None:
    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    api_key = os.environ.get("MEM0_API_KEY", "")

    # ── 1. Basic Mem0 memory across sessions ──────────────────────────────
    section("1. Basic Mem0 Memory Across Sessions")

    try:
        provider = Mem0ContextProvider(
            source_id="mem0",
            user_id="user-123",
            api_key=api_key or None,
        )

        async with Agent(
            client,
            name="MemoryAgent",
            instructions="You remember what users tell you across conversations. Keep answers brief.",
            context_providers=[provider],
        ) as agent:
            # First conversation — agent learns about the user
            r1 = await agent.run("My name is Alice and I love Julia programming.")
            print(f"Turn 1: {r1}")

            # New session — memories persist via Mem0
            session2 = agent.create_session()
            r2 = await agent.run("What do you know about me?", session=session2)
            print(f"Turn 2 (new session): {r2}")
    except Exception as exc:
        print(f"  ⚠ Skipped (Mem0 unavailable): {type(exc).__name__}: {exc}")

    # ── 2. Agent-scoped memory isolation ──────────────────────────────────
    section("2. Agent-Scoped Memory Isolation")

    try:
        personal_provider = Mem0ContextProvider(
            source_id="mem0",
            agent_id="personal-assistant",
            user_id="user-123",
            api_key=api_key or None,
        )

        work_provider = Mem0ContextProvider(
            source_id="mem0",
            agent_id="work-assistant",
            user_id="user-123",
            api_key=api_key or None,
        )

        async with (
            Agent(
                client,
                name="PersonalAssistant",
                instructions="You help with personal tasks and remember preferences.",
                context_providers=[personal_provider],
            ) as personal_agent,
            Agent(
                client,
                name="WorkAssistant",
                instructions="You help with professional tasks and remember work context.",
                context_providers=[work_provider],
            ) as work_agent,
        ):
            # Store personal information
            r = await personal_agent.run(
                "Remember that I exercise at 6 AM and prefer outdoor activities."
            )
            print(f"Personal Agent: {r}")

            # Store work information
            r = await work_agent.run(
                "Remember that I have team meetings every Tuesday at 2 PM."
            )
            print(f"Work Agent: {r}")

            # Each agent only sees its own scoped memories
            personal_session = personal_agent.create_session()
            r = await personal_agent.run(
                "What do you know about my schedule?", session=personal_session
            )
            print(f"Personal Agent (new session): {r}")

            work_session = work_agent.create_session()
            r = await work_agent.run(
                "What do you know about my schedule?", session=work_session
            )
            print(f"Work Agent (new session): {r}")
    except Exception as exc:
        print(f"  ⚠ Skipped (Mem0 unavailable): {type(exc).__name__}: {exc}")

    # ── 3. Application-scoped memory ──────────────────────────────────────
    section("3. Application-Scoped Memory")

    try:
        app_provider = Mem0ContextProvider(
            source_id="mem0",
            application_id="my-app",
            user_id="user-456",
            api_key=api_key or None,
        )

        async with Agent(
            client,
            name="AppAgent",
            instructions="You are an assistant scoped to a specific application.",
            context_providers=[app_provider],
        ) as app_agent:
            r = await app_agent.run("I prefer dark mode and metric units.")
            print(f"App Agent: {r}")

            app_session = app_agent.create_session()
            r = await app_agent.run("What are my preferences?", session=app_session)
            print(f"App Agent (new session): {r}")
    except Exception as exc:
        print(f"  ⚠ Skipped (Mem0 unavailable): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
