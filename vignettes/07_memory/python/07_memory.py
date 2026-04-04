"""
Memory and Context Providers — Python

This sample demonstrates persistent memory using context providers:
  1. InMemoryHistoryProvider for automatic conversation history.
  2. A custom context provider with before_run/after_run hooks.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
from datetime import datetime
from typing import Any

from agent_framework import BaseContextProvider, AgentSession, SessionContext
from agent_framework.ollama import OllamaChatClient


# --------------------------------------------------------------------------- #
# Custom context provider: injects the current date/time into instructions.   #
# --------------------------------------------------------------------------- #


class DateTimeProvider(BaseContextProvider):
    """Injects the current date and time as a system instruction."""

    DEFAULT_SOURCE_ID = "datetime"

    def __init__(self) -> None:
        super().__init__(self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context.extend_instructions(
            self.source_id,
            f"Current date and time: {now_str}",
        )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        pass  # Nothing to persist for this simple provider.


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


async def main() -> None:
    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    # ── 1. Stateless agent (no memory) ────────────────────────────────────
    print("=== Stateless Agent ===")
    agent_no_memory = client.as_agent(
        name="ForgetfulBot",
        instructions="You are a helpful assistant. Keep answers brief.",
    )

    r1 = await agent_no_memory.run("My name is Alice.")
    print(f"Turn 1: {r1}")

    r2 = await agent_no_memory.run("What is my name?")
    print(f"Turn 2: {r2}")

    # ── 2. Agent with InMemoryHistoryProvider ─────────────────────────────
    print("\n=== Agent with Memory ===")
    from agent_framework import InMemoryHistoryProvider

    memory_agent = client.as_agent(
        name="MemoryBot",
        instructions="You are a helpful assistant. Keep answers brief.",
        context_providers=[InMemoryHistoryProvider()],
    )

    session = memory_agent.create_session()

    r1 = await memory_agent.run("My name is Alice.", session=session)
    print(f"Turn 1: {r1}")

    r2 = await memory_agent.run("What is my name?", session=session)
    print(f"Turn 2: {r2}")

    # ── 3. Agent with custom DateTimeProvider ─────────────────────────────
    print("\n=== Agent with Custom DateTimeProvider ===")
    datetime_agent = client.as_agent(
        name="TimeAwareBot",
        instructions="You are a helpful assistant. Keep answers brief.",
        context_providers=[DateTimeProvider(), InMemoryHistoryProvider()],
    )

    session = datetime_agent.create_session()
    r = await datetime_agent.run("What time is it right now?", session=session)
    print(f"Agent: {r}")


if __name__ == "__main__":
    asyncio.run(main())
