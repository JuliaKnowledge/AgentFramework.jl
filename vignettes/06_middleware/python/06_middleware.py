"""
Middleware — Intercepting and modifying agent behaviour (Python)

This sample demonstrates how to use middleware to add logging, timing,
and guardrail capabilities to agents at three levels: agent, chat client,
and function invocation. It mirrors the Julia vignette 06_middleware.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
import time
from typing import Awaitable, Callable

from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    ChatContext,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    tool,
)
from agent_framework.ollama import OllamaChatClient


# ── Define a tool ────────────────────────────────────────────────────────

@tool(approval_mode="never_require")
def get_population(city: str) -> str:
    """Get the population of a city.

    Args:
        city: The city name.

    Returns:
        The population as a string.
    """
    populations = {
        "Paris": "2.1 million",
        "London": "8.8 million",
        "Tokyo": "14 million",
    }
    return populations.get(city, "Unknown")


# ── Agent middleware: logging ────────────────────────────────────────────

class LoggingAgentMiddleware(AgentMiddleware):
    """Logs the start and end of every agent run."""

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        agent_name = context.agent.name
        n_messages = len(context.messages)
        print(f"[AgentMW] Starting {agent_name} with {n_messages} message(s)")

        await call_next()

        print(f"[AgentMW] {agent_name} completed")


# ── Chat middleware: timing ──────────────────────────────────────────────

class TimingChatMiddleware(ChatMiddleware):
    """Measures the duration of each LLM API call."""

    async def process(
        self,
        context: ChatContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        n_msgs = len(context.messages)
        print(f"  [ChatMW] Sending {n_msgs} messages to LLM...")

        start = time.time()
        await call_next()
        elapsed = time.time() - start

        print(f"  [ChatMW] LLM responded in {elapsed:.2f}s")


# ── Function middleware: tool logging ────────────────────────────────────

class ToolLoggingMiddleware(FunctionMiddleware):
    """Logs every tool invocation with its arguments and result."""

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        tool_name = context.function.name
        print(f"    [FuncMW] Calling tool: {tool_name}")

        await call_next()

        print(f"    [FuncMW] Tool {tool_name} returned: {context.result}")


# ── Security guardrail middleware ────────────────────────────────────────

class SecurityMiddleware(AgentMiddleware):
    """Blocks requests containing sensitive keywords."""

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        for msg in context.messages:
            if msg.text and any(
                kw in msg.text.lower() for kw in ("password", "secret")
            ):
                print("[Security] Blocked request: sensitive content detected")
                context.result = AgentResponse(
                    messages=[
                        Message(
                            "assistant",
                            ["I cannot process requests containing sensitive information."],
                        )
                    ]
                )
                return  # short-circuit — do not call next

        await call_next()


async def main() -> None:
    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    # ── Agent with all three middleware layers ────────────────────────────
    print("=== Full middleware pipeline ===")
    agent = client.as_agent(
        name="CityBot",
        instructions="You are a helpful city information assistant. Use the get_population tool when asked about population.",
        tools=[get_population],
        middleware=[
            LoggingAgentMiddleware(),
            TimingChatMiddleware(),
            ToolLoggingMiddleware(),
        ],
    )

    result = await agent.run("What is the population of Paris?")
    print(f"\nAnswer: {result.text}")
    print()

    # ── Security guardrail ───────────────────────────────────────────────
    print("=== Security guardrail ===")
    secure_agent = client.as_agent(
        name="SecureBot",
        instructions="You are a helpful assistant.",
        middleware=[SecurityMiddleware()],
    )

    result = await secure_agent.run("What is my password?")
    print(f"Blocked: {result.text}")

    result = await secure_agent.run("What is the capital of France?")
    print(f"Allowed: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
