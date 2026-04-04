"""
Multi-Agent Handoffs — Python

This sample demonstrates agent-to-agent delegation using handoffs.
A triage agent routes questions to specialist agents.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama

Note: HandoffBuilder requires tool-call support from the LLM provider.
  Ollama's agent-framework integration has a known compatibility issue with
  `allow_multiple_tool_calls`.  We demonstrate a manual handoff pattern
  that works with any provider, plus the HandoffBuilder pattern for
  providers that fully support tool calling (e.g., OpenAI).
"""

import asyncio

from agent_framework.ollama import OllamaChatClient


async def main() -> None:
    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    # ── Specialist agents ─────────────────────────────────────────────────
    math_agent = client.as_agent(
        name="MathExpert",
        instructions=(
            "You are a math expert. Solve problems step by step. "
            "Show your working clearly."
        ),
    )

    general_agent = client.as_agent(
        name="GeneralAssistant",
        instructions=(
            "You are a general knowledge assistant. "
            "Answer questions concisely about history, science, geography."
        ),
    )

    # ── Triage agent (classifier) ─────────────────────────────────────────
    triage_agent = client.as_agent(
        name="TriageAgent",
        instructions=(
            "You are a routing agent. Classify the user's question into one "
            "of these categories. Reply with ONLY the category name:\n"
            "- MATH — for math, arithmetic, algebra, calculus questions\n"
            "- GENERAL — for everything else\n"
            "Reply with just the single word: MATH or GENERAL."
        ),
    )

    # ── Manual handoff pattern (works with any provider) ──────────────────
    async def triage_and_route(question: str) -> str:
        """Route a question through triage to the right specialist."""
        # Step 1: classify
        classification = await triage_agent.run(question)
        category = classification.text.strip().upper()

        # Step 2: route to specialist
        if "MATH" in category:
            print(f"  [Triage] Routing to MathExpert")
            result = await math_agent.run(question)
        else:
            print(f"  [Triage] Routing to GeneralAssistant")
            result = await general_agent.run(question)

        return result.text

    # ── Run: math question ────────────────────────────────────────────────
    print("=== Math Question ===")
    answer = await triage_and_route("What is the integral of x^2?")
    print(f"Agent: {answer}\n")

    # ── Run: general knowledge question ───────────────────────────────────
    print("=== General Knowledge Question ===")
    answer = await triage_and_route("Who painted the Mona Lisa?")
    print(f"Agent: {answer}\n")

    # ── HandoffBuilder pattern (for providers with full tool-call support) ─
    # Uncomment the following when using OpenAI or another provider that
    # supports tool calling with allow_multiple_tool_calls:
    #
    # from agent_framework.orchestrations import HandoffBuilder
    #
    # workflow = (
    #     HandoffBuilder(
    #         name="triage_handoff",
    #         participants=[triage_agent, math_agent, general_agent],
    #     )
    #     .with_start_agent(triage_agent)
    #     .build()
    # )
    #
    # result = await workflow.run("What is the integral of x^2?")
    # for event in result:
    #     if event.type == "output" and hasattr(event.data, "text"):
    #         print(f"Agent: {event.data.text}")


if __name__ == "__main__":
    asyncio.run(main())
