"""
Evaluation Framework — Testing and validating agent behaviour (Python)

This sample demonstrates how to define checks, build evaluators, and run
them against agents and workflows to verify correct behaviour. It mirrors
the Julia vignette 17_evaluation.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
from typing import Any

from agent_framework import Agent, Message, tool
from agent_framework import (
    CheckResult,
    EvalItem,
    EvalResults,
    ExpectedToolCall,
    LocalEvaluator,
    evaluate_agent,
    evaluator,
    keyword_check,
    tool_call_args_match,
    tool_called_check,
    tool_calls_present,
)
from agent_framework.ollama import OllamaChatClient


# ── Weather tool ──────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    lookup = {
        "paris": "Paris: Sunny, 22°C",
        "london": "London: Rainy, 14°C",
    }
    return lookup.get(city.lower(), f"{city}: Cloudy, 18°C")


# ── Custom evaluators ────────────────────────────────────────────────────

@evaluator(name="response_length")
def length_check(response: str) -> bool:
    """Pass if the response is longer than 50 characters."""
    return len(response) > 50


@evaluator(name="semantic_match")
def similarity_check(response: str, expected_output: str) -> bool:
    """Pass if the response contains the expected text (case-insensitive)."""
    return expected_output.lower() in response.lower()


@evaluator(name="keyword_coverage")
def coverage_check(response: str) -> float:
    """Return fraction of target keywords found in the response."""
    keywords = ["sunny", "temperature", "degrees"]
    matched = sum(1 for k in keywords if k in response.lower())
    return matched / len(keywords)


# ── Main ──────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    weather_agent = Agent(
        client,
        name="WeatherBot",
        instructions=(
            "You are a weather assistant. Use the get_weather tool to "
            "answer weather questions. Always include the temperature."
        ),
        tools=[get_weather],
    )

    # ── Built-in checks ──────────────────────────────────────────────────
    print("=== Built-in Checks ===")

    local_eval = LocalEvaluator(
        keyword_check("weather"),
        length_check,
        tool_called_check("get_weather"),
        tool_calls_present,
    )

    # ── Evaluate a single agent ──────────────────────────────────────────
    print("\n=== evaluate_agent ===")

    results: list[EvalResults] = await evaluate_agent(
        agent=weather_agent,
        queries=["What's the weather in Paris?", "Will it rain in London?"],
        evaluators=local_eval,
        eval_name="weather_basic",
    )

    for r in results:
        print(f"Provider : {r.provider}")
        print(f"Passed   : {r.passed} / {r.total}")
        print(f"All OK?  : {r.all_passed}")
        for item_result in r.items or []:
            print(f"  Item {item_result.item_id}: {item_result.status}")
            for score in item_result.scores:
                mark = "PASS" if score.passed else "FAIL"
                print(f"    {score.name}: {mark} (score={score.score:.2f})")
    print()

    # ── Evaluate with expected output ────────────────────────────────────
    print("=== evaluate_agent with expected_output ===")

    results = await evaluate_agent(
        agent=weather_agent,
        queries=["What's the weather in Paris?"],
        expected_output="Sunny",
        evaluators=similarity_check,
    )

    for r in results:
        print(f"Passed: {r.passed} / {r.total}")
    print()

    # ── Evaluate with expected tool calls ────────────────────────────────
    print("=== evaluate_agent with expected_tool_calls ===")

    results = await evaluate_agent(
        agent=weather_agent,
        queries=["What's the weather in Paris?"],
        expected_tool_calls=[ExpectedToolCall("get_weather", {"city": "Paris"})],
        evaluators=[
            LocalEvaluator(tool_call_args_match, tool_calls_present),
        ],
    )

    for r in results:
        print(f"Passed: {r.passed} / {r.total}")
    print()

    # ── EvalItem with pre-built conversation ─────────────────────────────
    print("=== EvalItem ===")

    item = EvalItem(
        conversation=[
            Message("user", ["What's the weather?"]),
            Message("assistant", ["It's sunny and 22°C in Paris."]),
        ],
        expected_output="sunny",
    )
    print(f"Query   : {item.query}")
    print(f"Response: {item.response}")
    print()

    # ── raise_for_status ─────────────────────────────────────────────────
    print("=== raise_for_status ===")

    results = await evaluate_agent(
        agent=weather_agent,
        queries=["What's the weather in Paris?"],
        evaluators=local_eval,
    )
    try:
        results[0].raise_for_status(msg="Quality gate failed")
        print("All checks passed!")
    except Exception as exc:
        print(f"Evaluation failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
