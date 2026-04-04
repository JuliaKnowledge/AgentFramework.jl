"""
Adding Tools to Agents (Python)

This sample demonstrates how to define function tools with the @tool decorator
and attach them to an agent so the model can call them automatically.
It mirrors the Julia vignette 02_tools.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
from random import choice, randint
from typing import Annotated

from agent_framework import tool
from agent_framework.ollama import OllamaChatClient
from pydantic import Field


# Define tools using the @tool decorator.
# NOTE: approval_mode="never_require" is for sample brevity.
# Use "always_require" in production for user confirmation before tool execution.
@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the current weather for a location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {choice(conditions)} with a high of {randint(10, 30)}°C."


@tool(approval_mode="never_require")
def get_population(
    country: Annotated[str, Field(description="The country to look up.")],
) -> str:
    """Get the approximate population of a country in millions."""
    populations = {
        "France": 68,
        "Germany": 84,
        "Japan": 125,
        "Brazil": 214,
        "Australia": 26,
    }
    pop = populations.get(country)
    if pop is not None:
        return f"{country} has approximately {pop} million people."
    return f"Population data not available for {country}."


@tool(approval_mode="never_require")
def calculate(
    expression: Annotated[str, Field(description="A mathematical expression to evaluate.")],
) -> str:
    """Evaluate a mathematical expression and return the result."""
    # WARNING: eval is used here for demo purposes only.
    result = eval(expression)  # noqa: S307
    return str(result)


async def main() -> None:
    # Create a chat client and agent with tools.
    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    agent = client.as_agent(
        name="ToolAgent",
        instructions="You are a helpful assistant. Use the available tools to answer questions accurately. Be concise.",
        tools=[get_weather, get_population, calculate],
    )

    # The agent will automatically call the get_weather tool.
    result = await agent.run("What's the weather like in Tokyo?")
    print(f"Agent: {result}\n")

    # Ask about population — triggers the get_population tool.
    result = await agent.run("What is the population of Brazil?")
    print(f"Agent: {result}\n")

    # A question that may require multiple tool calls.
    result = await agent.run("What is the combined population of France and Germany?")
    print(f"Agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
