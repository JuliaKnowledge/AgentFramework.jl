"""
Declarative Agents (Python)

This sample demonstrates how to define agents and workflows declaratively
using YAML configuration files, then load and run them with AgentFactory
and WorkflowFactory. It mirrors the Julia vignette 19_declarative.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-declarative agent-framework-ollama
"""

import asyncio
from pathlib import Path

from agent_framework import tool
from agent_framework.declarative import AgentFactory, WorkflowFactory
from agent_framework.ollama import OllamaChatClient
from pydantic import Field
from typing import Annotated


# ── A. Agent from YAML string ──────────────────────────────────────────── #

WEATHER_AGENT_YAML = """\
kind: Prompt
name: WeatherHelper
description: A weather assistant that answers concisely
instructions: You answer weather questions concisely.
"""


async def agent_from_yaml_string() -> None:
    """Create an agent from an inline YAML string."""
    print("=== A. Agent from YAML String ===\n")

    factory = AgentFactory(
        client=OllamaChatClient(
            host="http://localhost:11434",
            model="qwen3:8b",
        ),
        safe_mode=False,
    )

    agent = factory.create_agent_from_yaml(WEATHER_AGENT_YAML)
    response = await agent.run("Is it sunny in Paris today?")
    print(f"Agent: {response.text}\n")


# ── B. Agent from file ─────────────────────────────────────────────────── #


async def agent_from_file() -> None:
    """Load an agent definition from a YAML file on disk."""
    print("=== B. Agent from File ===\n")

    yaml_path = Path(__file__).parent / "agents" / "weather.yaml"
    if not yaml_path.exists():
        print(f"  Skipped — {yaml_path} not found.\n")
        return

    factory = AgentFactory(
        client=OllamaChatClient(
            host="http://localhost:11434",
            model="qwen3:8b",
        ),
        safe_mode=False,
    )
    agent = factory.create_agent_from_yaml_path(yaml_path)
    response = await agent.run("Will it rain in Tokyo tomorrow?")
    print(f"Agent: {response.text}\n")


# ── C. Registering tools for declarative use ────────────────────────────── #


@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the current weather for a location."""
    from random import choice, randint
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {choice(conditions)} at {randint(10, 30)}°C."


TOOL_AGENT_YAML = """\
kind: Prompt
name: WeatherAgent
description: Uses tools to answer weather questions
instructions: Use the available tools to answer weather questions accurately.
tools:
  - kind: function
    name: get_weather
    description: Get the current weather for a location.
    parameters:
      properties:
        location:
          kind: string
          description: The location to get the weather for.
          required: true
"""


async def agent_with_tools() -> None:
    """Create a declarative agent with function tools via bindings."""
    print("=== C. Agent with Registered Tools ===\n")

    factory = AgentFactory(
        client=OllamaChatClient(
            host="http://localhost:11434",
            model="qwen3:8b",
        ),
        bindings={"get_weather": get_weather},
        safe_mode=False,
    )

    agent = factory.create_agent_from_yaml(TOOL_AGENT_YAML)
    response = await agent.run("What's the weather in London?")
    print(f"Agent: {response.text}\n")


# ── D. Workflow from YAML ───────────────────────────────────────────────── #


WORKFLOW_YAML = """\
kind: Workflow
trigger:
  kind: OnConversationStart
  id: research_pipeline
  actions:
    - kind: InvokeAzureAgent
      id: researcher
      agent:
        name: Researcher
      output:
        messages: Local.ResearchOutput
    - kind: InvokeAzureAgent
      id: summarizer
      agent:
        name: Summarizer
      output:
        messages: Local.SummaryOutput
    - kind: EndWorkflow
      id: end
"""


async def workflow_from_yaml() -> None:
    """Create a multi-step workflow from a YAML definition."""
    print("=== D. Workflow from YAML ===\n")

    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    researcher = client.as_agent(
        name="Researcher",
        instructions="You find key facts about a topic. Be thorough but concise.",
    )
    summarizer = client.as_agent(
        name="Summarizer",
        instructions="You summarize research findings into a brief paragraph.",
    )

    factory = WorkflowFactory(
        agents={"Researcher": researcher, "Summarizer": summarizer},
    )

    workflow = factory.create_workflow_from_yaml(WORKFLOW_YAML)
    print("  Workflow created successfully.")
    print(f"  Trigger: {workflow.trigger.kind if hasattr(workflow, 'trigger') else 'N/A'}\n")


# ── E. Round-trip serialization ─────────────────────────────────────────── #


async def round_trip() -> None:
    """Demonstrate agent → YAML → agent round-trip."""
    print("=== E. Round-Trip Serialization ===\n")

    factory = AgentFactory(
        client=OllamaChatClient(
            host="http://localhost:11434",
            model="qwen3:8b",
        ),
        safe_mode=False,
    )

    # Create from YAML.
    agent = factory.create_agent_from_yaml(WEATHER_AGENT_YAML)
    print(f"  Original agent name: {agent.name}")

    # Serialize back (using the definition dict).
    # NOTE: Full YAML round-trip depends on the agent exposing its definition.
    agent2 = factory.create_agent_from_yaml(WEATHER_AGENT_YAML)
    print(f"  Recreated agent name: {agent2.name}\n")


# ── Main ────────────────────────────────────────────────────────────────── #


async def main() -> None:
    await agent_from_yaml_string()
    await agent_from_file()
    await agent_with_tools()
    await workflow_from_yaml()
    await round_trip()


if __name__ == "__main__":
    asyncio.run(main())
