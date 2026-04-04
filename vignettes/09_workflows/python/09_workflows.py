"""
DAG-Based Workflows — Python

This sample demonstrates the workflow engine:
  1. Define executors (class-based and function-based).
  2. Build a pipeline with WorkflowBuilder.
  3. Run the workflow and inspect outputs.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
)
from agent_framework.ollama import OllamaChatClient
from typing_extensions import Never


# --------------------------------------------------------------------------- #
# Class-based executor: converts text to uppercase.                           #
# --------------------------------------------------------------------------- #


class UpperCase(Executor):
    """Converts input text to uppercase and sends it downstream."""

    def __init__(self) -> None:
        super().__init__(id="uppercase")

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        result = text.upper()
        await ctx.send_message(result)


# --------------------------------------------------------------------------- #
# Class-based executor: reverses text.                                        #
# --------------------------------------------------------------------------- #


class ReverseText(Executor):
    """Reverses the input string and sends it downstream."""

    def __init__(self) -> None:
        super().__init__(id="reverse")

    @handler
    async def reverse_text(self, text: str, ctx: WorkflowContext[str]) -> None:
        result = text[::-1]
        await ctx.send_message(result)


# --------------------------------------------------------------------------- #
# Function-based executor: adds exclamation marks and yields output.          #
# --------------------------------------------------------------------------- #


@executor(id="exclaim")
async def exclaim(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """Appends exclamation marks and yields as workflow output."""
    result = f"{text}!!!"
    await ctx.yield_output(result)


# --------------------------------------------------------------------------- #
# Main: build and run the pipeline.                                           #
# --------------------------------------------------------------------------- #


async def main() -> None:
    # ── 1. Simple text pipeline ───────────────────────────────────────────
    print("=== Text Pipeline: uppercase → reverse → exclaim ===\n")

    upper = UpperCase()
    reverse = ReverseText()

    workflow = (
        WorkflowBuilder(start_executor=upper)
        .add_edge(upper, reverse)
        .add_edge(reverse, exclaim)
        .build()
    )

    events = await workflow.run("hello world")
    outputs = events.get_outputs()
    print(f"Output: {outputs[0]}")  # !DLROW OLLEH!!!

    # ── 2. Agent-based workflow ───────────────────────────────────────────
    print("\n=== Agent Workflow: writer → reviewer ===\n")

    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    writer = client.as_agent(
        name="Writer",
        instructions="You write creative one-sentence slogans.",
    )

    reviewer = client.as_agent(
        name="Reviewer",
        instructions="You review slogans and provide brief, actionable feedback.",
    )

    agent_workflow = (
        WorkflowBuilder(start_executor=writer)
        .add_edge(writer, reviewer)
        .build()
    )

    events = await agent_workflow.run("Create a slogan for an electric bicycle.")
    outputs = events.get_outputs()
    for output in outputs:
        if hasattr(output, "text"):
            print(f"{output.messages[0].author_name}: {output.text}")
        else:
            print(f"Output: {output}")


if __name__ == "__main__":
    asyncio.run(main())
