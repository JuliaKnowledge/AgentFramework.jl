"""
Skills System (Python)

This sample demonstrates the Agent Skills system: reusable, discoverable
instruction sets that extend agent capabilities via progressive disclosure.
It mirrors the Julia vignette 23_skills.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
import json

from textwrap import dedent
from typing import Any

from agent_framework import Agent, Skill, SkillResource, SkillsProvider
from agent_framework.ollama import OllamaChatClient


# ── Parsing SKILL.md files ───────────────────────────────────────────────────

# File-based skills are discovered from SKILL.md files on disk.
# Each skill lives in its own directory:
#
#   skills/
#   ├── data_analysis/
#   │   ├── SKILL.md
#   │   └── schema.json
#   └── code_review/
#       ├── SKILL.md
#       └── style_guide.md
#
# The SKILL.md format uses YAML frontmatter:
#
#   ---
#   name: DataAnalysis
#   description: Skill for analyzing tabular data
#   version: 1.0.0
#   tags: [data, analysis, csv]
#   ---
#
#   # Data Analysis Instructions
#
#   When asked to analyze data:
#   1. First understand the structure ...


# ── Code-Defined Skills ──────────────────────────────────────────────────────

# 1. Create a skill with static resources (inline content).
unit_converter_skill = Skill(
    name="unit-converter",
    description="Convert between common units using a conversion factor",
    content=dedent("""\
        Use this skill when the user asks to convert between units.

        1. Review the conversion-tables resource to find the factor for the
           requested conversion.
        2. Check the conversion-policy resource for rounding and formatting rules.
        3. Use the convert script, passing the value and factor from the table.
    """),
    resources=[
        SkillResource(
            name="conversion-tables",
            content=dedent("""\
                # Conversion Tables

                Formula: **result = value × factor**

                | From        | To          | Factor   |
                |-------------|-------------|----------|
                | miles       | kilometers  | 1.60934  |
                | kilometers  | miles       | 0.621371 |
                | pounds      | kilograms   | 0.453592 |
                | kilograms   | pounds      | 2.20462  |
            """),
        ),
    ],
)


# 2. Dynamic resource — callable function via @skill.resource decorator.
@unit_converter_skill.resource(
    name="conversion-policy",
    description="Current conversion formatting and rounding policy",
)
def conversion_policy(**kwargs: Any) -> Any:
    """Return the current conversion policy (computed at runtime)."""
    precision = kwargs.get("precision", 4)
    return dedent(f"""\
        # Conversion Policy

        **Decimal places:** {precision}
        **Format:** Always show both the original and converted values with units
    """)


# 3. Dynamic script — in-process callable.
@unit_converter_skill.script(
    name="convert",
    description="Convert a value: result = value × factor",
)
def convert_units(value: float, factor: float, **kwargs: Any) -> str:
    """Convert a value using a multiplication factor."""
    precision = kwargs.get("precision", 4)
    result = round(value * factor, precision)
    return json.dumps({"value": value, "factor": factor, "result": result})


# ── Using SkillsProvider with an Agent ───────────────────────────────────────

async def code_defined_skills_example() -> None:
    """Demonstrate code-defined skills with an agent."""
    print("=== Code-Defined Skills ===\n")

    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    # Pass skills directly to the provider — no files needed.
    async with Agent(
        client=client,
        instructions="You are a helpful assistant that can convert units.",
        context_providers=[SkillsProvider(skills=[unit_converter_skill])],
    ) as agent:
        response = await agent.run(
            "How many kilometers is a marathon (26.2 miles)?",
        )
        print(f"Agent: {response.text}\n")


# ── File-Based Skills ────────────────────────────────────────────────────────

async def file_based_skills_example() -> None:
    """Demonstrate file-based skills discovered from a directory."""
    print("=== File-Based Skills ===\n")

    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    # Discover skills from a directory of SKILL.md files.
    skills_provider = SkillsProvider(skill_paths="./skills")

    async with Agent(
        client=client,
        instructions="You have access to various skills. Use them when appropriate.",
        context_providers=[skills_provider],
    ) as agent:
        response = await agent.run("Analyze this data: name,age\nAlice,30\nBob,25")
        print(f"Agent: {response.text}\n")


# ── Mixed Skills (Code + File) ───────────────────────────────────────────────

async def mixed_skills_example() -> None:
    """Combine code-defined and file-based skills in a single agent."""
    print("=== Mixed Skills ===\n")

    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    # Combine code skills with file-based discovery.
    skills_provider = SkillsProvider(
        skill_paths="./skills",
        skills=[unit_converter_skill],
    )

    async with Agent(
        client=client,
        instructions="You are a helpful assistant with access to various skills.",
        context_providers=[skills_provider],
    ) as agent:
        response = await agent.run(
            "How many kilometers is a marathon (26.2 miles)? "
            "And how many pounds is 75 kilograms?"
        )
        print(f"Agent: {response.text}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Code-defined skills (no files needed).
    await code_defined_skills_example()

    # File-based skills — uncomment if you have a skills/ directory:
    # await file_based_skills_example()

    # Mixed skills — uncomment if you have a skills/ directory:
    # await mixed_skills_example()


if __name__ == "__main__":
    asyncio.run(main())
