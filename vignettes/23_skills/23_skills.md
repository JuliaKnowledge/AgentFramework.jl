# Skills System
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [What Are Skills?](#what-are-skills)
- [The SKILL.md Format](#the-skillmd-format)
- [Parsing SKILL.md Files](#parsing-skillmd-files)
- [Discovering Skills from a
  Directory](#discovering-skills-from-a-directory)
  - [Discovery Options](#discovery-options)
- [SkillsProvider — Injecting Skills into an
  Agent](#skillsprovider--injecting-skills-into-an-agent)
  - [From a Directory](#from-a-directory)
  - [From Code-Defined Skills](#from-code-defined-skills)
- [Skill Resources](#skill-resources)
  - [Static Resources](#static-resources)
  - [Dynamic Resources](#dynamic-resources)
  - [Retrieving Resource Content](#retrieving-resource-content)
- [Composing Skill Sources](#composing-skill-sources)
  - [Source Types](#source-types)
- [Progressive Disclosure in Action](#progressive-disclosure-in-action)
- [Full Example: Multi-Skill Agent](#full-example-multi-skill-agent)
- [Summary](#summary)

## Overview

Skills are **reusable, discoverable instruction sets** that can be
dynamically loaded into agents — think of them as plugins for agent
behaviour. Instead of stuffing every possible instruction into the
system prompt, skills follow a **progressive disclosure** pattern:

1.  **Advertise** — skill names and descriptions are injected into the
    prompt.
2.  **Load** — full instructions are fetched on demand via a
    `load_skill` tool.
3.  **Read resources** — supplementary data is read via
    `read_skill_resource`.

In this vignette you will learn how to:

1.  Understand the `SKILL.md` file format and its YAML frontmatter.
2.  Parse a `SKILL.md` file into a `Skill` struct.
3.  Discover skills from a directory tree with `discover_skills`.
4.  Use `SkillsProvider` to inject skills into an agent’s context.
5.  Define skills entirely in code with `SkillResource`.
6.  Compose skill sources with `SkillSourceBuilder`.

## Prerequisites

You need [Ollama](https://ollama.com) running locally with the
`qwen3:8b` model pulled:

``` bash
ollama pull qwen3:8b
```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..",".."))
using AgentFramework
```

## What Are Skills?

An agent’s system prompt has finite capacity. Skills let you keep
instructions modular — each skill lives in its own `SKILL.md` file (or
is defined in code) and is only loaded when the LLM decides it needs it.
This keeps the base prompt small and focused while giving the agent
access to a rich library of domain-specific knowledge.

A skill consists of:

| Component        | Purpose                                              |
|------------------|------------------------------------------------------|
| **Name**         | Unique identifier for the skill                      |
| **Description**  | Short summary shown to the LLM in the prompt         |
| **Version**      | Semantic version string                              |
| **Tags**         | Categorisation labels for filtering/discovery        |
| **Instructions** | Full markdown body — loaded on demand                |
| **Resources**    | Named supplementary content (files, schemas, tables) |

## The SKILL.md Format

Each skill lives in its own directory with a `SKILL.md` file. The file
uses YAML frontmatter for metadata, followed by the instructions body:

``` markdown
---
name: DataAnalysis
description: Skill for analyzing tabular data
version: 1.0.0
tags: [data, analysis, csv]
---

# Data Analysis Instructions

When asked to analyze data:
1. First understand the structure (columns, types, row count)
2. Look for patterns, outliers, and trends
3. Summarize findings with statistics
4. Suggest visualizations where appropriate
```

The frontmatter fields (`name`, `description`, `version`, `tags`) are
all optional — sensible defaults are inferred from the directory name
and file content.

## Parsing SKILL.md Files

Use `parse_skill_md` to load a single skill file into a `Skill` struct:

``` julia
skill = parse_skill_md("skills/data_analysis/SKILL.md")
println(skill.name)         # "DataAnalysis"
println(skill.description)  # "Skill for analyzing tabular data"
println(skill.version)      # "1.0.0"
println(skill.tags)         # ["data", "analysis", "csv"]
println(skill.instructions) # The markdown content after frontmatter
```

You can also parse a raw string with `parse_skill_md_content`:

``` julia
content = """
---
name: Greeter
description: A simple greeting skill
version: 0.1.0
tags: [demo]
---

When asked to greet someone, respond warmly and use their name.
"""

skill = parse_skill_md_content(content)
println(skill.name)  # "Greeter"
```

## Discovering Skills from a Directory

A typical project organises skills in a directory tree:

    skills/
    ├── data_analysis/
    │   ├── SKILL.md
    │   └── schema.json
    ├── code_review/
    │   ├── SKILL.md
    │   └── style_guide.md
    └── summarisation/
        └── SKILL.md

Use `discover_skills` to scan the tree and collect all skills:

``` julia
skills = discover_skills("./skills"; recursive=true, max_depth=2)
for s in skills
    println("$(s.name) v$(s.version): $(s.description)")
end
```

**Expected output:**

    DataAnalysis v1.0.0: Skill for analyzing tabular data
    CodeReview v1.0.0: Skill for reviewing code quality
    Summarisation v1.0.0: Skill for summarizing long documents

The scanner automatically picks up file-based resources (`.md`, `.txt`,
`.json`, `.yaml`, `.yml`, `.jl`, `.py` files) in each skill directory
and attaches them to the skill.

### Discovery Options

| Keyword           | Default | Purpose                               |
|-------------------|---------|---------------------------------------|
| `recursive`       | `true`  | Recurse into subdirectories           |
| `max_depth`       | `2`     | Maximum recursion depth               |
| `follow_symlinks` | `false` | Follow symbolic links (security risk) |

## SkillsProvider — Injecting Skills into an Agent

The `SkillsProvider` is a context provider that wires skills into an
agent. On each call to `run_agent`, it:

1.  Injects a skill advertisement into the system prompt.
2.  Adds `load_skill` and `read_skill_resource` tools to the agent’s
    tool belt.

### From a Directory

``` julia
provider = SkillsProvider()
add_skills_from_directory!(provider, "./skills")

agent = Agent(
    name = "SkillfulAgent",
    instructions = "You have access to various skills. Use them when appropriate.",
    client = OllamaChatClient(model = "qwen3:8b"),
    context_providers = [provider],
)

response = run_agent(agent, "Analyze this CSV data: name,age\\nAlice,30\\nBob,25")
println(response.text)
```

When the agent runs, the LLM sees a compact skill listing in the system
prompt. If it decides a skill is relevant, it calls
`load_skill("DataAnalysis")` to get the full instructions, then follows
them.

### From Code-Defined Skills

You can also pass skills directly without any files on disk:

``` julia
skill = Skill(
    name = "UnitConverter",
    description = "Convert between common units of measurement",
    version = "1.0.0",
    instructions = """
        When asked to convert units:
        1. Identify the source and target units
        2. Apply the conversion factor
        3. Show the formula and result
    """,
    tags = ["math", "conversion"],
)

provider = SkillsProvider(skills = [skill])

agent = Agent(
    name = "ConverterAgent",
    instructions = "You are a helpful assistant.",
    client = OllamaChatClient(model = "qwen3:8b"),
    context_providers = [provider],
)

response = run_agent(agent, "How many kilometers is 26.2 miles?")
println(response.text)
```

**Expected output:**

    A marathon distance of 26.2 miles is approximately 42.16 kilometers
    (26.2 × 1.60934 = 42.16).

## Skill Resources

Resources are named supplementary content attached to a skill. They let
you separate reference data from instructions — the LLM reads resources
on demand via the `read_skill_resource` tool.

### Static Resources

Provide inline content at construction time:

``` julia
tables_resource = SkillResource(
    name = "conversion-tables",
    description = "Conversion factors for common units",
    content = """
        # Conversion Tables

        | From        | To          | Factor   |
        |-------------|-------------|----------|
        | miles       | kilometers  | 1.60934  |
        | kilometers  | miles       | 0.621371 |
        | pounds      | kilograms   | 0.453592 |
        | kilograms   | pounds      | 2.20462  |
    """,
    mime_type = "text/markdown",
)

skill = Skill(
    name = "UnitConverter",
    description = "Convert between common units using a conversion factor",
    instructions = """
        1. Review the conversion-tables resource to find the correct factor.
        2. Compute: result = value × factor.
        3. Present both the original and converted values with units.
    """,
    resources = Dict("conversion-tables" => tables_resource),
)
```

### Dynamic Resources

Use the `fn` field to compute content lazily at runtime:

``` julia
schema_resource = SkillResource(
    name = "schema",
    description = "JSON schema for data format",
    fn = () -> read("schema.json", String),
    mime_type = "application/json",
)

policy_resource = SkillResource(
    name = "policy",
    description = "Current formatting policy",
    fn = () -> """
        # Formatting Policy
        - Decimal places: 4
        - Always show both original and converted values
        - Generated at: $(now())
    """,
)

skill = Skill(
    name = "DataImporter",
    description = "Import data from various formats",
    instructions = "Use the schema resource to validate input data.",
    resources = Dict(
        "schema" => schema_resource,
        "policy" => policy_resource,
    ),
)
```

Dynamic resources are evaluated only when the LLM calls
`read_skill_resource`, so they can return live data like timestamps,
database lookups, or configuration values.

### Retrieving Resource Content

Use `get_resource_content` to read a resource’s value programmatically:

``` julia
content = get_resource_content(tables_resource)
println(content)  # prints the conversion table markdown
```

## Composing Skill Sources

For complex projects you may want to combine skills from multiple
directories, filter by tags, and deduplicate. The `SkillSourceBuilder`
provides a fluent API for this:

``` julia
source = SkillSourceBuilder() |>
    b -> add_directory!(b, "core_skills/") |>
    b -> add_directory!(b, "extra_skills/") |>
    b -> deduplicate!(b) |>
    b -> filter_by!(b, s -> "approved" in s.tags) |>
    build

provider = SkillsProvider()
load_skills!(provider, source)
```

### Source Types

AgentFramework.jl provides several composable source types:

| Source | Purpose |
|----|----|
| `StaticSkillSource(skills)` | Fixed vector of skills |
| `DirectorySkillSource(directory=...)` | Discover from filesystem |
| `DeduplicatingSkillSource(inner)` | Remove duplicates by name (first wins) |
| `FilteringSkillSource(inner, pred)` | Filter by a predicate function |
| `AggregatingSkillSource(sources)` | Combine multiple sources |

These can be nested arbitrarily. For example, deduplicate across two
directories and then filter:

``` julia
# Manual composition (equivalent to the builder above)
source = FilteringSkillSource(
    DeduplicatingSkillSource(
        AggregatingSkillSource([
            DirectorySkillSource(directory = "core_skills/"),
            DirectorySkillSource(directory = "extra_skills/"),
        ])
    ),
    s -> "approved" in s.tags,
)

skills = get_skills(source)
for s in skills
    println("$(s.name): $(s.description)")
end
```

## Progressive Disclosure in Action

When a `SkillsProvider` is attached to an agent, the following happens
automatically on each `run_agent` call:

**Step 1 — Advertise.** The system prompt includes a compact listing:

    You have access to the following skills. Use the load_skill tool to get
    detailed instructions before using a skill.

    Available skills:
    - **UnitConverter** (v1.0.0): Convert between common units [math, conversion]
      - Resource: conversion-tables
    - **DataAnalysis** (v1.0.0): Analyze tabular data [data, analysis, csv]
      - Resource: schema.json

**Step 2 — Load.** When the LLM calls `load_skill("UnitConverter")`, it
receives the full instructions and a list of available resources.

**Step 3 — Read.** The LLM calls
`read_skill_resource("UnitConverter", "conversion-tables")` to get the
factor table, then computes the answer.

This three-step pattern keeps the base prompt small while giving agents
access to arbitrarily rich domain knowledge.

## Full Example: Multi-Skill Agent

Putting it all together — a code-defined skill with resources and a
directory of file-based skills:

``` julia
# Code-defined skill with static and dynamic resources
converter_skill = Skill(
    name = "UnitConverter",
    description = "Convert between common units using a conversion factor",
    instructions = """
        1. Read the conversion-tables resource for the correct factor.
        2. Compute: result = value × factor.
        3. Present both original and converted values.
    """,
    resources = Dict(
        "conversion-tables" => SkillResource(
            name = "conversion-tables",
            description = "Conversion factors table",
            content = """
                | From   | To         | Factor  |
                |--------|------------|---------|
                | miles  | kilometers | 1.60934 |
                | pounds | kilograms  | 0.453592|
            """,
        ),
        "policy" => SkillResource(
            name = "policy",
            description = "Formatting rules",
            fn = () -> "Round to 2 decimal places. Show both units.",
        ),
    ),
    tags = ["math", "approved"],
)

# Combine code skill with file-based skills
source = SkillSourceBuilder() |>
    b -> add_skills!(b, [converter_skill]) |>
    b -> add_directory!(b, "skills/") |>
    b -> deduplicate!(b) |>
    build

provider = SkillsProvider()
load_skills!(provider, source)

agent = Agent(
    name = "MultiSkillAgent",
    instructions = "You are a helpful assistant with access to various skills.",
    client = OllamaChatClient(model = "qwen3:8b"),
    context_providers = [provider],
)

response = run_agent(agent, "How many kilometers is a marathon (26.2 miles)?")
println(response.text)
```

**Expected output:**

    A marathon is 26.2 miles, which is approximately 42.16 kilometers
    (26.2 × 1.60934 ≈ 42.16).

## Summary

| Concept | Type / Function | Purpose |
|----|----|----|
| **Skill** | `Skill` | Discoverable instruction set with metadata and resources |
| **SkillResource** | `SkillResource` | Named supplementary content (static or dynamic) |
| **Parse** | `parse_skill_md(path)` | Load a `SKILL.md` file into a `Skill` |
| **Discover** | `discover_skills(dir)` | Scan a directory tree for skills |
| **Provide** | `SkillsProvider` | Context provider that injects skills into agents |
| **Compose** | `SkillSourceBuilder` | Fluent API for combining, filtering, deduplicating sources |
| **Sources** | `StaticSkillSource`, `DirectorySkillSource`, … | Composable skill source decorators |

Skills follow **progressive disclosure** — advertise cheaply, load on
demand, read resources when needed. This keeps agents focused while
giving them access to a rich library of domain-specific knowledge.

Next, see [24 — Compaction](../24_compaction/24_compaction.qmd) to learn
how to manage conversation length with message compaction strategies.
