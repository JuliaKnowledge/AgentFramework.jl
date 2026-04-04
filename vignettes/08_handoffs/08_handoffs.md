# Multi-Agent Handoffs
AgentFramework.jl

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [What Are Handoffs?](#what-are-handoffs)
- [Creating Specialist Agents](#creating-specialist-agents)
- [Creating Handoff Tools](#creating-handoff-tools)
- [Exposing Handoffs as Function
  Tools](#exposing-handoffs-as-function-tools)
- [Running the Triage Agent](#running-the-triage-agent)
- [Transfer Instructions](#transfer-instructions)
- [Including Conversation History](#including-conversation-history)
- [Direct Handoff Execution](#direct-handoff-execution)
- [The Handoff Flow](#the-handoff-flow)
- [Handoffs vs. Workflows](#handoffs-vs-workflows)
- [Summary](#summary)

## Overview

A single agent can only do so much. In practice you often want
**specialists** — one agent that triages questions, another that handles
math, a third for creative writing. **Handoffs** let agents delegate
work to each other.

By the end you will know how to:

1.  Create a `HandoffTool` that connects two agents.
2.  Use `handoff_as_function_tool` to expose a handoff as a regular
    tool.
3.  Configure `transfer_instructions` and `include_history`.
4.  Decide when to use handoffs vs. workflows.

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

## What Are Handoffs?

A handoff is a directed link from one agent to another. When a
**triage** agent decides it cannot answer a question itself, it calls a
handoff tool that transfers control to a **specialist** agent. The
specialist runs with the user’s message (and optionally the conversation
history) and returns a response.

    User ──► Triage Agent ──handoff──► Math Expert ──► Response

## Creating Specialist Agents

Let’s build a triage agent that routes to either a **math expert** or a
**general knowledge** agent:

``` julia
client = OllamaChatClient(model = "qwen3:8b")

math_agent = Agent(
    name = "MathExpert",
    description = "Specialist for math and arithmetic questions.",
    instructions = """You are a math expert. Solve problems step by step.
    Show your working clearly. Only answer math questions.""",
    client = client,
)

general_agent = Agent(
    name = "GeneralAssistant",
    description = "Handles general knowledge questions.",
    instructions = """You are a general knowledge assistant.
    Answer questions about history, science, geography, and culture.
    Keep answers concise.""",
    client = client,
)
```

    Agent("GeneralAssistant", 0 tools)

## Creating Handoff Tools

A `HandoffTool` wraps a target agent as a callable tool. The triage
agent sees it in its tool list and can invoke it to delegate:

``` julia
math_handoff = HandoffTool(
    name = "transfer_to_math",
    description = "Transfer to the math expert for math questions.",
    target = math_agent,
)

general_handoff = HandoffTool(
    name = "transfer_to_general",
    description = "Transfer to general assistant for non-math questions.",
    target = general_agent,
)
```

    HandoffTool("transfer_to_general" → "GeneralAssistant")

## Exposing Handoffs as Function Tools

To add handoffs to a triage agent’s tool list, convert them with
`handoff_as_function_tool`:

``` julia
triage_agent = Agent(
    name = "TriageAgent",
    instructions = """You are a routing agent. Your ONLY job is to decide
    which specialist should handle the user's question:
    - For math/arithmetic → use transfer_to_math
    - For everything else → use transfer_to_general
    Always transfer. Never answer directly.""",
    client = client,
    tools = [
        handoff_as_function_tool(math_handoff),
        handoff_as_function_tool(general_handoff),
    ],
)
```

    Agent("TriageAgent", 2 tools)

## Running the Triage Agent

When the triage agent receives a question, it selects the appropriate
handoff tool. The framework automatically executes the handoff and
returns the specialist’s response:

``` julia
r = run_agent(triage_agent, "What is the integral of x^2?")
println(r.text)
```

**Expected output:**

    The integral of x² is (x³)/3 + C.

    Step by step:
    ∫ x² dx = x^(2+1) / (2+1) + C = x³/3 + C

``` julia
r = run_agent(triage_agent, "Who painted the Mona Lisa?")
println(r.text)
```

**Expected output:**

    The Mona Lisa was painted by Leonardo da Vinci, completed around 1519.

## Transfer Instructions

You can add `transfer_instructions` that are prepended to the
specialist’s system prompt when a handoff occurs. This provides context
about what the specialist should focus on:

``` julia
detailed_math_handoff = HandoffTool(
    name = "transfer_to_math_detailed",
    description = "Transfer to math expert with detailed instructions.",
    target = math_agent,
    transfer_instructions = "The user needs a detailed step-by-step solution. "
        * "Show all intermediate calculations.",
)
```

    HandoffTool("transfer_to_math_detailed" → "MathExpert")

## Including Conversation History

By default, `include_history = true` — the specialist receives the full
conversation so far. Set it to `false` for a clean-slate handoff:

``` julia
fresh_handoff = HandoffTool(
    name = "transfer_fresh",
    description = "Transfer with no history.",
    target = general_agent,
    include_history = false,
)
```

    HandoffTool("transfer_fresh" → "GeneralAssistant")

When `include_history = false`, only the current user message is sent to
the specialist. This is useful when the prior conversation might confuse
the specialist or when you want strict isolation.

## Direct Handoff Execution

You can also execute a handoff programmatically without going through
the tool system, using `execute_handoff`:

``` julia
using AgentFramework: execute_handoff, Message

messages = [Message(role = "user", text = "Solve 2x + 3 = 11")]
response = execute_handoff(math_handoff, messages)
println(response.text)
```

**Expected output:**

    2x + 3 = 11
    2x = 8
    x = 4

## The Handoff Flow

Here is what happens when the triage agent invokes a handoff tool:

1.  The triage agent calls `transfer_to_math(message="What is 2+2?")`.
2.  `handoff_as_function_tool` intercepts the call.
3.  `execute_handoff` runs the target agent (`MathExpert`) with the
    message.
4.  The specialist’s response is returned as the tool result.
5.  The triage agent incorporates the result into its own response.

## Handoffs vs. Workflows

| Feature    | Handoffs                    | Workflows                 |
|------------|-----------------------------|---------------------------|
| Topology   | Star (triage → specialists) | DAG (any graph)           |
| Control    | Agent decides at runtime    | Pre-defined edges         |
| Complexity | Simple delegation           | Multi-step pipelines      |
| State      | Conversation-based          | Shared workflow state     |
| Best for   | Routing, delegation         | Pipelines, fan-out/fan-in |

Use **handoffs** when an agent should dynamically choose a specialist.
Use **workflows** (see [09 —
Workflows](../09_workflows/09_workflows.qmd)) when you need a fixed
processing pipeline.

## Summary

1.  **`HandoffTool`** connects a source agent to a target agent.
2.  **`handoff_as_function_tool`** makes a handoff callable as a regular
    tool.
3.  **`execute_handoff`** runs a handoff programmatically.
4.  **`transfer_instructions`** customize the specialist’s behavior per
    handoff.
5.  **`include_history`** controls whether the full conversation is
    forwarded.

Next, see [09 — Workflows](../09_workflows/09_workflows.qmd) to build
DAG-based multi-agent pipelines.
