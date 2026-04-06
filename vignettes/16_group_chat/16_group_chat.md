# Group Chat Orchestrations
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Group Chat Patterns at a Glance](#group-chat-patterns-at-a-glance)
- [Creating the Agents](#creating-the-agents)
- [Round-Robin Group Chat](#round-robin-group-chat)
- [Selector-Based Group Chat](#selector-based-group-chat)
  - [Using an Orchestrator Agent](#using-an-orchestrator-agent)
  - [Using a Custom Selection
    Function](#using-a-custom-selection-function)
- [Magentic Orchestration](#magentic-orchestration)
  - [Building a Magentic Workflow](#building-a-magentic-workflow)
  - [Running the Magentic Workflow](#running-the-magentic-workflow)
- [Termination Conditions](#termination-conditions)
- [Summary](#summary)

## Overview

When multiple agents need to collaborate on a problem — debating,
reviewing, or building on each other’s ideas — you need a **group
chat**. AgentFramework provides three orchestration patterns with
increasing sophistication.

By the end you will know how to:

1.  Run a **round-robin** group chat where agents take turns in
    sequence.
2.  Use a **selector-based** group chat with an orchestrator agent or
    function.
3.  Build a **Magentic** orchestration with planning and stall
    detection.
4.  Inspect `GroupChatState` and `MagenticContext` during orchestration.
5.  Customise termination conditions and selection functions.

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

## Group Chat Patterns at a Glance

The three patterns share the same execution model — a workflow of agents
communicating through a shared conversation — but differ in how the next
speaker is chosen:

| Pattern | Selection Method | Best For |
|----|----|----|
| Round-Robin | Fixed rotation through participants | Simple brainstorming, review cycles |
| Selector | LLM agent or custom function | Dynamic expert routing |
| Magentic | Manager with task ledger + progress tracking | Complex multi-step tasks |

## Creating the Agents

We will use three philosophical personas that debate a topic: an
**Optimist**, a **Pessimist**, and a **Moderator**.

``` julia
client = OllamaChatClient(model = "qwen3:8b")

optimist = Agent(
    name = "Optimist",
    instructions = "You are an optimistic philosopher. You see the bright side of every argument. Keep responses to 2-3 sentences.",
    client = client,
)

pessimist = Agent(
    name = "Pessimist",
    instructions = "You are a pessimistic philosopher. You challenge assumptions and point out risks. Keep responses to 2-3 sentences.",
    client = client,
)

moderator = Agent(
    name = "Moderator",
    instructions = "You are a neutral moderator. Summarise the discussion so far and ask a probing follow-up question. Keep responses to 2-3 sentences.",
    client = client,
)
```

## Round-Robin Group Chat

The simplest pattern: agents speak in the order they are listed, cycling
through the list until a maximum number of rounds is reached.

``` julia
group = GroupChatBuilder(
    participants = [optimist, pessimist, moderator],
    max_rounds = 6,
    intermediate_outputs = true,
    name = "PhilosophyRoundRobin",
)

workflow = build(group)
```

Run the workflow with a debate topic:

``` julia
result = run_workflow(workflow, "Is technology making humanity better or worse?")

for evt in result.events
    if evt.type == :executor_completed && hasattr(evt, :data)
        println("[$(evt.executor_id)] $(evt.data)\n")
    end
end
```

**Expected output:**

    [Optimist] Technology has connected billions of people, enabling unprecedented
    access to education and healthcare.

    [Pessimist] Yet that same connectivity breeds isolation and surveillance. We
    trade privacy and genuine human connection for convenience.

    [Moderator] Both raise valid points. How do we measure whether the net effect
    is positive — individual well-being or collective progress?
    ...

Internally, `GroupChatBuilder` without a `selection_func` or
`orchestrator_agent` defaults to round-robin, equivalent to:

``` julia
round_robin_select = state -> begin
    idx = mod1(state.round + 1, length(state.participant_ids))
    state.participant_ids[idx]
end
```

## Selector-Based Group Chat

For more dynamic conversations, an **orchestrator agent** decides who
speaks next based on the conversation so far. This is useful when you
want experts to be called upon only when relevant.

### Using an Orchestrator Agent

``` julia
coordinator = Agent(
    name = "Coordinator",
    instructions = """You are a debate coordinator. Given the conversation so
    far, decide which participant should speak next. Reply with ONLY the name:
    Optimist, Pessimist, or Moderator.""",
    client = client,
)

group = GroupChatBuilder(
    participants = [optimist, pessimist, moderator],
    orchestrator_agent = coordinator,
    max_rounds = 10,
    termination_condition = state -> state.round >= 8,
    intermediate_outputs = true,
    name = "PhilosophySelector",
)

workflow = build(group)
```

``` julia
result = run_workflow(workflow, "Should artificial intelligence have rights?")

for evt in result.events
    if evt.type == :executor_completed && hasattr(evt, :data)
        println("[$(evt.executor_id)] $(evt.data)\n")
    end
end
```

**Expected output:**

    [Optimist] If AI reaches self-awareness and can experience suffering, denying
    it rights would repeat the moral failures of history.

    [Pessimist] Self-awareness is a philosophical minefield — we can barely define
    consciousness in humans.

    [Moderator] The debate hinges on defining consciousness. Should rights be tied
    to subjective experience, or could functional equivalence suffice?
    ...

### Using a Custom Selection Function

Instead of an LLM, provide a deterministic selection function:

``` julia
custom_select = state -> begin
    if state.last_speaker == "Optimist"     "Pessimist"
    elseif state.last_speaker == "Pessimist" "Moderator"
    else                                     "Optimist"
    end
end

group = GroupChatBuilder(
    participants = [optimist, pessimist, moderator],
    selection_func = custom_select,
    max_rounds = 9,
)
workflow = build(group)
```

The `GroupChatState` passed to selection and termination functions
contains: `conversation::Vector{Message}`,
`participant_ids::Vector{String}`, `round::Int`,
`last_speaker::Union{Nothing, String}`, and
`metadata::Dict{String, Any}`.

## Magentic Orchestration

For complex multi-step tasks, the **Magentic** pattern adds a planning
layer. A manager creates a task ledger, selects participants, tracks
progress, and replans if the conversation stalls.

      Plan → Select participant → Response → Update progress
       ▲         Stalled? replan ──┘         Done? finalize

### Building a Magentic Workflow

``` julia
researcher = Agent(
    name = "Researcher",
    instructions = "You are a philosophical researcher. Provide relevant historical context, cite key thinkers, and identify core tensions. Keep responses to 3-4 sentences.",
    client = client,
)

analyst = Agent(
    name = "Analyst",
    instructions = "You are a philosophical analyst. Evaluate arguments for logical consistency, identify fallacies, and suggest stronger formulations. Keep responses to 3-4 sentences.",
    client = client,
)

magentic = MagenticBuilder(
    participants = [researcher, analyst],
    manager = StandardMagenticManager(),
    max_stall_count = 3,
    max_round_count = 10,
    intermediate_outputs = true,
    name = "PhilosophyMagentic",
)

workflow = build(magentic)
```

### Running the Magentic Workflow

``` julia
result = run_workflow(
    workflow,
    "Analyse the trolley problem and its implications for autonomous vehicles.",
)

for evt in result.events
    if evt.type == :executor_completed && hasattr(evt, :data)
        println("[$(evt.executor_id)] $(evt.data)\n")
    end
end

for output in get_outputs(result)
    println("=== Final Answer ===\n", output)
end
```

**Expected output:**

    [Researcher] The trolley problem, introduced by Philippa Foot in 1967, explores
    whether it is morally permissible to divert harm from many to few.

    [Analyst] The utilitarian framing assumes we can quantify lives equally, but
    autonomous vehicles face uncertainty — probabilities, not certainties.

    === Final Answer ===
    The trolley problem reveals tensions between utilitarian and deontological
    ethics that become acute in autonomous vehicle design. A pragmatic approach
    combines virtue ethics with transparent, auditable decision rules.

`StandardMagenticManager` has configurable `planner`, `selector`, and
`finalizer` hooks:

``` julia
custom_manager = StandardMagenticManager(
    planner = ctx -> MagenticTaskLedger(
        facts = ["Ethics and AI topic"], plan = ["Research", "Analyse", "Synthesise"], current_step = 1,
    ),
    selector = ctx -> ctx.task_ledger.current_step <= 1 ? "Researcher" : "Analyst",
    finalizer = ctx -> [Message(role = :assistant, contents = [TextContent("Final synthesis...")])],
)

magentic = MagenticBuilder(participants = [researcher, analyst], manager = custom_manager,
    max_stall_count = 3, max_round_count = 8)
workflow = build(magentic)
```

## Termination Conditions

All group chat patterns support custom termination via a function that
receives the current state and returns `true` to stop:

``` julia
stop_on_consensus = state -> any(
    occursin("consensus", lowercase(msg.text))
    for msg in state.conversation if msg.text !== nothing
)

group = GroupChatBuilder(
    participants = [optimist, pessimist, moderator],
    termination_condition = state -> stop_on_consensus(state) || state.round >= 5,
    max_rounds = 20,   # hard safety limit
    name = "TerminationDemo",
)
```

## Summary

| Pattern | Builder | Selection | Replanning | Use Case |
|----|----|----|----|----|
| Round-Robin | `GroupChatBuilder` | Fixed rotation | No | Simple multi-agent discussion |
| Selector | `GroupChatBuilder` + `orchestrator_agent` | LLM or function | No | Dynamic expert routing |
| Magentic | `MagenticBuilder` | Manager with progress ledger | Yes (on stall) | Complex task decomposition |

Key takeaways:

- **`GroupChatBuilder`** handles both round-robin and selector patterns.
- **`MagenticBuilder`** adds planning, progress tracking, and
  replanning.
- All patterns produce a standard `Workflow` via `build()` and use the
  same `run_workflow` / event inspection API.
- Use `intermediate_outputs = true` to stream agent responses as they
  happen.

Next, see [17 — Evaluation](../17_evaluation/17_evaluation.qmd) to learn
how to measure and benchmark your agents.
