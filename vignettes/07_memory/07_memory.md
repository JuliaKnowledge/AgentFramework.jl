# Memory and Context Providers
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [The Problem: Stateless Agents](#the-problem-stateless-agents)
- [InMemoryHistoryProvider](#inmemoryhistoryprovider)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [FileHistoryProvider — Persist to
  Disk](#filehistoryprovider--persist-to-disk)
- [Custom Context Provider](#custom-context-provider)
- [The Context Provider Protocol](#the-context-provider-protocol)
- [Summary](#summary)

## Overview

Agents are stateless by default — each call to `run_agent` starts with a
blank slate. Real assistants, however, remember what you said five
minutes ago. This vignette shows how to give agents **persistent
memory** using context providers.

By the end you will know how to:

1.  Use `InMemoryHistoryProvider` for automatic conversation history.
2.  Use `FileHistoryProvider` to persist history to disk.
3.  Write a custom context provider with the `before_run!`/`after_run!`
    protocol.

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

## The Problem: Stateless Agents

Without memory, every call is independent. The agent cannot recall
earlier exchanges:

``` julia
client = OllamaChatClient(model = "qwen3:8b")

agent_no_memory = Agent(
    name = "ForgetfulBot",
    instructions = "You are a helpful assistant. Keep answers brief.",
    client = client,
)
```

    Agent("ForgetfulBot", 0 tools)

``` julia
r1 = run_agent(agent_no_memory, "My name is Alice.")
println(r1.text)

r2 = run_agent(agent_no_memory, "What is my name?")
println(r2.text)
```

**Expected output:**

    Nice to meet you, Alice!
    I'm sorry, I don't know your name. Could you tell me?

The second call has no idea who Alice is. Let’s fix that.

## InMemoryHistoryProvider

`InMemoryHistoryProvider` automatically captures every user/assistant
exchange and replays the history on subsequent calls. Add it via
`context_providers`:

``` julia
memory_agent = Agent(
    name = "MemoryBot",
    instructions = "You are a helpful assistant. Keep answers brief.",
    client = client,
    context_providers = [InMemoryHistoryProvider()],
)
```

    Agent("MemoryBot", 0 tools)

Now create a **session** — this is the container that ties multiple
calls together:

``` julia
session = create_session(memory_agent)

r1 = run_agent(memory_agent, "My name is Alice.", session = session)
println("Turn 1: ", r1.text)

r2 = run_agent(memory_agent, "What is my name?", session = session)
println("Turn 2: ", r2.text)
```

**Expected output:**

    Turn 1: Nice to meet you, Alice!
    Turn 2: Your name is Alice.

The provider intercepts each run via the `before_run!`/`after_run!`
hooks:

- **`before_run!`** — loads stored messages into the session context.
- **`after_run!`** — saves the new user and assistant messages.

## How It Works Under the Hood

The `InMemoryHistoryProvider` stores messages in a
`Dict{String, Vector{Message}}` keyed by session ID. Each time the agent
runs:

1.  `before_run!` calls `get_messages(provider, session_id)` and inserts
    them into the `SessionContext` via `extend_messages!`.
2.  The agent sees the full conversation history in its prompt.
3.  `after_run!` calls `save_messages!(provider, session_id, messages)`
    with the new user input and assistant response.

``` julia
# Inspect accumulated history
provider = memory_agent.context_providers[1]
msgs = get_messages(provider, session.id)
println("Stored messages: ", length(msgs))
for m in msgs
    println("  [$(m.role)] $(m.text)")
end
```

**Expected output:**

    Stored messages: 4
      [user] My name is Alice.
      [assistant] Nice to meet you, Alice!
      [user] What is my name?
      [assistant] Your name is Alice.

## FileHistoryProvider — Persist to Disk

`InMemoryHistoryProvider` loses its data when the process exits. For
persistence across restarts, use `FileHistoryProvider`:

``` julia
file_agent = Agent(
    name = "PersistentBot",
    instructions = "You are a helpful assistant. Keep answers brief.",
    client = client,
    context_providers = [FileHistoryProvider(directory = "/tmp/agent_history")],
)

session = create_session(file_agent)
run_agent(file_agent, "Remember: my favorite color is blue.", session = session)

# Later (even after restarting Julia):
run_agent(file_agent, "What is my favorite color?", session = session)
```

Files are stored as JSON, one per session ID, in the specified
directory.

## Custom Context Provider

You can create your own provider by implementing `before_run!` and
`after_run!`. Here is a provider that injects the current date and time:

``` julia
using Dates

struct DateTimeProvider <: AgentFramework.BaseContextProvider
    source_id::String
end
DateTimeProvider() = DateTimeProvider("datetime")

function AgentFramework.before_run!(
    provider::DateTimeProvider, agent, session, ctx, state
)
    now_str = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    extend_instructions!(ctx, "Current date and time: $now_str")
end

# after_run! is optional — omit it if you don't need post-processing.
```

``` julia
datetime_agent = Agent(
    name = "TimeAwareBot",
    instructions = "You are a helpful assistant. Keep answers brief.",
    client = client,
    context_providers = [DateTimeProvider(), InMemoryHistoryProvider()],
)

session = create_session(datetime_agent)
r = run_agent(datetime_agent, "What time is it right now?", session = session)
println(r.text)
```

**Expected output:**

    The current date and time is 2025-01-15 14:30:00.

Multiple context providers compose naturally — the agent sees
instructions and history from all of them.

## The Context Provider Protocol

Every context provider follows a two-phase lifecycle:

| Phase | Method | Purpose |
|----|----|----|
| Before | `before_run!(provider, agent, session, ctx, state)` | Inject context (instructions, messages, tools) |
| After | `after_run!(provider, agent, session, ctx, state)` | Persist results, update state |

The `ctx` argument is a `SessionContext` with these key methods:

- `extend_messages!(ctx, source_id, messages)` — add context messages
- `extend_instructions!(ctx, instruction)` — add system instructions
- `extend_tools!(ctx, tools)` — add tools dynamically

The `state` argument is a mutable `Dict{String, Any}` stored on the
session, allowing providers to stash data between runs.

## Summary

| Provider | Storage | Persistence | Use Case |
|----|----|----|----|
| `InMemoryHistoryProvider` | Dict in memory | Process lifetime | Prototyping, tests |
| `FileHistoryProvider` | JSON files on disk | Across restarts | Local development |
| `DBInterfaceHistoryProvider` | SQL database | Permanent | Production |
| Custom provider | You choose | You choose | Domain-specific context |

Key takeaways:

1.  **Sessions** tie multiple `run_agent` calls together.
2.  **Context providers** inject history, instructions, or tools before
    each run.
3.  The `before_run!`/`after_run!` protocol is simple to implement for
    custom needs.

Next, see [08 — Handoffs](../08_handoffs/08_handoffs.qmd) to learn how
agents can delegate work to other agents.
