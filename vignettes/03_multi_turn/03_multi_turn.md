# Multi-Turn Conversations
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Why Sessions Matter](#why-sessions-matter)
- [Creating a Session](#creating-a-session)
- [Multi-Turn Without History
  Provider](#multi-turn-without-history-provider)
- [Using InMemoryHistoryProvider](#using-inmemoryhistoryprovider)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Session State](#session-state)
- [Multiple Concurrent Sessions](#multiple-concurrent-sessions)
- [Summary](#summary)

## Overview

A single prompt-response exchange is rarely enough for real-world
applications. Users expect the agent to remember what was said earlier
in the conversation. This vignette shows how to:

1.  Create an `AgentSession` to persist conversation state.
2.  Pass the session across multiple `run_agent` calls so the agent
    retains context.
3.  Use `InMemoryHistoryProvider` for automatic conversation history
    management.
4.  Store custom state in the session’s `state` dictionary.

## Prerequisites

You need [Ollama](https://ollama.com) running locally with the
`qwen3:8b` model:

``` bash
ollama pull qwen3:8b
```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..",".."))
using AgentFramework
```

## Why Sessions Matter

Without a session, every call to `run_agent` is independent — the model
has no memory of prior exchanges:

    Turn 1:  User → "My name is Alice"       → Agent → "Nice to meet you, Alice!"
    Turn 2:  User → "What is my name?"        → Agent → "I don't know your name."

With a session carrying conversation history, the agent can recall
context:

    Turn 1:  User → "My name is Alice"       → Agent → "Nice to meet you, Alice!"
    Turn 2:  User → "What is my name?"        → Agent → "Your name is Alice!"

## Creating a Session

An `AgentSession` is a lightweight container with a unique ID and a
mutable `state` dictionary:

``` julia
session = AgentSession()
println(session)
println("Session ID: ", session.id)
```

    AgentSession("12f63d43-6e71-4718-9857-9525b87031f8")
    Session ID: 12f63d43-6e71-4718-9857-9525b87031f8

You can also supply your own ID:

``` julia
custom_session = AgentSession(id = "my-conversation-001")
println(custom_session)
```

    AgentSession("my-conversation-001")

## Multi-Turn Without History Provider

The simplest approach passes the same session to each `run_agent` call.
However, without a history provider the session alone does not replay
prior messages — you would need to send the full conversation manually.
Let us first see this limitation, then solve it properly.

``` julia
client = OllamaChatClient(model = "qwen3:8b")

basic_agent = Agent(
    name = "BasicAgent",
    instructions = "You are a friendly assistant. Keep your answers brief.",
    client = client,
)
```

    Agent("BasicAgent", 0 tools)

``` julia
session = create_session(basic_agent)

# Turn 1
response1 = run_agent(basic_agent, "My name is Alice and I love hiking.", session = session)
println("Turn 1: ", response1.text)

# Turn 2 — without a history provider the agent has no memory of Turn 1
response2 = run_agent(basic_agent, "What do you remember about me?", session = session)
println("Turn 2: ", response2.text)
```

**Expected output (without history):**

    Turn 1: Nice to meet you, Alice! Hiking is a wonderful hobby.
    Turn 2: I don't have any prior information about you.

## Using InMemoryHistoryProvider

The `InMemoryHistoryProvider` is a *context provider* that automatically
stores and replays conversation history. Attach it to the agent via
`context_providers`:

``` julia
history = InMemoryHistoryProvider()

agent = Agent(
    name = "ConversationAgent",
    instructions = "You are a friendly assistant. Keep your answers brief.",
    client = client,
    context_providers = [history],
)
println(agent)
```

    Agent("ConversationAgent", 0 tools)

Now the session + history provider work together to maintain full
conversation context:

``` julia
session = create_session(agent)

# Turn 1
response1 = run_agent(agent, "My name is Alice and I love hiking.", session = session)
println("Turn 1: ", response1.text)

# Turn 2 — the history provider replays Turn 1 automatically
response2 = run_agent(agent, "What do you remember about me?", session = session)
println("Turn 2: ", response2.text)

# Turn 3 — continues building on the conversation
response3 = run_agent(agent, "Suggest a hiking trail for me.", session = session)
println("Turn 3: ", response3.text)
```

**Expected output:**

    Turn 1: Nice to meet you, Alice! Hiking is a wonderful hobby.
    Turn 2: Your name is Alice and you love hiking!
    Turn 3: You might enjoy the Appalachian Trail — it has beautiful scenery and
            trails for all skill levels.

## How It Works Under the Hood

The `InMemoryHistoryProvider` implements two lifecycle hooks:

- **`before_run!`** — retrieves stored messages for the session and
  injects them as context messages so the LLM sees the full conversation
  history.
- **`after_run!`** — saves both the new user message and the assistant’s
  response into the history store.

This means each call to `run_agent` automatically builds on previous
turns without any manual message management.

## Session State

The `session.state` dictionary lets you store arbitrary data that
persists across turns. This is useful for tracking application-level
state such as user preferences, accumulated results, or workflow
progress:

``` julia
session = AgentSession()

# Store custom state
session.state["user_name"] = "Alice"
session.state["turn_count"] = 0
session.state["preferences"] = Dict("units" => "metric", "language" => "en")

println("State: ", session.state)
```

    State: Dict{String, Any}("preferences" => Dict("units" => "metric", "language" => "en"), "user_name" => "Alice", "turn_count" => 0)

You can read and update state between turns:

``` julia
session = create_session(agent)
session.state["turn_count"] = 0

for prompt in ["Hello, I'm Bob.", "What's my name?", "Tell me a joke."]
    session.state["turn_count"] += 1
    response = run_agent(agent, prompt, session = session)
    println("Turn $(session.state["turn_count"]): $(response.text)")
end

println("\nTotal turns: ", session.state["turn_count"])
```

**Expected output:**

    Turn 1: Hello Bob! Nice to meet you.
    Turn 2: Your name is Bob!
    Turn 3: Why don't scientists trust atoms? Because they make up everything!

    Total turns: 3

## Multiple Concurrent Sessions

The `InMemoryHistoryProvider` stores history keyed by session ID, so you
can run multiple independent conversations simultaneously:

``` julia
session_a = create_session(agent; session_id = "session-a")
session_b = create_session(agent; session_id = "session-b")

run_agent(agent, "My name is Alice.", session = session_a)
run_agent(agent, "My name is Bob.", session = session_b)

resp_a = run_agent(agent, "What is my name?", session = session_a)
resp_b = run_agent(agent, "What is my name?", session = session_b)

println("Session A: ", resp_a.text)  # "Your name is Alice!"
println("Session B: ", resp_b.text)  # "Your name is Bob!"
```

## Summary

| Concept | Type | Purpose |
|----|----|----|
| **Session** | `AgentSession` | Holds a unique ID and mutable `state` dict |
| **History provider** | `InMemoryHistoryProvider` | Automatically stores and replays conversation messages |
| **`session.state`** | `Dict{String, Any}` | User-managed key-value store for custom data |

The multi-turn pattern is:

1.  Create a session with `create_session(agent)`.
2.  Attach an `InMemoryHistoryProvider` to the agent’s
    `context_providers`.
3.  Pass `session = session` to every `run_agent` call — history is
    managed automatically.

This forms the foundation for building chatbots, interactive assistants,
and multi-step workflows where context continuity is essential.
