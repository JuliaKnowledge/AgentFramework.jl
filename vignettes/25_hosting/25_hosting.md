# Hosting and Agent-to-Agent Protocol
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Part 1 — Hosted Runtime](#part-1--hosted-runtime)
  - [What Is HostedRuntime?](#what-is-hostedruntime)
  - [Creating a Runtime](#creating-a-runtime)
  - [Registering Agents](#registering-agents)
  - [Running Agents Through the
    Runtime](#running-agents-through-the-runtime)
  - [Session Management](#session-management)
  - [Registering and Running
    Workflows](#registering-and-running-workflows)
  - [HTTP Server](#http-server)
- [Part 2 — Agent-to-Agent (A2A)
  Protocol](#part-2--agent-to-agent-a2a-protocol)
  - [What Is A2A?](#what-is-a2a)
  - [A2AClient — Low-Level Protocol
    Access](#a2aclient--low-level-protocol-access)
  - [A2ARemoteAgent — Use Remote Agents Like Local
    Ones](#a2aremoteagent--use-remote-agents-like-local-ones)
  - [Task Lifecycle](#task-lifecycle)
  - [Multi-Agent Architecture](#multi-agent-architecture)
- [Key Types Reference](#key-types-reference)
- [Summary](#summary)

## Overview

Production agent systems need two capabilities beyond running agents
interactively: **hosting** — managing agent and workflow lifecycles with
persistence and HTTP serving — and **inter-agent communication** —
letting agents invoke each other across processes or machines via a
standard protocol.

In this vignette you will learn how to:

- Create a `HostedRuntime` that manages agents and workflows.
- Register agents and run them with persistent sessions.
- Register workflows and inspect workflow runs.
- Serve the runtime over HTTP.
- Connect to a remote agent with `A2AClient` and `A2ARemoteAgent`.
- Understand the A2A task lifecycle.

## Prerequisites

- **Ollama** running locally with a model:

  ``` bash
  ollama pull qwen3:8b
  ```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using AgentFramework
```

## Part 1 — Hosted Runtime

### What Is HostedRuntime?

A `HostedRuntime` is a local runtime container that manages the full
lifecycle of agents and workflows — session persistence, conversation
history, workflow checkpointing, and HTTP serving.

### Creating a Runtime

``` julia
using AgentFramework.Hosting

# In-memory runtime (ephemeral)
runtime = HostedRuntime()

# File-backed runtime (persistent — survives restarts)
runtime = HostedRuntime("./agent_data")
```

### Registering Agents

Register named agents with the runtime. Each agent gets automatic
history provider injection for conversation persistence:

``` julia
client = OllamaChatClient(model = "qwen3:8b")

helper = Agent(
    name = "Helper",
    instructions = "You are a helpful assistant.",
    client = client,
)

coder = Agent(
    name = "Coder",
    instructions = "You write Julia code.",
    client = client,
)

register_agent!(runtime, helper)
register_agent!(runtime, coder)

println(list_registered_agents(runtime))  # ["Coder", "Helper"]
```

### Running Agents Through the Runtime

`run_agent!` creates or loads a session, runs the agent, persists state,
and returns the response with session and conversation history:

``` julia
result = run_agent!(runtime, "Helper", "What is 2+2?")
println(result.response.text)
println(result.session.id)
println(length(result.history))
```

Continue a conversation by passing a `session_id`:

``` julia
result1 = run_agent!(runtime, "Helper", "My name is Alice"; session_id = "session-1")
result2 = run_agent!(runtime, "Helper", "What's my name?"; session_id = "session-1")
println(result2.response.text)  # "Alice"
```

### Session Management

``` julia
# List all sessions for an agent
sessions = list_agent_sessions(runtime, "Helper")

# Get a session with its conversation history
info = get_agent_session(runtime, "Helper", "session-1")
println("Messages: ", length(info.history))

# Delete a session and its history
delete_agent_session!(runtime, "Helper", "session-1")
```

### Registering and Running Workflows

Workflows from `WorkflowBuilder` (see [09 —
Workflows](../09_workflows/09_workflows.qmd)) can also be hosted:

``` julia
register_workflow!(runtime, workflow; name = "TextPipeline")

run = start_workflow_run!(
    runtime,
    "TextPipeline",
    [Message(ROLE_USER, [text_content("hello world")])],
)
println(run.state)    # WF_STARTED, WF_IDLE, WF_COMPLETED, etc.
println(run.outputs)
```

List and inspect runs:

``` julia
all_runs = list_workflow_runs(runtime, "TextPipeline")
run = get_workflow_run(runtime, "TextPipeline", run.id)
```

If a workflow pauses for human input, resume it:

``` julia
resumed = resume_workflow_run!(
    runtime,
    "TextPipeline",
    run.id,
    Dict("request-id-1" => "approved"),
)
println(resumed.state)
```

### HTTP Server

The built-in server exposes the runtime as a REST API:

``` julia
server = serve(runtime; port = 8080, host = "127.0.0.1")
```

| Method   | Path                                 | Description           |
|----------|--------------------------------------|-----------------------|
| `GET`    | `/health`                            | Health check          |
| `GET`    | `/agents`                            | List agents           |
| `POST`   | `/agents/{name}/run`                 | Run an agent          |
| `GET`    | `/agents/{name}/sessions`            | List sessions         |
| `GET`    | `/agents/{name}/sessions/{id}`       | Get session + history |
| `DELETE` | `/agents/{name}/sessions/{id}`       | Delete a session      |
| `POST`   | `/workflows/{name}/runs`             | Start a workflow run  |
| `GET`    | `/workflows/{name}/runs/{id}`        | Get a run             |
| `POST`   | `/workflows/{name}/runs/{id}/resume` | Resume a paused run   |

Example client request:

``` julia
using HTTP, JSON3

response = HTTP.post(
    "http://127.0.0.1:8080/agents/Helper/run",
    ["Content-Type" => "application/json"],
    JSON3.write(Dict("message" => "What is 2+2?", "session_id" => "s1")),
)
body = JSON3.read(String(response.body))
println(body["response"]["text"])
```

## Part 2 — Agent-to-Agent (A2A) Protocol

### What Is A2A?

The [Agent-to-Agent (A2A)](https://a2a-protocol.org/latest/) protocol is
an open standard for agents to communicate across processes, machines,
or organisations. It uses JSON-RPC 2.0 over HTTP, enabling
interoperability between agents built with different frameworks.

    ┌───────────────────┐   JSON-RPC / HTTP   ┌───────────────────┐
    │  A2AClient /       │ ─────────────────►  │  Remote A2A Agent │
    │  A2ARemoteAgent    │ ◄─────────────────  │  (any framework)  │
    └───────────────────┘                     └───────────────────┘

### A2AClient — Low-Level Protocol Access

``` julia
using AgentFramework.A2A

client = A2AClient(
    base_url = "http://localhost:8080",
    poll_interval = 1.0,
    max_polls = 300,
)
```

Fetch the remote agent’s capabilities:

``` julia
card = get_agent_card(client)
println("Agent: $(card.name) — $(card.description)")
```

Send a message and get a response:

``` julia
message = Message(ROLE_USER, [text_content("What is 2+2?")])
response = send_message(client, message)
println(response.text)
```

For long-running tasks, use background mode and poll:

``` julia
message = Message(ROLE_USER, [text_content("Analyse this large dataset")])
response = send_message(client, message; background = true)

if response.continuation_token !== nothing
    println("Task submitted: ", response.continuation_token.task_id)
    completed = wait_for_completion(client, response.continuation_token)
    println("Result: ", completed.text)
end
```

### A2ARemoteAgent — Use Remote Agents Like Local Ones

`A2ARemoteAgent` wraps an A2A endpoint as a standard `AbstractAgent`:

``` julia
remote = A2ARemoteAgent(
    url = "http://localhost:8080",
    name = "RemoteHelper",
    poll_interval = 1.0,
)

response = run_agent(remote, "What is 2+2?")
println(response.text)
```

Sessions maintain conversation context:

``` julia
session = create_session(remote)

response1 = run_agent(remote, "My name is Alice"; session = session)
response2 = run_agent(remote, "What's my name?"; session = session)
println(response2.text)
```

Stream responses:

``` julia
stream = run_agent_streaming(remote, "Tell me a story about Julia")
for update in stream
    print(update.text)
end
println()
```

### Task Lifecycle

A2A tasks follow a state machine:

     SUBMITTED ──► WORKING ──► COMPLETED
                      │
                      ├──► INPUT_REQUIRED  (paused — needs human input)
                      ├──► AUTH_REQUIRED   (paused — needs authentication)
                      ├──► FAILED
                      └──► CANCELED

``` julia
is_terminal_task_state(A2A_TASK_COMPLETED)       # true
is_in_progress_task_state(A2A_TASK_WORKING)      # true
```

When a task is in progress, the response carries a `continuation_token`
for polling:

``` julia
response = send_message(client, message; background = true)

if response.continuation_token !== nothing
    token = response.continuation_token
    println("Task: ", token.task_id)

    # Wait until terminal state
    final = wait_for_completion(client, token)
    println("Result: ", final.text)
end
```

### Multi-Agent Architecture

Combine hosting and A2A for distributed multi-agent systems:

    ┌─────────────────┐    A2A / HTTP     ┌──────────────────────┐
    │  Orchestrator    │ ────────────────► │  Hosted Runtime      │
    │  (A2ARemoteAgent │ ◄──────────────── │    Agent A           │
    │   wrappers)      │                   │    Agent B           │
    └─────────────────┘                   └──────────────────────┘

``` julia
# Server side
server_runtime = HostedRuntime("./server_data")
register_agent!(server_runtime, Agent(
    name = "Specialist",
    instructions = "You are a domain specialist.",
    client = OllamaChatClient(model = "qwen3:8b"),
))
serve(server_runtime; port = 9090)

# Client side
specialist = A2ARemoteAgent(url = "http://localhost:9090", name = "RemoteSpecialist")
response = run_agent(specialist, "Explain quantum computing")
println(response.text)
```

## Key Types Reference

| Type | Module | Description |
|----|----|----|
| `HostedRuntime` | `Hosting` | Runtime container for agents and workflows |
| `HostedWorkflowRun` | `Hosting` | Record of a workflow execution |
| `InMemoryHostedRunStore` | `Hosting` | Ephemeral storage for workflow runs |
| `FileHostedRunStore` | `Hosting` | Persistent file-backed storage for runs |
| `A2AClient` | `A2A` | Low-level A2A protocol client |
| `A2ARemoteAgent` | `A2A` | Remote agent wrapper — use like a local agent |
| `A2AAgentCard` | `A2A` | Remote agent capabilities and metadata |
| `A2ATask` | `A2A` | Task with status, artifacts, and history |
| `A2ATaskState` | `A2A` | Enum: SUBMITTED, WORKING, COMPLETED, etc. |
| `A2AContinuationToken` | `A2A` | Token for polling in-progress tasks |

## Summary

- **`HostedRuntime`** manages agent and workflow lifecycles with session
  persistence, history tracking, and workflow checkpointing.
- Use **in-memory** storage for development and **file-backed** storage
  for production.
- The built-in **HTTP server** (`serve`) exposes agents and workflows as
  REST endpoints.
- The **A2A protocol** enables inter-agent communication across
  processes and frameworks via JSON-RPC 2.0 over HTTP.
- **`A2AClient`** provides low-level protocol access;
  **`A2ARemoteAgent`** wraps remote endpoints so they behave like local
  agents.
- A2A tasks follow a state machine: `SUBMITTED → WORKING → COMPLETED`
  (or `FAILED` / `CANCELED` / `INPUT_REQUIRED`).
- Combine hosting and A2A to build **distributed multi-agent
  architectures**.
