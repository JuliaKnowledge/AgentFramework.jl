# Hosting

The `AgentFramework.Hosting` submodule provides a local hosted runtime for running agents and workflows over HTTP. It bundles session persistence, conversation history, workflow checkpointing, and a REST API server into a single `HostedRuntime` object.

## Overview

`HostedRuntime` is a self-contained server that:

- **Registers agents and workflows** by name
- **Manages sessions** with automatic history tracking
- **Runs durable workflows** with checkpointing and human-in-the-loop resume
- **Exposes a REST API** via an HTTP server built on HTTP.jl

## Quick Start

```julia
using AgentFramework
using AgentFramework.Hosting

# Create a runtime (in-memory by default)
runtime = HostedRuntime()

# Or persist to disk
runtime = HostedRuntime(storage_dir)

# Register an agent
agent = Agent(
    name = "helper",
    instructions = "You are a helpful assistant.",
    client = OpenAIChatClient(model = "gpt-4o-mini"),
)
register_agent!(runtime, agent)

# Run via Julia API
result = run_agent!(runtime, "helper", "Hello!")
println(get_text(result.response))
println("Session: ", result.session.id)

# Or start an HTTP server
serve(runtime; host = "127.0.0.1", port = 8080)
```

## HostedRuntime

```julia
Base.@kwdef mutable struct HostedRuntime
    agents::Dict{String, Agent}
    workflows::Dict{String, Workflow}
    session_store::AbstractSessionStore
    history_provider::AbstractHistoryProvider
    checkpoint_storage::AbstractCheckpointStorage
    run_store::AbstractHostedRunStore
    lock::ReentrantLock
end
```

### Constructors

**In-memory runtime** (no persistence):

```julia
runtime = HostedRuntime()
```

**File-backed runtime** (persists sessions, history, checkpoints, and workflow runs to disk):

```julia
runtime = HostedRuntime(storage_dir)
```

The directory-based constructor creates subdirectories for `sessions/`, `history/`, `checkpoints/`, and `runs/`.

## Agent Management

### `register_agent!`

Register an [`Agent`](@ref) with the runtime under a given name.

```julia
register_agent!(runtime, agent)                    # uses agent.name
register_agent!(runtime, agent; name = "my-agent")  # custom name
```

Throws `ArgumentError` if an agent with that name is already registered.

### `list_registered_agents`

```julia
names = list_registered_agents(runtime)  # => ["helper", "reviewer"]
```

### `run_agent!`

Run a registered agent with input, automatically managing session lifecycle.

```julia
result = run_agent!(runtime, "helper", "What is 2+2?")
result = run_agent!(runtime, "helper", "Follow up"; session_id = result.session.id)
```

Returns a named tuple `(response, session, history)`:
- `response` — the [`AgentResponse`](@ref)
- `session` — the [`AgentSession`](@ref) (created or loaded)
- `history` — `Vector{Message}` of the conversation so far

### `get_agent_session`

Retrieve an existing session and its conversation history.

```julia
info = get_agent_session(runtime, "helper", result.session.id)
info.session   # => AgentSession
info.history   # => Vector{Message}
```

### `list_agent_sessions`

```julia
sessions = list_agent_sessions(runtime, "helper")
```

### `delete_agent_session!`

```julia
deleted = delete_agent_session!(runtime, "helper", session_id)  # => true/false
```

## Workflow Management

### `register_workflow!`

Register a [`Workflow`](@ref) with the runtime.

```julia
register_workflow!(runtime, workflow)
register_workflow!(runtime, workflow; name = "my-workflow")
```

### `list_registered_workflows`

```julia
names = list_registered_workflows(runtime)
```

### `start_workflow_run!`

Start a new workflow run. The workflow is deep-copied so the blueprint is not modified.

```julia
run = start_workflow_run!(runtime, "my-workflow", input_data)
run = start_workflow_run!(runtime, "my-workflow", input_data; run_id = "custom-id", metadata = Dict("key" => "value"))
```

Returns a [`HostedWorkflowRun`](#hostedworkflowrun).

### `get_workflow_run`

```julia
run = get_workflow_run(runtime, "my-workflow", run.id)
```

### `list_workflow_runs`

```julia
runs = list_workflow_runs(runtime, "my-workflow")
runs = list_workflow_runs(runtime)  # all workflows
```

### `resume_workflow_run!`

Resume a paused workflow (e.g., after human-in-the-loop approval).

```julia
run = resume_workflow_run!(runtime, "my-workflow", run.id, Dict("request_id" => "approved"))
```

## HostedWorkflowRun

```julia
Base.@kwdef mutable struct HostedWorkflowRun
    id::String
    workflow_name::String
    internal_workflow_name::String
    state::WorkflowRunState
    checkpoint_id::Union{Nothing, String}
    outputs::Vector{Any}
    pending_requests::Vector{Dict{String, Any}}
    events::Vector{Dict{String, Any}}
    error::Union{Nothing, String}
    created_at::DateTime
    updated_at::DateTime
    metadata::Dict{String, Any}
end
```

Use `hosted_workflow_run_to_dict(run)` to serialize a run to a JSON-compatible dictionary.

## Run Stores

The runtime uses an `AbstractHostedRunStore` to persist workflow runs. Two implementations are provided:

### `InMemoryHostedRunStore`

Stores runs in a thread-safe in-memory dictionary. Data is lost when the process exits.

```julia
store = InMemoryHostedRunStore()
```

### `FileHostedRunStore`

Stores each run as a JSON file in a directory.

```julia
store = FileHostedRunStore(runs_dir)
```

Both stores implement:
- `load_run(store, run_id)` — load a run (returns `nothing` if not found)
- `save_run!(store, run)` — persist a run
- `delete_run!(store, run_id)` — delete a run
- `list_runs(store)` / `list_runs(store, workflow_name)` — list runs

## HTTP Server & REST API

### `serve`

Start an HTTP server backed by the runtime.

```julia
serve(runtime; host = "127.0.0.1", port = 8080)
```

### `handle_request`

Process a single `HTTP.Request` and return an `HTTP.Response`. Useful for testing or embedding in a custom server.

```julia
response = handle_request(runtime, request)
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (`{"status": "ok"}`) |
| `GET` | `/agents` | List registered agents |
| `GET` | `/workflows` | List registered workflows |
| `POST` | `/agents/{name}/run` | Run an agent (`{"message": "...", "session_id": "...", "options": {...}}`) |
| `GET` | `/agents/{name}/sessions` | List sessions for an agent |
| `GET` | `/agents/{name}/sessions/{id}` | Get session details and history |
| `DELETE` | `/agents/{name}/sessions/{id}` | Delete a session |
| `POST` | `/workflows/{name}/runs` | Start a workflow run (`{"input": ..., "run_id": "...", "metadata": {...}}`) |
| `GET` | `/workflows/{name}/runs` | List workflow runs |
| `GET` | `/workflows/{name}/runs/{id}` | Get a specific workflow run |
| `POST` | `/workflows/{name}/runs/{id}/resume` | Resume a paused run (`{"responses": {...}}`) |

### Example: Calling the API with `curl`

```bash
# Health check
curl http://localhost:8080/health

# Run an agent
curl -X POST http://localhost:8080/agents/helper/run \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Julia?"}'

# Continue a conversation
curl -X POST http://localhost:8080/agents/helper/run \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me more", "session_id": "abc-123"}'

# Start a workflow
curl -X POST http://localhost:8080/workflows/my-workflow/runs \
  -H "Content-Type: application/json" \
  -d '{"input": "some data"}'
```
