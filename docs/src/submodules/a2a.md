# A2A Protocol

The `AgentFramework.A2A` submodule implements the [Agent-to-Agent (A2A)](https://google.github.io/A2A/) interoperability protocol, enabling communication between AgentFramework.jl agents and any A2A-compatible remote agent over HTTP using JSON-RPC.

The module provides two primary entry points:

- **[`A2AClient`](#AgentFramework.A2A.A2AClient)** — a low-level HTTP client for sending messages, retrieving tasks, and fetching agent cards from an A2A server.
- **[`A2ARemoteAgent`](#AgentFramework.A2A.A2ARemoteAgent)** — a high-level [`AbstractAgent`](@ref) wrapper that lets you use a remote A2A agent with the standard `run_agent` / `run_agent_streaming` interface.

## Quick Start

```julia
using AgentFramework
using AgentFramework.A2A

# Connect to a remote A2A agent
client = A2AClient(base_url = "https://my-a2a-server.example.com")

# Fetch the agent's capabilities
card = get_agent_card(client)
println(card.name, ": ", card.description)

# Send a message and wait for completion
response = AgentFramework.A2A.send_message(client, Message(:user, "Hello, remote agent!"))
println(get_text(response))
```

### Using A2ARemoteAgent with `run_agent`

```julia
using AgentFramework
using AgentFramework.A2A

# Wrap a remote agent so it works like any local agent
agent = A2ARemoteAgent(url = "https://my-a2a-server.example.com", name = "Remote Helper")

# Use the standard AgentFramework interface
response = run_agent(agent, "Summarize the latest news")
println(get_text(response))

# Streaming is also supported
stream = run_agent_streaming(agent, "Tell me a story")
for update in stream
    print(get_text(update))
end
```

## Task States

A2A tasks progress through a lifecycle of states. The module represents these with the `A2ATaskState` enum:

| Constant | Description |
|----------|-------------|
| `A2A_TASK_SUBMITTED` | Task has been submitted but not yet started |
| `A2A_TASK_WORKING` | Task is actively being processed |
| `A2A_TASK_INPUT_REQUIRED` | Task is paused waiting for additional user input |
| `A2A_TASK_AUTH_REQUIRED` | Task is paused waiting for authentication |
| `A2A_TASK_COMPLETED` | Task finished successfully |
| `A2A_TASK_FAILED` | Task failed with an error |
| `A2A_TASK_CANCELED` | Task was canceled |
| `A2A_TASK_REJECTED` | Task was rejected by the remote agent |
| `A2A_TASK_UNKNOWN` | Unknown or unrecognized state |

### Task State Helpers

```julia
# Convert between states and strings
task_state_string(A2A_TASK_COMPLETED)     # => "completed"
parse_task_state("working")               # => A2A_TASK_WORKING

# Check state categories
is_terminal_task_state(A2A_TASK_COMPLETED)      # => true
is_in_progress_task_state(A2A_TASK_WORKING)     # => true
```

## Protocol Types

### A2AAgentCard

Describes the capabilities and metadata of a remote A2A agent. Retrieved via `get_agent_card`.

```julia
Base.@kwdef mutable struct A2AAgentCard
    name::Union{Nothing, String} = nothing
    description::Union{Nothing, String} = nothing
    url::String
    version::Union{Nothing, String} = nothing
    default_input_modes::Vector{String} = String[]
    default_output_modes::Vector{String} = String[]
    capabilities::Dict{String, Any} = Dict{String, Any}()
    skills::Vector{Dict{String, Any}} = Dict{String, Any}[]
    additional_properties::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end
```

### A2ATask

Represents a task on the remote A2A server, including its status, artifacts, and history.

```julia
Base.@kwdef mutable struct A2ATask
    id::String
    context_id::Union{Nothing, String} = nothing
    status::A2ATaskStatus = A2ATaskStatus()
    artifacts::Vector{A2AArtifact} = A2AArtifact[]
    history::Vector{Message} = Message[]
    metadata::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end
```

### A2ATaskStatus

```julia
Base.@kwdef mutable struct A2ATaskStatus
    state::A2ATaskState = A2A_TASK_UNKNOWN
    message::Union{Nothing, Message} = nothing
    timestamp::Union{Nothing, String} = nothing
    raw_representation::Any = nothing
end
```

### A2AArtifact

Represents an output artifact from a completed A2A task.

```julia
Base.@kwdef mutable struct A2AArtifact
    artifact_id::Union{Nothing, String} = nothing
    name::Union{Nothing, String} = nothing
    description::Union{Nothing, String} = nothing
    contents::Vector{Content} = Content[]
    additional_properties::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end
```

### A2AContinuationToken

Used internally to track tasks that are still in progress and may need polling.

```julia
Base.@kwdef mutable struct A2AContinuationToken
    task_id::String
    context_id::Union{Nothing, String} = nothing
end
```

## A2AClient

The `A2AClient` is a low-level HTTP client for the A2A JSON-RPC protocol.

```julia
client = A2AClient(
    base_url = "https://my-agent.example.com",
    timeout = 60.0,           # HTTP timeout in seconds
    headers = nothing,        # optional custom headers
    poll_interval = 1.0,      # seconds between poll attempts
    max_polls = 300,          # maximum number of polls before timeout
)
```

### `get_agent_card`

Fetches the agent card from the `/.well-known/agent.json` endpoint.

```julia
card = get_agent_card(client)
println(card.name)
```

### `send_message`

Sends a [`Message`](@ref) to the remote agent. By default, blocks until the task completes. Set `background = true` to return immediately with a continuation token.

```julia
# Blocking — waits for completion
response = AgentFramework.A2A.send_message(client, Message(:user, "Hello"))

# Non-blocking — returns immediately, may include a continuation token
response = AgentFramework.A2A.send_message(client, Message(:user, "Hello"); background = true)
```

**Keyword arguments:**
- `context_id` — optional context/session ID for the A2A conversation
- `reference_task_ids` — vector of related task IDs
- `background` — if `true`, return immediately without polling
- `poll_interval` — override the client's default poll interval
- `max_polls` — override the client's default max polls

### `get_task`

Retrieves the current state of a task by its continuation token or task ID.

```julia
response = AgentFramework.A2A.get_task(client, token)
response = AgentFramework.A2A.get_task(client, "task-id-string")
```

### `wait_for_completion`

Polls a task until it reaches a terminal state or exceeds `max_polls`.

```julia
response = AgentFramework.A2A.wait_for_completion(client, token; poll_interval = 2.0, max_polls = 100)
```

## A2ARemoteAgent

`A2ARemoteAgent` is an [`AbstractAgent`](@ref) that wraps an `A2AClient`, providing integration with the standard AgentFramework `run_agent` and `run_agent_streaming` methods.

```julia
agent = A2ARemoteAgent(
    url = "https://my-agent.example.com",    # or pass client = A2AClient(...)
    name = "Remote Agent",
    description = "A remote A2A-compatible agent",
    context_providers = [],                   # optional context providers
    poll_interval = 1.0,
    max_polls = nothing,                      # nothing = use client default
)
```

### `run_agent`

```julia
response = run_agent(agent, "What's the weather?"; session = session)
```

Sends the input messages to the remote agent and returns an [`AgentResponse`](@ref). Session state is automatically updated with the A2A context and task IDs.

### `run_agent_streaming`

```julia
stream = run_agent_streaming(agent, "Tell me a joke")
for update in stream
    print(get_text(update))
end
```

Returns a `ResponseStream{AgentResponseUpdate}` that yields incremental updates by polling the remote task.

### `poll_task`

Manually poll a task by its continuation token.

```julia
response = AgentFramework.A2A.poll_task(agent, token; session = session, background = true)
```

## Conversion Utilities

These functions convert between AgentFramework types and A2A JSON-RPC representations:

| Function | Description |
|----------|-------------|
| `message_to_a2a_dict(message)` | Convert a [`Message`](@ref) to an A2A message dictionary |
| `a2a_message_to_message(dict)` | Convert an A2A message dictionary to a [`Message`](@ref) |
| `a2a_agent_card_from_dict(dict)` | Parse an A2A agent card from a dictionary |
| `a2a_task_from_dict(dict)` | Parse an A2A task from a dictionary |
| `task_to_response(task)` | Convert an `A2ATask` to an [`AgentResponse`](@ref) |
| `continuation_token_to_dict(token)` | Serialize an `A2AContinuationToken` to a dictionary |
| `continuation_token_from_dict(dict)` | Deserialize an `A2AContinuationToken` from a dictionary |

## Exception Types

| Exception | Description |
|-----------|-------------|
| `A2AError` | General A2A error with an optional inner exception |
| `A2AProtocolError` | JSON-RPC protocol-level error (includes the method name) |
| `A2ATaskError` | Task-specific error (includes the task ID) |
| `A2ATimeoutError` | Raised when polling exceeds `max_polls` |

```julia
try
    response = AgentFramework.A2A.send_message(client, Message(:user, "Hello"))
catch e
    if e isa A2ATimeoutError
        println("Task timed out: ", e.message)
    elseif e isa A2AProtocolError
        println("Protocol error on method ", e.method, ": ", e.message)
    else
        rethrow()
    end
end
```
