# Mem0 Integration

The `AgentFramework.Mem0Integration` submodule connects [Mem0](https://mem0.ai) semantic memory to AgentFramework.jl agents. It provides a context provider that automatically searches for relevant memories before each agent run and stores new memories after the run completes.

## Overview

Mem0 is a memory layer for AI agents that stores and retrieves conversation context using semantic search. This integration supports both:

- **Mem0 Platform** (hosted SaaS at `api.mem0.ai`) — requires an API key
- **Mem0 OSS** (self-hosted) — runs locally, typically at `http://localhost:8000`

The integration works through two components:

1. **`Mem0Client`** — HTTP client for the Mem0 API (search and store memories)
2. **`Mem0ContextProvider`** — a [`BaseContextProvider`](@ref) that hooks into the agent lifecycle

## Quick Start

```julia
using AgentFramework
using AgentFramework.Mem0Integration

# Create an agent with Mem0 memory
agent = Agent(
    name = "memory-agent",
    instructions = "You are a helpful assistant with memory.",
    chat_client = OpenAIChatClient(model = "gpt-4o-mini"),
    context_providers = [
        Mem0ContextProvider(
            api_key = "your-mem0-api-key",
            user_id = "user-123",
        ),
    ],
)

# First conversation
session = AgentSession()
response = run_agent(agent, "My favorite color is blue."; session = session)

# Later conversation — the agent remembers
response = run_agent(agent, "What's my favorite color?"; session = session)
# The Mem0 context provider searches for relevant memories and injects them
```

## Mem0Client

`Mem0Client` is the low-level HTTP client for communicating with the Mem0 API.

```julia
mutable struct Mem0Client
    api_key::Union{Nothing, String}
    base_url::String
    deployment::Symbol           # :platform or :oss
    request_runner::Function
end
```

### Constructor

```julia
client = Mem0Client(
    api_key = "your-api-key",          # or set MEM0_API_KEY env var
    base_url = nothing,                 # auto-detected from deployment type
    deployment = MEM0_PLATFORM,         # :platform or :oss
)
```

**Credential resolution order:**
1. Explicit `api_key` parameter
2. `MEM0_API_KEY` environment variable

**Base URL resolution order:**
1. Explicit `base_url` parameter
2. `MEM0_BASE_URL` environment variable
3. Default for the deployment type:
   - Platform: `https://api.mem0.ai`
   - OSS: `http://localhost:8000`

### Deployment Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MEM0_PLATFORM` | `:platform` | Mem0 hosted platform (requires API key) |
| `MEM0_OSS` | `:oss` | Self-hosted Mem0 instance |

## Mem0ContextProvider

`Mem0ContextProvider` is a [`BaseContextProvider`](@ref) that integrates Mem0 memory into the agent lifecycle. It implements two hooks:

- **`before_run!`** — searches Mem0 for memories relevant to the user's input and injects them as context messages
- **`after_run!`** — stores the conversation (input + response) as new memories in Mem0

```julia
mutable struct Mem0ContextProvider <: BaseContextProvider
    client::Mem0Client
    source_id::String
    application_id::Union{Nothing, String}
    agent_id::Union{Nothing, String}
    user_id::Union{Nothing, String}
    context_prompt::String
    top_k::Int
    rerank::Union{Nothing, Bool}
    store_roles::Vector{Symbol}
end
```

### Constructor

```julia
provider = Mem0ContextProvider(
    # Connection — provide a client OR individual settings
    client = nothing,                     # pre-configured Mem0Client
    api_key = nothing,                    # or pass connection settings directly
    base_url = nothing,
    deployment = MEM0_PLATFORM,

    # Scoping — at least one of these must be set
    user_id = "user-123",                 # scope memories to a user
    agent_id = nothing,                   # scope to agent (defaults to agent.name)
    application_id = nothing,             # scope to application

    # Behavior
    source_id = "mem0",                   # identifier for this context provider
    context_prompt = "## Memories\nConsider the following memories when answering user questions:",
    top_k = 5,                            # number of memories to retrieve
    rerank = nothing,                     # enable Mem0 reranking
    store_roles = [:user, :assistant, :system],  # which message roles to store
)
```

### Memory Scoping

Mem0 organizes memories by scope. At least one scope identifier must be provided (either directly or resolvable from the session):

| Parameter | Fallback | Description |
|-----------|----------|-------------|
| `user_id` | `session.user_id` | The user whose memories to access |
| `agent_id` | `agent.name` | The agent identity |
| `application_id` | `session.metadata["application_id"]` | The application context |

### How It Works

**Before each agent run (`before_run!`):**

1. Extracts the user's input text from the session context
2. Searches Mem0 for semantically similar memories using the configured scope
3. Formats matching memories with the `context_prompt` header
4. Injects the formatted memories as a user message in the conversation

**After each agent run (`after_run!`):**

1. Collects messages from both the input and the response
2. Filters by `store_roles` (default: user, assistant, system)
3. Sends the messages to Mem0's add endpoint for storage

### Example: Agent with Memory and Session Metadata

```julia
using AgentFramework
using AgentFramework.Mem0Integration

provider = Mem0ContextProvider(
    api_key = ENV["MEM0_API_KEY"],
    top_k = 10,
    rerank = true,
)

agent = Agent(
    name = "personal-assistant",
    instructions = "You are a personal assistant with long-term memory.",
    chat_client = OpenAIChatClient(model = "gpt-4o"),
    context_providers = [provider],
)

# The user_id is resolved from the session
session = AgentSession(user_id = "alice")
response = run_agent(agent, "Remember that I prefer dark mode."; session = session)
```

### Example: Self-Hosted Mem0

```julia
using AgentFramework.Mem0Integration

provider = Mem0ContextProvider(
    deployment = MEM0_OSS,
    base_url = "http://localhost:8000",
    user_id = "local-user",
    top_k = 3,
)
```

## Exception Types

| Exception | Fields | Description |
|-----------|--------|-------------|
| `Mem0Error` | `message`, `status`, `body` | Raised on Mem0 API errors |

```julia
try
    response = run_agent(agent, "Hello"; session = session)
catch e::Mem0Error
    println("Mem0 error (HTTP ", e.status, "): ", e.message)
    e.body !== nothing && println("Response body: ", e.body)
end
```
