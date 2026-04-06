# Bedrock

The `AgentFramework.Bedrock` submodule provides Amazon Bedrock support for AgentFramework.jl, including a chat client for conversational AI models and an embedding client for text embeddings. Both clients use AWS Signature V4 authentication and the Bedrock Converse API.

## Overview

This module exports three types:

- **`BedrockChatClient`** — an [`AbstractChatClient`](@ref) for Bedrock conversational models (Claude, Titan, Llama, etc.)
- **`BedrockEmbeddingClient`** — a client for Bedrock text embedding models (Titan Embeddings, etc.)
- **`BedrockCredentials`** — explicit AWS credentials for authentication

## Quick Start

```julia
using AgentFramework
using AgentFramework.Bedrock

# Create a chat client (credentials from environment or ~/.aws/credentials)
client = BedrockChatClient(model = "anthropic.claude-3-haiku-20240307-v1:0")

# Use with an agent
agent = Agent(
    name = "bedrock-agent",
    instructions = "You are a helpful assistant.",
    chat_client = client,
)

response = run_agent(agent, "What is Amazon Bedrock?")
println(get_text(response))
```

## AWS Authentication

The Bedrock clients support multiple authentication methods, resolved in this priority order:

### 1. Explicit Credentials

Pass credentials directly to the client:

```julia
client = BedrockChatClient(
    model = "anthropic.claude-3-haiku-20240307-v1:0",
    access_key_id = "AKIA...",
    secret_access_key = "secret...",
    session_token = "token...",  # optional, for temporary credentials
)
```

Or use a `BedrockCredentials` object:

```julia
creds = BedrockCredentials(
    access_key_id = "AKIA...",
    secret_access_key = "secret...",
    session_token = nothing,
)

client = BedrockChatClient(
    model = "anthropic.claude-3-haiku-20240307-v1:0",
    credentials = creds,
)
```

### 2. Environment Variables

The following environment variables are checked (in order):

| Purpose | Variables (checked in order) |
|---------|------------------------------|
| Access Key | `BEDROCK_ACCESS_KEY_ID`, `BEDROCK_ACCESS_KEY`, `AWS_ACCESS_KEY_ID` |
| Secret Key | `BEDROCK_SECRET_ACCESS_KEY`, `BEDROCK_SECRET_KEY`, `AWS_SECRET_ACCESS_KEY` |
| Session Token | `BEDROCK_SESSION_TOKEN`, `AWS_SESSION_TOKEN` |
| Region | `BEDROCK_REGION`, `AWS_REGION`, `AWS_DEFAULT_REGION` |

### 3. AWS Profile

Falls back to `~/.aws/credentials` and `~/.aws/config`:

```julia
client = BedrockChatClient(
    model = "anthropic.claude-3-haiku-20240307-v1:0",
    profile = "my-profile",  # defaults to AWS_PROFILE or "default"
)
```

### Region Resolution

The AWS region is resolved in this order:

1. Explicit `region` parameter on the client
2. `BEDROCK_REGION` / `AWS_REGION` / `AWS_DEFAULT_REGION` environment variables
3. `region` setting in the AWS profile configuration
4. Default: `us-east-1`

## BedrockChatClient

```julia
Base.@kwdef mutable struct BedrockChatClient <: AbstractChatClient
    model::String = ""                      # or set BEDROCK_CHAT_MODEL env var
    region::String = ""                      # resolved via chain above
    endpoint::String = ""                    # custom endpoint URL (optional)
    credentials::Union{Nothing, BedrockCredentials} = nothing
    access_key_id::String = ""
    secret_access_key::String = ""
    session_token::Union{Nothing, String} = nothing
    profile::String = ""                     # AWS profile name
    default_headers::Dict{String, String} = Dict{String, String}()
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120                  # HTTP read timeout in seconds
end
```

### Capabilities

`BedrockChatClient` declares:
- `HasStreaming()` — streaming responses are supported (emulated via single response)
- `HasToolCalling()` — native tool/function calling via the Converse API

### Model Configuration

The model can be set in three ways:

```julia
# 1. Constructor parameter
client = BedrockChatClient(model = "anthropic.claude-3-haiku-20240307-v1:0")

# 2. Environment variable
ENV["BEDROCK_CHAT_MODEL"] = "anthropic.claude-3-haiku-20240307-v1:0"
client = BedrockChatClient()

# 3. Per-request via ChatOptions
response = get_response(client, messages, ChatOptions(model = "meta.llama3-8b-instruct-v1:0"))
```

### Custom Endpoint

For VPC endpoints or local testing:

```julia
client = BedrockChatClient(
    model = "anthropic.claude-3-haiku-20240307-v1:0",
    endpoint = "https://vpce-xxx.bedrock-runtime.us-east-1.vpce.amazonaws.com",
)
```

### Supported Models

Any model available through the [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html), including:

| Provider | Example Model IDs |
|----------|-------------------|
| Anthropic | `anthropic.claude-3-haiku-20240307-v1:0`, `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| Amazon | `amazon.titan-text-express-v1`, `amazon.titan-text-premier-v1:0` |
| Meta | `meta.llama3-8b-instruct-v1:0`, `meta.llama3-70b-instruct-v1:0` |
| Mistral | `mistral.mistral-7b-instruct-v0:2`, `mistral.mixtral-8x7b-instruct-v0:1` |
| Cohere | `cohere.command-r-plus-v1:0` |

### Tool Calling

Tool calling is supported natively through the Bedrock Converse API:

```julia
using AgentFramework
using AgentFramework.Bedrock

@tool function get_weather(location::String)
    "It's sunny in $location"
end

agent = Agent(
    name = "weather-agent",
    instructions = "Help users with weather queries.",
    chat_client = BedrockChatClient(model = "anthropic.claude-3-haiku-20240307-v1:0"),
    tools = [get_weather],
)

response = run_agent(agent, "What's the weather in Seattle?")
```

### ChatOptions

Standard [`ChatOptions`](@ref) are mapped to Bedrock parameters:

| ChatOptions field | Bedrock parameter |
|-------------------|-------------------|
| `model` | `modelId` |
| `temperature` | `inferenceConfig.temperature` |
| `top_p` | `inferenceConfig.topP` |
| `max_tokens` | `inferenceConfig.maxTokens` (default: 1024) |
| `stop` | `inferenceConfig.stopSequences` |
| `tool_choice` | `toolConfig.toolChoice` (`"auto"`, `"required"`, `"none"`, or tool name) |
| `additional` | Merged into the top-level request body |

## BedrockEmbeddingClient

Generates text embeddings using Bedrock's Invoke Model API.

```julia
Base.@kwdef mutable struct BedrockEmbeddingClient
    model::String = ""              # or set BEDROCK_EMBEDDING_MODEL env var
    region::String = ""
    endpoint::String = ""
    credentials::Union{Nothing, BedrockCredentials} = nothing
    access_key_id::String = ""
    secret_access_key::String = ""
    session_token::Union{Nothing, String} = nothing
    profile::String = ""
    default_headers::Dict{String, String} = Dict{String, String}()
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120
end
```

### `get_embeddings`

```julia
client = BedrockEmbeddingClient(model = "amazon.titan-embed-text-v2:0")

vectors = get_embeddings(client, ["Hello world", "How are you?"])
# => Vector{Vector{Float64}} with 2 elements
```

Texts are embedded one at a time (Bedrock does not support batched embedding requests).

**Keyword arguments:**
- `model` — override the client's default model for this request

## BedrockCredentials

Explicit AWS credential container:

```julia
struct BedrockCredentials
    access_key_id::String
    secret_access_key::String
    session_token::Union{Nothing, String}
end
```

## Error Handling

Bedrock errors are raised as standard AgentFramework chat client exceptions:

- `ChatClientError` — general API errors (HTTP status, response body)
- `ChatClientInvalidAuthError` — missing or invalid AWS credentials
- `ChatClientInvalidRequestError` — invalid request configuration (missing model, empty messages, etc.)
- `ChatClientInvalidResponseError` — unexpected response format

```julia
try
    response = run_agent(agent, "Hello")
catch e::ChatClientInvalidAuthError
    println("AWS credentials not configured: ", e.message)
catch e::ChatClientError
    println("Bedrock API error: ", e.message)
end
```
