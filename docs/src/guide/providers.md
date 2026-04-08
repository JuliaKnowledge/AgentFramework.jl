# Providers

Providers are the LLM backends that agents communicate with. Each provider implements the [`AbstractChatClient`](@ref) interface, translating AgentFramework.jl's message format to and from the provider's API.

## Provider Architecture

All providers subtype [`AbstractChatClient`](@ref) and implement two methods:

```julia
get_response(client::AbstractChatClient, messages::Vector{Message}, options::ChatOptions) -> ChatResponse
get_response_streaming(client::AbstractChatClient, messages::Vector{Message}, options::ChatOptions) -> Channel{ChatResponseUpdate}
```

This abstraction means agents are provider-agnostic — you can swap providers without changing agent code.

## Built-in Providers

### OllamaChatClient

[`OllamaChatClient`](@ref) connects to a local [Ollama](https://ollama.com) instance via its OpenAI-compatible API. This is the easiest way to get started — no API keys needed.

```julia
using AgentFramework

client = OllamaChatClient(
    model = "qwen3:8b",                           # Required: Ollama model name
    base_url = "http://localhost:11434",            # Default Ollama URL
    options = Dict{String, Any}("num_ctx" => 8192), # Ollama-specific options
    read_timeout = 300,                             # Timeout in seconds
)
```

**Prerequisites:** Install Ollama and pull a model:
```bash
ollama pull qwen3:8b
```

### OpenAIChatClient

[`OpenAIChatClient`](@ref) works with the standard OpenAI API and any OpenAI-compatible endpoint (vLLM, LM Studio, Together AI, etc.):

```julia
client = OpenAIChatClient(
    model = "gpt-4o",                              # Model name
    api_key = "",                                   # Falls back to ENV["OPENAI_API_KEY"]
    base_url = "https://api.openai.com/v1",        # Override for compatible APIs
    organization = nothing,                         # Optional org ID
    read_timeout = 120,
)
```

**Using with OpenAI-compatible services:**

```julia
# vLLM
client = OpenAIChatClient(
    model = "meta-llama/Llama-3-8b",
    base_url = "http://localhost:8000/v1",
    api_key = "not-needed",
)

# LM Studio
client = OpenAIChatClient(
    model = "local-model",
    base_url = "http://localhost:1234/v1",
    api_key = "lm-studio",
)
```

### AzureOpenAIChatClient

[`AzureOpenAIChatClient`](@ref) connects to Azure OpenAI Service:

```julia
client = AzureOpenAIChatClient(
    model = "gpt-4o",                                        # Deployment name
    endpoint = "https://myresource.openai.azure.com",        # Azure endpoint
    api_key = "",                                             # Falls back to ENV["AZURE_OPENAI_API_KEY"]
    api_version = "2024-06-01",                               # API version
    read_timeout = 120,
)
```

**With Azure Identity (Entra ID / managed identity):**

```julia
using AzureIdentity

client = AzureOpenAIChatClient(
    model = "gpt-4o",
    endpoint = "https://myresource.openai.azure.com",
    credential = DefaultAzureCredential(),
)
```

You can also supply a custom `token_provider` function:

```julia
client = AzureOpenAIChatClient(
    model = "gpt-4o",
    endpoint = "https://myresource.openai.azure.com",
    token_provider = () -> get_my_token(),
)
```

### AnthropicChatClient

[`AnthropicChatClient`](@ref) connects to Anthropic's Claude Messages API:

```julia
client = AnthropicChatClient(
    model = "claude-sonnet-4-20250514",            # Model name
    api_key = "",                                   # Falls back to ENV["ANTHROPIC_API_KEY"]
    base_url = "https://api.anthropic.com",
    api_version = "2023-06-01",                     # Anthropic API version header
    read_timeout = 120,
)
```

### FoundryChatClient

[`FoundryChatClient`](@ref) connects to Microsoft Foundry project endpoints:

```julia
using AzureIdentity

client = FoundryChatClient(
    model = "gpt-4o",                              # Deployment name
    project_endpoint = "https://my-project.api.azureml.ms",
    credential = DefaultAzureCredential(),
)
```

Or with environment variables:

```julia
# Set FOUNDRY_MODEL and FOUNDRY_PROJECT_ENDPOINT
client = FoundryChatClient(
    credential = DefaultAzureCredential(),
)
```

## ChatOptions

[`ChatOptions`](@ref) configure individual LLM requests:

```julia
options = ChatOptions(
    model = nothing,              # Override model for this request
    temperature = 0.7,            # Sampling temperature (0.0–2.0)
    top_p = 0.9,                  # Nucleus sampling parameter
    max_tokens = 1000,            # Maximum response tokens
    stop = ["END"],               # Stop sequences
    tools = FunctionTool[],       # Tools for this request (usually set by agent)
    tool_choice = "auto",         # "auto", "none", "required", or a function name
    response_format = nothing,    # Structured output format
    additional = Dict{String, Any}(), # Provider-specific options
)
```

Pass options per-call or set as agent defaults:

```julia
# Per-call override
response = run_agent(agent, "Hello", options=ChatOptions(temperature=0.0))

# Agent defaults
agent = Agent(
    client = client,
    options = ChatOptions(temperature=0.7, max_tokens=2000),
)
```

Options merge with precedence: per-call > agent defaults > provider defaults.

## Capability System

AgentFramework.jl uses the [Holy Traits pattern](https://www.juliabloggers.com/the-emergent-features-of-julialang-part-ii-traits/) for runtime capability detection:

```julia
# Query capabilities
supports_streaming(client)          # true/false
supports_tool_calling(client)       # true/false
supports_structured_output(client)  # true/false
supports_embeddings(client)         # true/false
supports_image_generation(client)   # true/false
supports_code_interpreter(client)   # true/false
supports_file_search(client)        # true/false
supports_web_search(client)         # true/false
```

### Available Capability Traits

| Trait | Description |
|:------|:------------|
| [`HasStreaming`](@ref) | Streaming responses via `get_response_streaming` |
| [`HasToolCalling`](@ref) | Function/tool calling support |
| [`HasStructuredOutput`](@ref) | JSON schema response format |
| [`HasEmbeddings`](@ref) | Text embedding generation |
| [`HasImageGeneration`](@ref) | Image generation |
| [`HasCodeInterpreter`](@ref) | Code execution sandbox |
| [`HasFileSearch`](@ref) | File search across uploaded documents |
| [`HasWebSearch`](@ref) | Web search integration |

### Requiring Capabilities

```julia
require_capability(client, streaming_capability, "Streaming required for this operation")
```

This throws an error if the client doesn't support the capability.

### Listing Capabilities

```julia
caps = list_capabilities(client)
# Vector of Capability instances
```

## Implementing Custom Providers

To create a custom provider, subtype [`AbstractChatClient`](@ref) and implement the required methods:

```julia
struct MyCustomClient <: AbstractChatClient
    api_url::String
    api_key::String
end

function AgentFramework.get_response(
    client::MyCustomClient,
    messages::Vector{Message},
    options::ChatOptions,
)::ChatResponse
    # 1. Convert messages to your API format
    api_messages = convert_messages(messages)

    # 2. Make the API call
    response = HTTP.Response(200, JSON3.write(Dict("text" => "Hello from a custom provider")))

    # 3. Parse the response into a ChatResponse
    body = JSON3.read(response.body)
    return ChatResponse(
        messages = [Message(:assistant, body["text"])],
        finish_reason = STOP,
        model_id = "my-model",
    )
end

function AgentFramework.get_response_streaming(
    client::MyCustomClient,
    messages::Vector{Message},
    options::ChatOptions,
)::Channel{ChatResponseUpdate}
    Channel{ChatResponseUpdate}(32) do ch
        # Stream chunks into the channel
        for chunk in stream_api_call(client, messages)
            put!(ch, ChatResponseUpdate(
                contents = [text_content(chunk.text)],
            ))
        end
        # Final update with finish reason
        put!(ch, ChatResponseUpdate(finish_reason=STOP))
    end
end
```

### Declaring Capabilities

Override the trait functions to declare what your provider supports:

```julia
AgentFramework.streaming_capability(::Type{MyCustomClient}) = HasStreaming()
AgentFramework.tool_calling_capability(::Type{MyCustomClient}) = HasToolCalling()
```

## Provider-Specific Features

### Ollama Options

Pass Ollama-specific options via the `options` field:

```julia
client = OllamaChatClient(
    model = "qwen3:8b",
    options = Dict{String, Any}(
        "num_ctx" => 16384,    # Context window size
        "seed" => 42,          # Reproducible outputs
        "num_gpu" => 1,        # GPU layers
    ),
)
```

### Azure OpenAI Data Sources

Pass Azure-specific features via `ChatOptions.additional`:

```julia
options = ChatOptions(
    additional = Dict{String, Any}(
        "data_sources" => [
            Dict("type" => "azure_search", "parameters" => Dict("endpoint" => "..."))
        ],
    ),
)
```

## Next Steps

- [Streaming](@ref) — Detailed streaming patterns with different providers
- [Agents](@ref) — How agents use providers via the chat client abstraction
- [Advanced Topics](@ref) — FoundryEmbeddingClient and embedding generation
