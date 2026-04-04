# OpenAI-compatible chat client for AgentFramework.jl
# Works with any OpenAI-compatible API: OpenAI, Azure OpenAI, vLLM, LM Studio, etc.
# Reuses _messages_to_openai, _tools_to_openai, _parse_openai_response, _parse_openai_stream_chunk
# from ollama.jl (module-level functions).
# Uses curl subprocess for streaming (HTTP.jl's HTTP.open has connection pool issues).

const DEFAULT_AZURE_OPENAI_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"

# ── OpenAI Client ────────────────────────────────────────────────────────────

"""
    OpenAIChatClient <: AbstractChatClient

Chat client for the standard OpenAI API and any OpenAI-compatible endpoint
(vLLM, LM Studio, Together AI, etc.).

# Fields
- `model::String`: Model name (default: "gpt-4o").
- `api_key::String`: API key. Falls back to `ENV["OPENAI_API_KEY"]` if empty.
- `base_url::String`: API base URL (default: "https://api.openai.com/v1").
- `organization::Union{Nothing, String}`: Optional OpenAI organization ID.
- `options::Dict{String, Any}`: Default provider-specific options.
- `read_timeout::Int`: Read timeout in seconds (default: 120).

# Examples
```julia
client = OpenAIChatClient(model="gpt-4o")
response = get_response(client, [Message(:user, "Hello!")], ChatOptions())
println(response.text)
```
"""
Base.@kwdef mutable struct OpenAIChatClient <: AbstractChatClient
    model::String = "gpt-4o"
    api_key::String = ""
    base_url::String = "https://api.openai.com/v1"
    organization::Union{Nothing, String} = nothing
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120
end

function Base.show(io::IO, c::OpenAIChatClient)
    print(io, "OpenAIChatClient(\"", c.model, "\")")
end

# ── Azure OpenAI Client ──────────────────────────────────────────────────────

"""
    AzureOpenAIChatClient <: AbstractChatClient

Chat client for Azure OpenAI Service.

# Fields
- `model::String`: Azure deployment name.
- `endpoint::String`: Azure resource endpoint (e.g., "https://myresource.openai.azure.com").
- `api_key::String`: API key. Falls back to `ENV["AZURE_OPENAI_API_KEY"]` if empty.
- `credential`: Optional Azure credential resolved through the AzureIdentity.jl extension.
- `token_provider`: Optional callable returning a bearer token string.
- `token_scope::String`: Token scope for credential-based auth (default: Azure OpenAI scope).
- `api_version::String`: Azure API version (default: "2024-06-01").
- `options::Dict{String, Any}`: Default provider-specific options.
- `read_timeout::Int`: Read timeout in seconds (default: 120).

# Examples
```julia
client = AzureOpenAIChatClient(
    model="gpt-4o",
    endpoint="https://myresource.openai.azure.com",
)
response = get_response(client, [Message(:user, "Hello!")], ChatOptions())
println(response.text)

using AzureIdentity
client = AzureOpenAIChatClient(
    model="gpt-4o",
    endpoint="https://myresource.openai.azure.com",
    credential=DefaultAzureCredential(),
)
```
"""
Base.@kwdef mutable struct AzureOpenAIChatClient <: AbstractChatClient
    model::String
    endpoint::String
    api_key::String = ""
    credential::Any = nothing
    token_provider::Union{Nothing, Function} = nothing
    token_scope::String = get(
        ENV,
        "AZURE_OPENAI_TOKEN_SCOPE",
        get(ENV, "AZURE_OPENAI_TOKEN_ENDPOINT", DEFAULT_AZURE_OPENAI_TOKEN_SCOPE),
    )
    api_version::String = "2024-06-01"
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120
end

function Base.show(io::IO, c::AzureOpenAIChatClient)
    print(io, "AzureOpenAIChatClient(\"", c.model, "\")")
end

# ── API Key Resolution ───────────────────────────────────────────────────────

function _resolve_api_key(client::OpenAIChatClient)::String
    key = client.api_key
    if isempty(key)
        key = get(ENV, "OPENAI_API_KEY", "")
    end
    if isempty(key)
        throw(ChatClientError("OpenAI API key not set. Provide api_key or set OPENAI_API_KEY."))
    end
    return key
end

function _resolve_api_key(client::AzureOpenAIChatClient)::String
    key = client.api_key
    if isempty(key)
        key = get(ENV, "AZURE_OPENAI_API_KEY", "")
    end
    if isempty(key)
        throw(ChatClientError("Azure OpenAI API key not set. Provide api_key or set AZURE_OPENAI_API_KEY."))
    end
    return key
end

function _credential_to_token_provider(credential, token_scope::String)
    throw(
        ChatClientInvalidAuthError(
            "Credential-based Azure OpenAI auth requires AzureIdentity.jl to be loaded.",
        ),
    )
end

function _resolve_token_provider(client::AzureOpenAIChatClient)
    client.token_provider !== nothing && return client.token_provider
    client.credential === nothing && return nothing
    return _credential_to_token_provider(client.credential, client.token_scope)
end

function _resolve_bearer_token(client::AzureOpenAIChatClient)::Union{Nothing, String}
    provider = _resolve_token_provider(client)
    provider === nothing && return nothing
    token = provider()
    token isa AbstractString || throw(
        ChatClientInvalidAuthError("Azure OpenAI token provider must return a token string."),
    )
    return String(token)
end

# ── URL Construction ─────────────────────────────────────────────────────────

function _chat_completions_url(client::OpenAIChatClient)::String
    base = rstrip(client.base_url, '/')
    return base * "/chat/completions"
end

function _chat_completions_url(client::AzureOpenAIChatClient)::String
    base = rstrip(client.endpoint, '/')
    return base * "/openai/deployments/" * client.model * "/chat/completions?api-version=" * client.api_version
end

# ── Header Construction ──────────────────────────────────────────────────────

function _build_headers(client::OpenAIChatClient)::Vector{Pair{String, String}}
    key = _resolve_api_key(client)
    headers = Pair{String, String}[
        "Content-Type" => "application/json",
        "Authorization" => "Bearer " * key,
        "Connection" => "close",
    ]
    if client.organization !== nothing
        push!(headers, "OpenAI-Organization" => client.organization)
    end
    return headers
end

function _build_headers(client::AzureOpenAIChatClient)::Vector{Pair{String, String}}
    headers = Pair{String, String}[
        "Content-Type" => "application/json",
        "Connection" => "close",
    ]
    token = _resolve_bearer_token(client)
    if token !== nothing
        push!(headers, "Authorization" => "Bearer $token")
    else
        push!(headers, "api-key" => _resolve_api_key(client))
    end
    return headers
end

# ── Request Body Construction ────────────────────────────────────────────────

function _build_request_body(client::Union{OpenAIChatClient, AzureOpenAIChatClient},
                             messages::Vector{Message}, options::ChatOptions; stream::Bool=false)
    model = options.model !== nothing ? options.model : client.model
    all_tools = options.tools

    body = Dict{String, Any}(
        "model" => model,
        "messages" => _messages_to_openai(messages, all_tools),
        "stream" => stream,
    )

    tools_json = _tools_to_openai(all_tools)
    if tools_json !== nothing
        body["tools"] = tools_json
    end
    if options.temperature !== nothing
        body["temperature"] = options.temperature
    end
    if options.top_p !== nothing
        body["top_p"] = options.top_p
    end
    if options.max_tokens !== nothing
        body["max_tokens"] = options.max_tokens
    end
    if options.stop !== nothing
        body["stop"] = options.stop
    end
    if options.tool_choice !== nothing
        body["tool_choice"] = options.tool_choice
    end
    if options.response_format !== nothing
        body["response_format"] = options.response_format
    end

    for (k, v) in client.options
        body[k] = v
    end

    return body
end

# ── Non-Streaming Response ───────────────────────────────────────────────────

function get_response(client::OpenAIChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    body = _build_request_body(client, messages, options; stream=false)
    url = _chat_completions_url(client)
    headers = _build_headers(client)
    json_body = JSON3.write(body)

    resp = HTTP.post(url, headers, json_body;
        status_exception = false,
        readtimeout = client.read_timeout,
        connect_timeout = 10,
        retry = false,
    )

    if resp.status != 200
        throw(ChatClientError("OpenAI API error ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    return _parse_openai_response(data)
end

function get_response(client::AzureOpenAIChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    body = _build_request_body(client, messages, options; stream=false)
    url = _chat_completions_url(client)
    headers = _build_headers(client)
    json_body = JSON3.write(body)

    resp = HTTP.post(url, headers, json_body;
        status_exception = false,
        readtimeout = client.read_timeout,
        connect_timeout = 10,
        retry = false,
    )

    if resp.status != 200
        throw(ChatClientError("Azure OpenAI API error ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    return _parse_openai_response(data)
end

# ── Streaming Response ───────────────────────────────────────────────────────
# Uses curl subprocess for reliable SSE streaming (HTTP.jl's HTTP.open has
# connection pool issues that cause hangs with long-running connections).

function _build_curl_headers(client::OpenAIChatClient)::Vector{String}
    key = _resolve_api_key(client)
    headers = ["-H", "Content-Type: application/json",
               "-H", "Authorization: Bearer $key"]
    if client.organization !== nothing
        push!(headers, "-H", "OpenAI-Organization: $(client.organization)")
    end
    return headers
end

function _build_curl_headers(client::AzureOpenAIChatClient)::Vector{String}
    token = _resolve_bearer_token(client)
    if token !== nothing
        return ["-H", "Content-Type: application/json",
                "-H", "Authorization: Bearer $token"]
    end
    key = _resolve_api_key(client)
    return ["-H", "Content-Type: application/json",
            "-H", "api-key: $key"]
end

function get_response_streaming(client::OpenAIChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    _openai_streaming(client, messages, options)
end

function get_response_streaming(client::AzureOpenAIChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    _openai_streaming(client, messages, options)
end

function _openai_streaming(client::Union{OpenAIChatClient, AzureOpenAIChatClient},
                           messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    body = _build_request_body(client, messages, options; stream=true)
    url = _chat_completions_url(client)
    json_body = JSON3.write(body)
    curl_headers = _build_curl_headers(client)

    channel = Channel{ChatResponseUpdate}(32)

    Threads.@spawn begin
        proc = nothing
        try
            cmd = `curl -sN --max-time $(client.read_timeout) $curl_headers -d $json_body $url`
            proc = open(cmd, "r")

            for line in eachline(proc)
                line = strip(line)
                isempty(line) && continue
                startswith(line, "data: ") || continue
                payload = line[7:end]
                payload == "[DONE]" && break

                try
                    chunk = JSON3.read(payload, Dict{String, Any})
                    update = _parse_openai_stream_chunk(chunk)
                    if update !== nothing
                        put!(channel, update)
                    end
                catch e
                    @warn "Failed to parse streaming chunk" exception=e
                end
            end
        catch e
            if !(e isa InvalidStateException)
                @error "OpenAI streaming error" exception=(e, catch_backtrace())
            end
        finally
            if proc !== nothing
                try; close(proc); catch; end
            end
            close(channel)
        end
    end

    return channel
end

# ── Capability Traits ────────────────────────────────────────────────────────

AgentFramework.streaming_capability(::Type{OpenAIChatClient}) = HasStreaming()
AgentFramework.tool_calling_capability(::Type{OpenAIChatClient}) = HasToolCalling()
AgentFramework.structured_output_capability(::Type{OpenAIChatClient}) = HasStructuredOutput()
AgentFramework.embedding_capability(::Type{OpenAIChatClient}) = HasEmbeddings()
AgentFramework.image_generation_capability(::Type{OpenAIChatClient}) = HasImageGeneration()

AgentFramework.streaming_capability(::Type{AzureOpenAIChatClient}) = HasStreaming()
AgentFramework.tool_calling_capability(::Type{AzureOpenAIChatClient}) = HasToolCalling()
AgentFramework.structured_output_capability(::Type{AzureOpenAIChatClient}) = HasStructuredOutput()
AgentFramework.embedding_capability(::Type{AzureOpenAIChatClient}) = HasEmbeddings()
AgentFramework.image_generation_capability(::Type{AzureOpenAIChatClient}) = HasImageGeneration()

# ── Embeddings ───────────────────────────────────────────────────────────────

"""
    get_embeddings(client::OpenAIChatClient, texts::Vector{String}; model=nothing) -> Vector{Vector{Float64}}

Get embeddings from the OpenAI embeddings API.
"""
function get_embeddings(client::OpenAIChatClient, texts::Vector{String}; model::Union{Nothing, String} = nothing)::Vector{Vector{Float64}}
    api_key = !isempty(client.api_key) ? client.api_key : get(ENV, "OPENAI_API_KEY", "")
    isempty(api_key) && throw(ChatClientInvalidAuthError("No OpenAI API key"))

    embed_model = something(model, "text-embedding-3-small")
    body = Dict{String, Any}("model" => embed_model, "input" => texts)

    headers = ["Content-Type" => "application/json", "Authorization" => "Bearer $api_key"]
    if client.organization !== nothing
        push!(headers, "OpenAI-Organization" => client.organization)
    end

    resp = HTTP.post("$(client.base_url)/embeddings", headers, JSON3.write(body);
        status_exception=false, readtimeout=client.read_timeout)

    resp.status != 200 && throw(ChatClientError("OpenAI embeddings error: $(resp.status) — $(String(resp.body))"))

    result = JSON3.read(String(resp.body), Dict{String, Any})
    data = result["data"]
    sort!(data, by=d -> d["index"])
    return [Vector{Float64}(d["embedding"]) for d in data]
end

"""
    get_embeddings(client::AzureOpenAIChatClient, texts::Vector{String}; model=nothing) -> Vector{Vector{Float64}}

Get embeddings from the Azure OpenAI embeddings API.
"""
function get_embeddings(client::AzureOpenAIChatClient, texts::Vector{String}; model::Union{Nothing, String} = nothing)::Vector{Vector{Float64}}
    deploy = something(model, client.model)
    url = "$(client.endpoint)/openai/deployments/$deploy/embeddings?api-version=$(client.api_version)"
    body = Dict{String, Any}("input" => texts)
    headers = _build_headers(client)

    resp = HTTP.post(url, headers, JSON3.write(body); status_exception=false, readtimeout=client.read_timeout)
    resp.status != 200 && throw(ChatClientError("Azure embeddings error: $(resp.status) — $(String(resp.body))"))

    result = JSON3.read(String(resp.body), Dict{String, Any})
    data = result["data"]
    sort!(data, by=d -> d["index"])
    return [Vector{Float64}(d["embedding"]) for d in data]
end
