# AzureIdentity is an optional dependency; loaded via the extension when available
const _HAS_AZURE_IDENTITY = Ref(false)

function _check_azure_identity_credential(credential, label::String)
    # This is overridden by the AzureIdentityExt when AzureIdentity is loaded
    throw(ChatClientInvalidAuthError(
        "$label requires AzureIdentity.jl. Run `using AzureIdentity` first, or provide a token_provider function directly.",
    ))
end

function _get_azure_bearer_token_provider(credential, scope::String)
    # This is overridden by the AzureIdentityExt when AzureIdentity is loaded
    throw(ChatClientInvalidAuthError(
        "AzureIdentity.jl is not loaded. Run `using AzureIdentity` first.",
    ))
end

const DEFAULT_FOUNDRY_PROJECT_TOKEN_SCOPE = "https://ai.azure.com/.default"
const DEFAULT_FOUNDRY_MODELS_TOKEN_SCOPE = "https://ml.azure.com/.default"
const DEFAULT_FOUNDRY_MODELS_API_VERSION = "2024-05-01-preview"

"""
    FoundryChatClient <: AbstractChatClient

Chat client for Microsoft Foundry project endpoints using the OpenAI-compatible
data plane surfaced by Azure AI Projects.

# Fields
- `model::String`: Model deployment name. Falls back to `ENV["FOUNDRY_MODEL"]`.
- `project_endpoint::String`: Foundry project endpoint. Falls back to
  `ENV["FOUNDRY_PROJECT_ENDPOINT"]`.
- `credential`: Optional `AzureIdentity.AbstractAzureCredential`.
- `token_provider`: Optional callable returning a bearer token.
- `token_scope::String`: Token scope for credential-based auth.
- `default_headers::Dict{String, String}`: Additional request headers.
- `options::Dict{String, Any}`: Default provider-specific options.
- `read_timeout::Int`: Read timeout in seconds.
"""
Base.@kwdef mutable struct FoundryChatClient <: AbstractChatClient
    model::String = ""
    project_endpoint::String = ""
    credential::Any = nothing
    token_provider::Union{Nothing, Function} = nothing
    token_scope::String = get(ENV, "FOUNDRY_PROJECT_TOKEN_SCOPE", DEFAULT_FOUNDRY_PROJECT_TOKEN_SCOPE)
    default_headers::Dict{String, String} = Dict{String, String}()
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120
end

function Base.show(io::IO, c::FoundryChatClient)
    model = isempty(c.model) ? get(ENV, "FOUNDRY_MODEL", "") : c.model
    print(io, "FoundryChatClient(\"", model, "\")")
end

"""
    FoundryEmbeddingClient

Embedding client for Microsoft Foundry model inference endpoints.

# Fields
- `model::String`: Embedding model name. Falls back to `ENV["FOUNDRY_EMBEDDING_MODEL"]`.
- `endpoint::String`: Model inference endpoint. Falls back to `ENV["FOUNDRY_MODELS_ENDPOINT"]`.
- `api_key::String`: Optional API key. Falls back to `ENV["FOUNDRY_MODELS_API_KEY"]`.
- `credential`: Optional `AzureIdentity.AbstractAzureCredential`.
- `token_provider`: Optional callable returning a bearer token.
- `token_scope::String`: Token scope for credential-based auth.
- `api_version::String`: Model inference API version.
- `default_headers::Dict{String, String}`: Additional request headers.
- `options::Dict{String, Any}`: Default embedding request options.
- `read_timeout::Int`: Read timeout in seconds.
"""
Base.@kwdef mutable struct FoundryEmbeddingClient
    model::String = ""
    endpoint::String = ""
    api_key::String = ""
    credential::Any = nothing
    token_provider::Union{Nothing, Function} = nothing
    token_scope::String = get(ENV, "FOUNDRY_MODELS_TOKEN_SCOPE", DEFAULT_FOUNDRY_MODELS_TOKEN_SCOPE)
    api_version::String = get(ENV, "FOUNDRY_MODELS_API_VERSION", DEFAULT_FOUNDRY_MODELS_API_VERSION)
    default_headers::Dict{String, String} = Dict{String, String}()
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120
end

function Base.show(io::IO, c::FoundryEmbeddingClient)
    model = isempty(c.model) ? get(ENV, "FOUNDRY_EMBEDDING_MODEL", "") : c.model
    print(io, "FoundryEmbeddingClient(\"", model, "\")")
end

function _resolve_foundry_model(client::FoundryChatClient)::String
    model = isempty(client.model) ? get(ENV, "FOUNDRY_MODEL", "") : client.model
    isempty(model) && throw(
        ChatClientInvalidRequestError(
            "Foundry model not set. Provide model or set FOUNDRY_MODEL.",
        ),
    )
    return model
end

function _resolve_foundry_project_endpoint(client::FoundryChatClient)::String
    endpoint = isempty(client.project_endpoint) ? get(ENV, "FOUNDRY_PROJECT_ENDPOINT", "") : client.project_endpoint
    endpoint = String(strip(endpoint))
    isempty(endpoint) && throw(
        ChatClientInvalidRequestError(
            "Foundry project endpoint not set. Provide project_endpoint or set FOUNDRY_PROJECT_ENDPOINT.",
        ),
    )
    return endpoint
end

function _normalize_foundry_project_base_url(project_endpoint::String)::String
    endpoint = rstrip(String(strip(project_endpoint)), '/')
    endswith(endpoint, "/openai/v1") && return endpoint
    return endpoint * "/openai/v1"
end

function _resolve_foundry_embedding_model(client::FoundryEmbeddingClient)::String
    model = isempty(client.model) ? get(ENV, "FOUNDRY_EMBEDDING_MODEL", "") : client.model
    isempty(model) && throw(
        ChatClientInvalidRequestError(
            "Foundry embedding model not set. Provide model or set FOUNDRY_EMBEDDING_MODEL.",
        ),
    )
    return model
end

function _resolve_foundry_models_endpoint(client::FoundryEmbeddingClient)::String
    endpoint = isempty(client.endpoint) ? get(ENV, "FOUNDRY_MODELS_ENDPOINT", "") : client.endpoint
    endpoint = String(strip(endpoint))
    isempty(endpoint) && throw(
        ChatClientInvalidRequestError(
            "Foundry models endpoint not set. Provide endpoint or set FOUNDRY_MODELS_ENDPOINT.",
        ),
    )
    return endpoint
end

function _resolve_foundry_models_api_key(client::FoundryEmbeddingClient)::String
    key = isempty(client.api_key) ? get(ENV, "FOUNDRY_MODELS_API_KEY", "") : client.api_key
    isempty(key) && throw(
        ChatClientInvalidAuthError(
            "Foundry models API key not set. Provide api_key, credential, token_provider, or set FOUNDRY_MODELS_API_KEY.",
        ),
    )
    return key
end

function _resolve_foundry_token_provider(
    credential,
    token_provider::Union{Nothing, Function},
    token_scope::String,
    env_name::String,
    label::String,
)::Union{Nothing, Function}
    token_provider !== nothing && return token_provider
    credential === nothing && return nothing

    scope = String(strip(token_scope))
    isempty(scope) && throw(
        ChatClientInvalidAuthError(
            "$label token scope not set. Provide token_scope or set $env_name.",
        ),
    )

    if !_HAS_AZURE_IDENTITY[]
        throw(ChatClientInvalidAuthError(
            "$label requires AzureIdentity.jl. Run `using AzureIdentity` first, or provide a token_provider function directly.",
        ))
    end

    _check_azure_identity_credential(credential, label)

    return _get_azure_bearer_token_provider(credential, scope)
end

function _resolve_foundry_bearer_token(
    credential,
    token_provider::Union{Nothing, Function},
    token_scope::String,
    env_name::String,
    label::String,
)::Union{Nothing, String}
    provider = _resolve_foundry_token_provider(credential, token_provider, token_scope, env_name, label)
    provider === nothing && return nothing

    token = provider()
    token isa AbstractString || throw(
        ChatClientInvalidAuthError("$label token provider must return a token string."),
    )
    return String(token)
end

function _append_default_headers!(
    headers::Vector{Pair{String, String}},
    default_headers::Dict{String, String},
)
    for (key, value) in default_headers
        push!(headers, key => value)
    end
    return headers
end

function _append_default_curl_headers!(
    headers::Vector{String},
    default_headers::Dict{String, String},
)
    for (key, value) in default_headers
        push!(headers, "-H", "$key: $value")
    end
    return headers
end

function _chat_completions_url(client::FoundryChatClient)::String
    base = _normalize_foundry_project_base_url(_resolve_foundry_project_endpoint(client))
    return base * "/chat/completions"
end

function _build_headers(client::FoundryChatClient)::Vector{Pair{String, String}}
    token = _resolve_foundry_bearer_token(
        client.credential,
        client.token_provider,
        client.token_scope,
        "FOUNDRY_PROJECT_TOKEN_SCOPE",
        "Foundry project auth",
    )
    token === nothing && throw(
        ChatClientInvalidAuthError(
            "Foundry project auth requires credential or token_provider.",
        ),
    )

    headers = Pair{String, String}[
        "Content-Type" => "application/json",
        "Authorization" => "Bearer $token",
        "Connection" => "close",
    ]
    return _append_default_headers!(headers, client.default_headers)
end

function _build_curl_headers(client::FoundryChatClient)::Vector{String}
    token = _resolve_foundry_bearer_token(
        client.credential,
        client.token_provider,
        client.token_scope,
        "FOUNDRY_PROJECT_TOKEN_SCOPE",
        "Foundry project auth",
    )
    token === nothing && throw(
        ChatClientInvalidAuthError(
            "Foundry project auth requires credential or token_provider.",
        ),
    )

    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Authorization: Bearer $token",
    ]
    return _append_default_curl_headers!(headers, client.default_headers)
end

function _build_request_body(
    client::FoundryChatClient,
    messages::Vector{Message},
    options::ChatOptions;
    stream::Bool = false,
)
    model = options.model !== nothing ? options.model : _resolve_foundry_model(client)
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

    for (key, value) in client.options
        body[key] = value
    end

    return body
end

function get_response(
    client::FoundryChatClient,
    messages::Vector{Message},
    options::ChatOptions,
)::ChatResponse
    body = _build_request_body(client, messages, options; stream = false)
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
        throw(ChatClientError("Foundry API error ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    return _parse_openai_response(data)
end

function get_response_streaming(
    client::FoundryChatClient,
    messages::Vector{Message},
    options::ChatOptions,
)::Channel{ChatResponseUpdate}
    body = _build_request_body(client, messages, options; stream = true)
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
                catch exc
                    @warn "Failed to parse Foundry streaming chunk" exception = exc
                end
            end
        catch exc
            if !(exc isa InvalidStateException)
                @error "Foundry streaming error" exception = (exc, catch_backtrace())
            end
        finally
            if proc !== nothing
                try
                    close(proc)
                catch
                end
            end
            close(channel)
        end
    end

    return channel
end

function _embeddings_url(client::FoundryEmbeddingClient)::String
    endpoint = rstrip(_resolve_foundry_models_endpoint(client), '/')
    if occursin("/embeddings?", endpoint) || endswith(endpoint, "/embeddings")
        separator = occursin("?", endpoint) ? "&" : "?"
        return endpoint * separator * "api-version=$(client.api_version)"
    end
    return endpoint * "/embeddings?api-version=$(client.api_version)"
end

function _build_headers(client::FoundryEmbeddingClient)::Vector{Pair{String, String}}
    headers = Pair{String, String}[
        "Content-Type" => "application/json",
        "Connection" => "close",
    ]

    token = _resolve_foundry_bearer_token(
        client.credential,
        client.token_provider,
        client.token_scope,
        "FOUNDRY_MODELS_TOKEN_SCOPE",
        "Foundry models auth",
    )
    if token !== nothing
        push!(headers, "Authorization" => "Bearer $token")
    else
        push!(headers, "api-key" => _resolve_foundry_models_api_key(client))
    end

    return _append_default_headers!(headers, client.default_headers)
end

"""
    get_embeddings(client::FoundryEmbeddingClient, texts::Vector{String}; model=nothing)

Get embeddings from a Microsoft Foundry model inference endpoint.
"""
function get_embeddings(
    client::FoundryEmbeddingClient,
    texts::Vector{String};
    model::Union{Nothing, String} = nothing,
)::Vector{Vector{Float64}}
    isempty(texts) && return Vector{Float64}[]

    body = Dict{String, Any}(
        "input" => texts,
        "model" => something(model, _resolve_foundry_embedding_model(client)),
    )
    for (key, value) in client.options
        body[key] = value
    end

    resp = HTTP.post(
        _embeddings_url(client),
        _build_headers(client),
        JSON3.write(body);
        status_exception = false,
        readtimeout = client.read_timeout,
        connect_timeout = 10,
        retry = false,
    )

    if resp.status != 200
        throw(ChatClientError("Foundry embeddings error ($(resp.status)): $(String(resp.body))"))
    end

    result = JSON3.read(String(resp.body), Dict{String, Any})
    data = get(result, "data", Any[])
    sort!(data, by = item -> get(item, "index", 0))
    return [Vector{Float64}(item["embedding"]) for item in data]
end

AgentFramework.streaming_capability(::Type{FoundryChatClient}) = HasStreaming()
AgentFramework.tool_calling_capability(::Type{FoundryChatClient}) = HasToolCalling()
AgentFramework.structured_output_capability(::Type{FoundryChatClient}) = HasStructuredOutput()
