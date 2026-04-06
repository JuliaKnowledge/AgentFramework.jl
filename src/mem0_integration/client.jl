const MEM0_PLATFORM = :platform
const MEM0_OSS = :oss

const DEFAULT_MEM0_PLATFORM_BASE_URL = "https://api.mem0.ai"
const DEFAULT_MEM0_OSS_BASE_URL = "http://localhost:8000"
const DEFAULT_MEM0_SOURCE_ID = "mem0"
const DEFAULT_MEM0_CONTEXT_PROMPT = "## Memories\nConsider the following memories when answering user questions:"

function _nonempty_string(value)::Union{Nothing, String}
    value === nothing && return nothing
    text = strip(String(value))
    isempty(text) && return nothing
    return text
end

_env_string(name::AbstractString) = _nonempty_string(get(ENV, String(name), nothing))

function _normalize_mem0_deployment(deployment::Symbol)::Symbol
    deployment in (MEM0_PLATFORM, MEM0_OSS) && return deployment
    throw(ArgumentError("Unsupported Mem0 deployment: $(deployment). Use :platform or :oss."))
end

function _default_mem0_base_url(deployment::Symbol)::String
    deployment == MEM0_PLATFORM && return DEFAULT_MEM0_PLATFORM_BASE_URL
    deployment == MEM0_OSS && return DEFAULT_MEM0_OSS_BASE_URL
    throw(ArgumentError("Unsupported Mem0 deployment: $(deployment). Use :platform or :oss."))
end

function _strip_trailing_slashes(text::AbstractString)::String
    return replace(String(text), r"/+$" => "")
end

function _strip_leading_slashes(text::AbstractString)::String
    return replace(String(text), r"^/+" => "")
end

function _default_mem0_request(
    method::AbstractString,
    url::AbstractString;
    headers::AbstractVector{<:Pair} = Pair{String, String}[],
    body = nothing,
)
    payload = body === nothing ? nothing : JSON3.write(body)
    response = HTTP.request(String(method), String(url), collect(headers); body = payload, status_exception = false)
    if response.status < 200 || response.status >= 300
        body_text = isempty(response.body) ? nothing : String(response.body)
        throw(Mem0Error("Mem0 request failed for $(method) $(url).", response.status, body_text))
    end
    return response
end

mutable struct Mem0Client
    api_key::Union{Nothing, String}
    base_url::String
    deployment::Symbol
    request_runner::Function
end

function Mem0Client(;
    api_key::Union{Nothing, AbstractString} = nothing,
    base_url::Union{Nothing, AbstractString} = nothing,
    deployment::Symbol = MEM0_PLATFORM,
    request_runner::Function = _default_mem0_request,
)
    resolved_deployment = _normalize_mem0_deployment(deployment)
    resolved_api_key = _nonempty_string(api_key)
    resolved_api_key === nothing && (resolved_api_key = _env_string("MEM0_API_KEY"))

    if resolved_deployment == MEM0_PLATFORM && resolved_api_key === nothing
        throw(ArgumentError("Mem0 platform clients require `api_key` or MEM0_API_KEY."))
    end

    resolved_base_url = _nonempty_string(base_url)
    resolved_base_url === nothing && (resolved_base_url = _env_string("MEM0_BASE_URL"))
    resolved_base_url === nothing && (resolved_base_url = _default_mem0_base_url(resolved_deployment))

    return Mem0Client(
        resolved_api_key,
        resolved_base_url,
        resolved_deployment,
        request_runner,
    )
end

function Base.show(io::IO, client::Mem0Client)
    print(io, "Mem0Client(\"", client.base_url, "\", deployment=:", client.deployment, ")")
end

function _mem0_headers(client::Mem0Client)::Vector{Pair{String, String}}
    headers = Pair{String, String}[
        "Content-Type" => "application/json",
        "Accept" => "application/json",
    ]

    if client.api_key !== nothing
        if client.deployment == MEM0_PLATFORM
            push!(headers, "Authorization" => "Token " * client.api_key)
        else
            push!(headers, "X-API-Key" => client.api_key)
        end
    end

    return headers
end

function _mem0_url(client::Mem0Client, path::AbstractString)::String
    return string(_strip_trailing_slashes(client.base_url), "/", _strip_leading_slashes(path))
end

function _mem0_request(client::Mem0Client, method::AbstractString, path::AbstractString; body = nothing)
    return client.request_runner(
        String(method),
        _mem0_url(client, path);
        headers = _mem0_headers(client),
        body = body,
    )
end

function _read_mem0_payload(response::HTTP.Response)
    body_text = strip(String(response.body))
    isempty(body_text) && return nothing
    return JSON3.read(body_text)
end
