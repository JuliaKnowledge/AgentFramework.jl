# Chat client capability traits for AgentFramework.jl
# Uses the Holy Traits pattern for runtime capability detection and dispatch.

# ── Capability Trait Types ───────────────────────────────────────────────────

"""Base type for all capability traits."""
abstract type Capability end

"""Indicates a client supports text embedding generation."""
struct HasEmbeddings <: Capability end

"""Indicates a client supports image generation."""
struct HasImageGeneration <: Capability end

"""Indicates a client supports code interpretation/execution."""
struct HasCodeInterpreter <: Capability end

"""Indicates a client supports file search."""
struct HasFileSearch <: Capability end

"""Indicates a client supports web search."""
struct HasWebSearch <: Capability end

"""Indicates a client supports streaming responses."""
struct HasStreaming <: Capability end

"""Indicates a client supports structured output (JSON schema responses)."""
struct HasStructuredOutput <: Capability end

"""Indicates a client supports tool/function calling."""
struct HasToolCalling <: Capability end

"""Sentinel type indicating a capability is not supported."""
struct NoCapability end

# ── Trait Functions (defaults: no capability) ────────────────────────────────

embedding_capability(::Type{<:AbstractChatClient}) = NoCapability()
image_generation_capability(::Type{<:AbstractChatClient}) = NoCapability()
code_interpreter_capability(::Type{<:AbstractChatClient}) = NoCapability()
file_search_capability(::Type{<:AbstractChatClient}) = NoCapability()
web_search_capability(::Type{<:AbstractChatClient}) = NoCapability()
streaming_capability(::Type{<:AbstractChatClient}) = NoCapability()
structured_output_capability(::Type{<:AbstractChatClient}) = NoCapability()
tool_calling_capability(::Type{<:AbstractChatClient}) = NoCapability()

# ── Convenience Query Functions ──────────────────────────────────────────────

"""
    has_capability(client::AbstractChatClient, cap_fn::Function) -> Bool

Check whether `client` has the capability queried by `cap_fn`.
"""
has_capability(client::AbstractChatClient, cap_fn::Function) =
    !(cap_fn(typeof(client)) isa NoCapability)

supports_embeddings(client::AbstractChatClient) = has_capability(client, embedding_capability)
supports_image_generation(client::AbstractChatClient) = has_capability(client, image_generation_capability)
supports_code_interpreter(client::AbstractChatClient) = has_capability(client, code_interpreter_capability)
supports_file_search(client::AbstractChatClient) = has_capability(client, file_search_capability)
supports_web_search(client::AbstractChatClient) = has_capability(client, web_search_capability)
supports_streaming(client::AbstractChatClient) = has_capability(client, streaming_capability)
supports_structured_output(client::AbstractChatClient) = has_capability(client, structured_output_capability)
supports_tool_calling(client::AbstractChatClient) = has_capability(client, tool_calling_capability)

"""
    list_capabilities(client::AbstractChatClient) -> Vector{Symbol}

Return a list of capability symbols supported by `client`.
"""
function list_capabilities(client::AbstractChatClient)::Vector{Symbol}
    caps = Symbol[]
    supports_embeddings(client) && push!(caps, :embeddings)
    supports_image_generation(client) && push!(caps, :image_generation)
    supports_code_interpreter(client) && push!(caps, :code_interpreter)
    supports_file_search(client) && push!(caps, :file_search)
    supports_web_search(client) && push!(caps, :web_search)
    supports_streaming(client) && push!(caps, :streaming)
    supports_structured_output(client) && push!(caps, :structured_output)
    supports_tool_calling(client) && push!(caps, :tool_calling)
    return caps
end

# ── Require Capability ───────────────────────────────────────────────────────

"""
    require_capability(client::AbstractChatClient, cap_fn::Function, operation::String)

Throw a `ChatClientError` if `client` does not support the capability queried by `cap_fn`.
"""
function require_capability(client::AbstractChatClient, cap_fn::Function, operation::String)
    if !has_capability(client, cap_fn)
        throw(ChatClientError("$(typeof(client)) does not support $operation"))
    end
end

# ── Capability-Specific Interfaces ───────────────────────────────────────────

"""
    get_embeddings(client::AbstractChatClient, texts::Vector{String}; model=nothing) -> Vector{Vector{Float64}}

Generate embeddings for the given texts. Only available for clients with `HasEmbeddings`.
"""
function get_embeddings end

"""
    generate_image(client::AbstractChatClient, prompt::String; size=nothing, quality=nothing) -> Any

Generate an image from a text prompt. Only available for clients with `HasImageGeneration`.
"""
function generate_image end
