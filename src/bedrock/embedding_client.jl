Base.@kwdef mutable struct BedrockEmbeddingClient
    model::String = ""
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

function Base.show(io::IO, client::BedrockEmbeddingClient)
    model = isempty(client.model) ? get(ENV, "BEDROCK_EMBEDDING_MODEL", "") : client.model
    print(io, "BedrockEmbeddingClient(\"", model, "\")")
end

function _resolve_embedding_model(client::BedrockEmbeddingClient)::String
    model = isempty(client.model) ? get(ENV, "BEDROCK_EMBEDDING_MODEL", "") : client.model
    model = String(strip(model))
    isempty(model) && throw(
        ChatClientInvalidRequestError(
            "Bedrock embedding model not set. Provide model or set BEDROCK_EMBEDDING_MODEL.",
        ),
    )
    return model
end

function _resolve_bedrock_endpoint(client::BedrockEmbeddingClient)::String
    endpoint = _nonempty_string(client.endpoint)
    endpoint !== nothing && return rstrip(endpoint, '/')
    return "https://bedrock-runtime.$(_resolve_bedrock_region(client)).amazonaws.com"
end

function _bedrock_invoke_url(client::BedrockEmbeddingClient, model::AbstractString)::String
    return _resolve_bedrock_endpoint(client) * "/model/" * _aws_percent_encode(model) * "/invoke"
end

function _build_embedding_request(
    client::BedrockEmbeddingClient,
    text::AbstractString,
    model::AbstractString,
)::Dict{String, Any}
    body = Dict{String, Any}("inputText" => String(text))
    for (key, value) in client.options
        body[key] = value
    end
    body["modelId"] = model
    return body
end

function get_embeddings(
    client::BedrockEmbeddingClient,
    texts::Vector{String};
    model::Union{Nothing, String} = nothing,
)::Vector{Vector{Float64}}
    isempty(texts) && return Vector{Vector{Float64}}()

    embed_model = something(model, _resolve_embedding_model(client))
    url = _bedrock_invoke_url(client, embed_model)
    embeddings = Vector{Vector{Float64}}()

    for text in texts
        body = _build_embedding_request(client, text, embed_model)
        delete!(body, "modelId")
        response = _post_json(client, url, body; error_label = "Bedrock embeddings")
        vector = get(response, "embedding", nothing)
        vector isa AbstractVector || throw(
            ChatClientInvalidResponseError("Bedrock embeddings response was missing an embedding vector."),
        )
        push!(embeddings, [Float64(value) for value in vector])
    end

    return embeddings
end
