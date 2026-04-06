function _materialize(value)
    if value isa AbstractDict
        return Dict{String, Any}(string(key) => _materialize(item) for (key, item) in pairs(value))
    elseif value isa AbstractVector
        return Any[_materialize(item) for item in value]
    else
        return value
    end
end

function _dict(value)::Dict{String, Any}
    value === nothing && return Dict{String, Any}()
    value isa Dict{String, Any} && return value
    value isa AbstractDict && return Dict{String, Any}(string(key) => _materialize(item) for (key, item) in pairs(value))
    throw(A2AError("Expected a dictionary-compatible value, got $(typeof(value))"))
end

_maybe_string(value)::Union{Nothing, String} = value === nothing ? nothing : string(value)

function _string_vector(value)::Vector{String}
    value isa AbstractVector || return String[]
    return [string(item) for item in value]
end

function _filter_metadata(metadata)::Dict{String, Any}
    metadata isa AbstractDict || return Dict{String, Any}()
    filtered = Dict{String, Any}()
    for (key, value) in pairs(metadata)
        skey = string(key)
        if skey == "_attribution" || skey == "context_id" || startswith(skey, "__")
            continue
        end
        filtered[skey] = _materialize(value)
    end
    return filtered
end

function _normalize_base_url(url::AbstractString)::String
    normalized = strip(String(url))
    isempty(normalized) && throw(A2AError("A2A base_url cannot be empty"))
    return endswith(normalized, "/") ? normalized[1:end-1] : normalized
end

function _join_url(base::AbstractString, path::AbstractString)::String
    startswith(path, "/") && return string(_normalize_base_url(base), path)
    return string(_normalize_base_url(base), "/", path)
end

function _json_body(response::HTTP.Response)::Dict{String, Any}
    isempty(response.body) && return Dict{String, Any}()
    parsed = JSON3.read(String(response.body))
    return _dict(_materialize(parsed))
end

function _message_signature(message::Message)::String
    payload = Dict{String, Any}(
        "role" => String(message.role),
        "message_id" => message.message_id,
        "contents" => [
            Dict{String, Any}(
                "type" => content_type_string(content.type),
                "text" => get_text(content),
                "uri" => content.uri,
                "file_id" => content.file_id,
            )
            for content in message.contents
        ],
    )
    return JSON3.write(payload)
end

function _response_signature(response::AgentResponse)::String
    token = response.continuation_token isa A2AContinuationToken ? continuation_token_to_dict(response.continuation_token) : response.continuation_token
    payload = Dict{String, Any}(
        "messages" => [_message_signature(message) for message in response.messages],
        "response_id" => response.response_id,
        "finish_reason" => response.finish_reason === nothing ? nothing : string(response.finish_reason),
        "continuation_token" => token,
    )
    return JSON3.write(payload)
end
