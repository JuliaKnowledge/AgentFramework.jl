# Serialization support for AgentFramework.jl
# Provides a type-registry-based serialization system for round-tripping
# agents, messages, sessions, and other core types through JSON.

# ── Type Registry ────────────────────────────────────────────────────────────

"""Global registry mapping type names to deserialization constructors."""
const _TYPE_REGISTRY = Dict{String, Function}()

"""
    register_type!(type_name::String, constructor::Function)

Register a type for deserialization. The constructor should accept a
`Dict{String, Any}` and return the deserialized object.
"""
function register_type!(type_name::String, constructor::Function)
    _TYPE_REGISTRY[type_name] = constructor
end

"""
    register_type!(::Type{T}) where T

Register a Julia type using its name. Uses generic keyword-constructor deserialization.
"""
function register_type!(::Type{T}) where T
    _TYPE_REGISTRY[string(T)] = dict -> _deserialize_struct(T, dict)
end

"""
    register_state_type!(::Type{T}) where T

Register a structured type for recursive session/workflow state serialization.
This is an alias for `register_type!`.
"""
function register_state_type!(::Type{T}) where T
    register_type!(T)
end

function _deserialize_struct(::Type{T}, dict::Dict{String, Any}) where T
    kwargs = Dict{Symbol, Any}()
    for field in fieldnames(T)
        key = String(field)
        if haskey(dict, key)
            kwargs[field] = _deserialize_any_value(dict[key])
        end
    end
    T(; kwargs...)
end

function _serialize_registered_struct(value)
    type_name = string(typeof(value))
    d = Dict{String, Any}("_type" => type_name)
    for field in fieldnames(typeof(value))
        d[String(field)] = _serialize_any_value(getfield(value, field))
    end
    return d
end

function _serialize_any_value(value)
    if value === nothing || value isa AbstractString || value isa Number || value isa Bool
        return value
    elseif value isa AbstractDict
        return Dict{String, Any}(String(k) => _serialize_any_value(v) for (k, v) in pairs(value))
    elseif value isa AbstractVector
        return Any[_serialize_any_value(v) for v in value]
    elseif applicable(serialize_to_dict, value)
        return serialize_to_dict(value)
    elseif haskey(_TYPE_REGISTRY, string(typeof(value)))
        return _serialize_registered_struct(value)
    else
        return value
    end
end

function _deserialize_any_value(value)
    if value isa AbstractDict
        dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(value))
        type_name = get(dict, "_type", nothing)
        if type_name !== nothing && haskey(_TYPE_REGISTRY, type_name)
            return deserialize_from_dict(dict)
        end
        return Dict{String, Any}(k => _deserialize_any_value(v) for (k, v) in pairs(dict))
    elseif value isa AbstractVector
        return Any[_deserialize_any_value(v) for v in value]
    else
        return value
    end
end

# ── Core Interface ───────────────────────────────────────────────────────────

"""
    serialize_to_dict(obj) -> Dict{String, Any}

Serialize an object to a JSON-compatible dictionary.
Includes a `_type` field for round-trip deserialization.
"""
function serialize_to_dict end

"""
    deserialize_from_dict(dict::Dict{String, Any}) -> Any

Deserialize an object from a dictionary using the `_type` field
to look up the appropriate constructor from the type registry.
Returns the raw dict if the type is unknown.
"""
function deserialize_from_dict(dict::Dict{String, Any})
    type_name = get(dict, "_type", nothing)
    if type_name !== nothing && haskey(_TYPE_REGISTRY, type_name)
        return _TYPE_REGISTRY[type_name](dict)
    end
    return dict
end

# ── Content Serialization ────────────────────────────────────────────────────

function serialize_to_dict(c::Content)::Dict{String, Any}
    d = content_to_dict(c)
    d["_type"] = "Content"
    return d
end

# ── Message Serialization ────────────────────────────────────────────────────

function serialize_to_dict(msg::Message)::Dict{String, Any}
    d = Dict{String, Any}(
        "_type" => "Message",
        "role" => String(msg.role),
        "contents" => [serialize_to_dict(c) for c in msg.contents],
    )
    msg.author_name !== nothing && (d["author_name"] = msg.author_name)
    msg.message_id !== nothing && (d["message_id"] = msg.message_id)
    !isempty(msg.additional_properties) && (d["additional_properties"] = _serialize_any_value(msg.additional_properties))
    return d
end

# ── ChatOptions Serialization ────────────────────────────────────────────────

function serialize_to_dict(opts::ChatOptions)::Dict{String, Any}
    d = Dict{String, Any}("_type" => "ChatOptions")
    opts.model !== nothing && (d["model"] = opts.model)
    opts.temperature !== nothing && (d["temperature"] = opts.temperature)
    opts.top_p !== nothing && (d["top_p"] = opts.top_p)
    opts.max_tokens !== nothing && (d["max_tokens"] = opts.max_tokens)
    opts.stop !== nothing && (d["stop"] = _serialize_any_value(opts.stop))
    opts.tool_choice !== nothing && (d["tool_choice"] = _serialize_any_value(opts.tool_choice))
    opts.response_format !== nothing && (d["response_format"] = _serialize_any_value(opts.response_format))
    !isempty(opts.additional) && (d["additional"] = _serialize_any_value(opts.additional))
    return d
end

# ── AgentSession Serialization ───────────────────────────────────────────────

function serialize_to_dict(session::AgentSession)::Dict{String, Any}
    d = Dict{String, Any}(
        "_type" => "AgentSession",
        "id" => session.id,
    )
    !isempty(session.state) && (d["state"] = _serialize_any_value(session.state))
    session.user_id !== nothing && (d["user_id"] = session.user_id)
    session.thread_id !== nothing && (d["thread_id"] = session.thread_id)
    !isempty(session.metadata) && (d["metadata"] = _serialize_any_value(session.metadata))
    return d
end

# ── Agent Serialization (partial — client/middlewares cannot be serialized) ──

function serialize_to_dict(agent::Agent)::Dict{String, Any}
    Dict{String, Any}(
        "_type" => "Agent",
        "name" => agent.name,
        "description" => agent.description,
        "instructions" => agent.instructions,
        "options" => serialize_to_dict(agent.options),
        "max_tool_iterations" => agent.max_tool_iterations,
        "tool_names" => [t.name for t in agent.tools],
    )
end

# ── Deserialization ──────────────────────────────────────────────────────────

function _deserialize_content(d::Dict{String, Any})::Content
    d2 = copy(d)
    delete!(d2, "_type")
    return content_from_dict(d2)
end

function _deserialize_message(d::Dict{String, Any})::Message
    role = Symbol(get(d, "role", "user"))
    contents_raw = get(d, "contents", Any[])
    contents = Content[]
    for c in contents_raw
        if c isa AbstractDict
            push!(contents, _deserialize_content(Dict{String, Any}(String(k) => v for (k, v) in pairs(c))))
        end
    end
    Message(
        role = role,
        contents = contents,
        author_name = get(d, "author_name", nothing),
        message_id = get(d, "message_id", nothing),
        additional_properties = _deserialize_any_value(get(d, "additional_properties", Dict{String, Any}())),
    )
end

function _deserialize_chat_options(d::Dict{String, Any})::ChatOptions
    temp = get(d, "temperature", nothing)
    top_p = get(d, "top_p", nothing)
    max_tokens = get(d, "max_tokens", nothing)
    stop_raw = get(d, "stop", nothing)
    stop = stop_raw !== nothing ? String[string(s) for s in _deserialize_any_value(stop_raw)] : nothing
    response_format = _deserialize_any_value(get(d, "response_format", nothing))
    additional = _deserialize_any_value(get(d, "additional", Dict{String, Any}()))
    ChatOptions(
        model = get(d, "model", nothing),
        temperature = temp !== nothing ? Float64(temp) : nothing,
        top_p = top_p !== nothing ? Float64(top_p) : nothing,
        max_tokens = max_tokens !== nothing ? Int(max_tokens) : nothing,
        stop = stop,
        tool_choice = _deserialize_any_value(get(d, "tool_choice", nothing)),
        response_format = response_format,
        additional = additional,
    )
end

function _deserialize_session(d::Dict{String, Any})::AgentSession
    state = _deserialize_any_value(get(d, "state", Dict{String, Any}()))
    metadata = _deserialize_any_value(get(d, "metadata", Dict{String, Any}()))
    AgentSession(
        id = get(d, "id", string(UUIDs.uuid4())),
        state = state,
        user_id = get(d, "user_id", nothing),
        thread_id = get(d, "thread_id", nothing),
        metadata = metadata,
    )
end

# ── JSON Helpers ─────────────────────────────────────────────────────────────

"""
Recursively convert JSON3 lazy types to plain Julia Dict/Vector/scalar types.
"""
function _materialize_json(v)
    if v isa AbstractDict
        return Dict{String, Any}(String(k) => _materialize_json(val) for (k, val) in pairs(v))
    elseif v isa AbstractVector
        return Any[_materialize_json(x) for x in v]
    else
        return v
    end
end

# ── JSON Convenience ─────────────────────────────────────────────────────────

"""
    serialize_to_json(obj) -> String

Serialize an object to a JSON string via `serialize_to_dict`.
"""
function serialize_to_json(obj)::String
    JSON3.write(serialize_to_dict(obj))
end

"""
    deserialize_from_json(json::String) -> Any

Deserialize an object from a JSON string using the type registry.
"""
function deserialize_from_json(json::String)
    raw = JSON3.read(json)
    dict = _materialize_json(raw)
    return deserialize_from_dict(dict)
end

"""
    serialize_messages(messages::Vector{Message}) -> String

Serialize a vector of messages to a JSON string.
"""
function serialize_messages(messages::Vector{Message})::String
    JSON3.write([serialize_to_dict(m) for m in messages])
end

"""
    deserialize_messages(json::String) -> Vector{Message}

Deserialize a vector of messages from a JSON string.
"""
function deserialize_messages(json::String)::Vector{Message}
    raw = JSON3.read(json)
    arr = _materialize_json(raw)
    return [_deserialize_message(d) for d in arr]
end

# ── Register Built-in Types ──────────────────────────────────────────────────

function __init_serialization__()
    register_type!("Message", _deserialize_message)
    register_type!("Content", _deserialize_content)
    register_type!("ChatOptions", _deserialize_chat_options)
    register_type!("AgentSession", _deserialize_session)
end

__init_serialization__()
