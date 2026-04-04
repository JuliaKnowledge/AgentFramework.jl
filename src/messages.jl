# Message type for AgentFramework.jl
# Mirrors the Python Message class.

"""
    Role

Message author role. Standard values: `:user`, `:assistant`, `:system`, `:tool`.
"""
const Role = Symbol

const ROLE_USER = :user
const ROLE_ASSISTANT = :assistant
const ROLE_SYSTEM = :system
const ROLE_TOOL = :tool

"""
    Message

Represents a chat message with a role and content items.

# Fields
- `role::Symbol`: The author role (`:user`, `:assistant`, `:system`, `:tool`).
- `contents::Vector{Content}`: The content items in this message.
- `author_name::Union{Nothing, String}`: Optional author name.
- `message_id::Union{Nothing, String}`: Optional message ID.
- `additional_properties::Dict{String, Any}`: Extension properties (not sent to services).

# Examples
```julia
# Simple text message
msg = Message(:user, "What's the weather?")

# Message with explicit Content
msg = Message(:assistant, [text_content("It's sunny!")])

# Access aggregated text
msg.text  # via get_text(msg)
```
"""
Base.@kwdef mutable struct Message
    role::Symbol
    contents::Vector{Content} = Content[]
    author_name::Union{Nothing, String} = nothing
    message_id::Union{Nothing, String} = nothing
    additional_properties::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end

# Convenience constructors

"""
    Message(role, text::AbstractString) -> Message

Create a message with a single text content item.
"""
Message(role::Symbol, text::AbstractString) = Message(role=role, contents=[text_content(text)])

"""
    Message(role, contents::Vector{Content}) -> Message

Create a message with explicit content items.
"""
Message(role::Symbol, contents::Vector{Content}) = Message(role=role, contents=contents)

"""
    Message(role, contents::Vector) -> Message

Create a message from mixed content (strings auto-converted to text content).
"""
function Message(role::Symbol, contents::Vector)
    parsed = Content[]
    for item in contents
        if item isa Content
            push!(parsed, item)
        elseif item isa AbstractString
            push!(parsed, text_content(item))
        elseif item isa Dict
            push!(parsed, content_from_dict(item))
        else
            push!(parsed, text_content(string(item)))
        end
    end
    Message(role=role, contents=parsed)
end

"""
    get_text(msg::Message) -> String

Concatenate text from all text content items in the message.
"""
function get_text(msg::Message)::String
    join((get_text(c) for c in msg.contents if is_text(c)), " ")
end

# Property-style access
function Base.getproperty(msg::Message, name::Symbol)
    if name === :text
        return get_text(msg)
    else
        return getfield(msg, name)
    end
end

function Base.propertynames(msg::Message, private::Bool=false)
    return (:role, :contents, :author_name, :message_id, :additional_properties, :raw_representation, :text)
end

function Base.show(io::IO, msg::Message)
    txt = get_text(msg)
    preview = length(txt) > 60 ? txt[1:57] * "..." : txt
    print(io, "Message(:", msg.role, ", \"", preview, "\")")
end

function Base.:(==)(a::Message, b::Message)
    a.role == b.role || return false
    a.contents == b.contents || return false
    a.author_name == b.author_name || return false
    a.message_id == b.message_id || return false
    return true
end

# ── Input Normalization ──────────────────────────────────────────────────────

"""
    AgentRunInputs

Union of types accepted as agent run input — will be normalized to `Vector{Message}`.
"""
const AgentRunInputs = Union{String, Content, Message, Vector{String}, Vector{Content}, Vector{Message}, Vector{Any}}

"""
    normalize_messages(inputs) -> Vector{Message}

Normalize various input formats into a list of Message objects.

Accepts: `String`, `Content`, `Message`, or vectors of these.
Everything is wrapped in user-role messages.
"""
function normalize_messages(inputs::Nothing)::Vector{Message}
    return Message[]
end

function normalize_messages(inputs::AbstractString)::Vector{Message}
    return [Message(:user, inputs)]
end

function normalize_messages(inputs::Content)::Vector{Message}
    return [Message(role=:user, contents=[inputs])]
end

function normalize_messages(inputs::Message)::Vector{Message}
    return [inputs]
end

function normalize_messages(inputs::Vector{Message})::Vector{Message}
    return inputs
end

function normalize_messages(inputs::Vector)::Vector{Message}
    messages = Message[]
    for item in inputs
        if item isa Message
            push!(messages, item)
        elseif item isa Content
            push!(messages, Message(role=:user, contents=[item]))
        elseif item isa AbstractString
            push!(messages, Message(:user, item))
        else
            push!(messages, Message(:user, string(item)))
        end
    end
    return messages
end

"""
    prepend_instructions(messages, instructions) -> Vector{Message}

Prepend system-role instruction messages to a message list.
"""
function prepend_instructions(messages::Vector{Message}, instructions::Vector{String})::Vector{Message}
    isempty(instructions) && return messages
    sys_msgs = [Message(:system, inst) for inst in instructions]
    return vcat(sys_msgs, messages)
end

function prepend_instructions(messages::Vector{Message}, instruction::AbstractString)::Vector{Message}
    return prepend_instructions(messages, [String(instruction)])
end

# ── Serialization ────────────────────────────────────────────────────────────

"""
    message_to_dict(msg::Message; exclude_none=true) -> Dict{String, Any}

Serialize a Message to a Dict.
"""
function message_to_dict(msg::Message; exclude_none::Bool = true)::Dict{String, Any}
    d = Dict{String, Any}(
        "type" => "chat_message",
        "role" => String(msg.role),
        "contents" => [content_to_dict(c; exclude_none) for c in msg.contents],
    )
    if !exclude_none || msg.author_name !== nothing
        d["author_name"] = msg.author_name
    end
    if !exclude_none || msg.message_id !== nothing
        d["message_id"] = msg.message_id
    end
    if !isempty(msg.additional_properties)
        d["additional_properties"] = msg.additional_properties
    end
    return d
end

"""
    message_from_dict(d::Dict{String, Any}) -> Message

Deserialize a Message from a Dict.
"""
function message_from_dict(d::Dict{String, Any})::Message
    role = Symbol(d["role"]::String)
    contents_data = get(d, "contents", Any[])
    contents = Content[
        item isa Dict ? content_from_dict(item) : text_content(string(item))
        for item in contents_data
    ]
    # Handle legacy "text" field
    if haskey(d, "text") && d["text"] isa String
        push!(contents, text_content(d["text"]))
    end
    Message(
        role = role,
        contents = contents,
        author_name = get(d, "author_name", nothing),
        message_id = get(d, "message_id", nothing),
        additional_properties = get(d, "additional_properties", Dict{String, Any}()),
    )
end
