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

# ── Message Group Annotations ───────────────────────────────────────────────

"""
    MessageGroup

A logical group of consecutive messages with a label and optional metadata.
Used by compaction strategies to identify and operate on message boundaries.

# Fields
- `label::String`: Human-readable group label (e.g., "tool_calls", "system", "user_turn_3").
- `start_index::Int`: First message index in the group (1-based).
- `end_index::Int`: Last message index in the group (1-based, inclusive).
- `metadata::Dict{String, Any}`: Arbitrary metadata (e.g., token counts, importance scores).
"""
Base.@kwdef struct MessageGroup
    label::String
    start_index::Int
    end_index::Int
    metadata::Dict{String, Any} = Dict{String, Any}()
end

Base.length(g::MessageGroup) = g.end_index - g.start_index + 1

"""
    group_messages(messages::Vector{Message}; by=:role) -> Vector{MessageGroup}

Partition messages into groups based on a grouping strategy.

# Strategies
- `:role` (default): consecutive messages with the same role form a group.
- `:tool_calls`: tool_call + tool_result pairs form their own groups.
- `:turns`: alternating user/assistant turns form groups.
"""
function group_messages(messages::Vector{Message}; by::Symbol=:role)::Vector{MessageGroup}
    isempty(messages) && return MessageGroup[]
    if by == :role
        return _group_by_role(messages)
    elseif by == :tool_calls
        return _group_by_tool_calls(messages)
    elseif by == :turns
        return _group_by_turns(messages)
    else
        error("Unknown grouping strategy: $by. Use :role, :tool_calls, or :turns.")
    end
end

function _group_by_role(messages::Vector{Message})::Vector{MessageGroup}
    groups = MessageGroup[]
    start = 1
    for i in 2:length(messages)
        if messages[i].role != messages[start].role
            push!(groups, MessageGroup(
                label = string(messages[start].role),
                start_index = start,
                end_index = i - 1,
            ))
            start = i
        end
    end
    push!(groups, MessageGroup(
        label = string(messages[start].role),
        start_index = start,
        end_index = length(messages),
    ))
    return groups
end

function _group_by_tool_calls(messages::Vector{Message})::Vector{MessageGroup}
    groups = MessageGroup[]
    i = 1
    n = length(messages)
    while i <= n
        msg = messages[i]
        has_tool_calls = any(c -> c.type == FUNCTION_CALL, msg.contents)
        if has_tool_calls
            group_end = i
            # Consume following tool result messages
            while group_end + 1 <= n && messages[group_end + 1].role == :tool
                group_end += 1
            end
            push!(groups, MessageGroup(
                label = "tool_calls",
                start_index = i,
                end_index = group_end,
            ))
            i = group_end + 1
        else
            push!(groups, MessageGroup(
                label = string(msg.role),
                start_index = i,
                end_index = i,
            ))
            i += 1
        end
    end
    return groups
end

function _group_by_turns(messages::Vector{Message})::Vector{MessageGroup}
    groups = MessageGroup[]
    i = 1
    n = length(messages)
    turn_num = 0
    while i <= n
        msg = messages[i]
        if msg.role == :system
            push!(groups, MessageGroup(label="system", start_index=i, end_index=i))
            i += 1
            continue
        end
        turn_num += 1
        turn_start = i
        # A turn starts with a user message and extends through assistant+tool messages
        if msg.role == :user
            turn_end = i
            while turn_end + 1 <= n && messages[turn_end + 1].role != :user && messages[turn_end + 1].role != :system
                turn_end += 1
            end
            push!(groups, MessageGroup(
                label = "turn_$turn_num",
                start_index = turn_start,
                end_index = turn_end,
            ))
            i = turn_end + 1
        else
            push!(groups, MessageGroup(
                label = "turn_$turn_num",
                start_index = i,
                end_index = i,
            ))
            i += 1
        end
    end
    return groups
end

"""
    annotate_message_groups(messages::Vector{Message}, tokenizer::AbstractTokenizer) -> Vector{MessageGroup}

Create groups using `:turns` strategy and annotate each with a `token_count` in metadata.
"""
function annotate_message_groups(messages::Vector{Message}, tokenizer)::Vector{MessageGroup}
    groups = group_messages(messages; by=:turns)
    for g in groups
        group_msgs = messages[g.start_index:g.end_index]
        g.metadata["token_count"] = count_message_tokens(tokenizer, group_msgs)
    end
    return groups
end
