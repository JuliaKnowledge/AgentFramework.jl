"""
Conversation store for managing chat sessions in the DevUI.
"""

"""
    ConversationMessage

A single message in a conversation.
"""
mutable struct ConversationMessage
    id::String
    role::Symbol
    content::String
    tool_calls::Vector{Dict{String, Any}}
    tool_results::Vector{Dict{String, Any}}
    timestamp::DateTime
    metadata::Dict{String, Any}
end

function ConversationMessage(role::Symbol, content::String;
                             tool_calls::Vector{Dict{String, Any}} = Dict{String, Any}[],
                             tool_results::Vector{Dict{String, Any}} = Dict{String, Any}[],
                             metadata::Dict{String, Any} = Dict{String, Any}())
    ConversationMessage(
        string(UUIDs.uuid4()),
        role,
        content,
        tool_calls,
        tool_results,
        Dates.now(Dates.UTC),
        metadata,
    )
end

"""
    Conversation

A conversation between a user and an entity (agent/workflow).
"""
mutable struct Conversation
    id::String
    entity_id::String
    title::String
    messages::Vector{ConversationMessage}
    session::Union{Nothing, AgentFramework.AgentSession}
    created_at::DateTime
    updated_at::DateTime
end

"""
    ConversationStore

Thread-safe store for conversations.
"""
mutable struct ConversationStore
    conversations::Dict{String, Conversation}
    lock::ReentrantLock
end

ConversationStore() = ConversationStore(Dict{String, Conversation}(), ReentrantLock())

"""
    create_conversation!(store, entity_id; session) → Conversation

Create a new conversation for the given entity.
"""
function create_conversation!(store::ConversationStore, entity_id::AbstractString;
                              session::Union{Nothing, AgentFramework.AgentSession} = nothing,
                              conversation_id::Union{Nothing, AbstractString} = nothing)
    lock(store.lock) do
        id = conversation_id === nothing ? string(UUIDs.uuid4()) : String(conversation_id)
        haskey(store.conversations, id) && error("Conversation already exists: $id")
        now = Dates.now(Dates.UTC)
        title = "New conversation"
        conv = Conversation(id, String(entity_id), title, ConversationMessage[], session, now, now)
        store.conversations[id] = conv
        return conv
    end
end

"""
    add_message!(store, conv_id, role, content; tool_calls, tool_results, metadata) → ConversationMessage

Add a message to an existing conversation.
"""
function add_message!(store::ConversationStore, conv_id::AbstractString, role::Symbol, content::AbstractString;
                      tool_calls::Vector{Dict{String, Any}} = Dict{String, Any}[],
                      tool_results::Vector{Dict{String, Any}} = Dict{String, Any}[],
                      metadata::Dict{String, Any} = Dict{String, Any}())
    lock(store.lock) do
        conv = get(store.conversations, conv_id, nothing)
        conv === nothing && error("Conversation not found: $conv_id")
        msg = ConversationMessage(role, content; tool_calls, tool_results, metadata)
        push!(conv.messages, msg)
        conv.updated_at = Dates.now(Dates.UTC)
        return msg
    end
end

"""
    get_conversation(store, id) → Conversation or nothing
"""
function get_conversation(store::ConversationStore, id::AbstractString)
    lock(store.lock) do
        return get(store.conversations, id, nothing)
    end
end

"""
    list_conversations(store) → Vector{Conversation}
"""
function list_conversations(store::ConversationStore)
    lock(store.lock) do
        return collect(values(store.conversations))
    end
end

"""
    delete_conversation!(store, id) → Bool

Delete a conversation. Returns true if it existed.
"""
function delete_conversation!(store::ConversationStore, id::AbstractString)
    lock(store.lock) do
        return delete!(store.conversations, id) !== nothing
    end
end

"""
    rename_conversation!(store, id, title) → Bool

Rename a conversation. Returns true if found.
"""
function rename_conversation!(store::ConversationStore, id::AbstractString, title::AbstractString)
    lock(store.lock) do
        conv = get(store.conversations, String(id), nothing)
        conv === nothing && return false
        conv.title = String(title)
        conv.updated_at = Dates.now(Dates.UTC)
        return true
    end
end
