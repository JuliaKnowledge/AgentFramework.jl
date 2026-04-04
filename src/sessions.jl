# Session management for AgentFramework.jl
# Mirrors Python AgentSession, SessionContext, context/history providers.

"""
    AgentSession

Lightweight session state container for agent conversations.

# Fields
- `id::String`: Unique session identifier.
- `state::Dict{String, Any}`: User-managed mutable state.
- `user_id::Union{Nothing, String}`: Optional user identifier.
- `thread_id::Union{Nothing, String}`: Service-managed thread ID (for service-backed agents).
- `metadata::Dict{String, Any}`: Session metadata.
"""
Base.@kwdef mutable struct AgentSession
    id::String = string(UUIDs.uuid4())
    state::Dict{String, Any} = Dict{String, Any}()
    user_id::Union{Nothing, String} = nothing
    thread_id::Union{Nothing, String} = nothing
    metadata::Dict{String, Any} = Dict{String, Any}()
end

function Base.show(io::IO, s::AgentSession)
    print(io, "AgentSession(\"", s.id, "\")")
end

"""
    session_to_dict(session::AgentSession) -> Dict{String, Any}
"""
function session_to_dict(session::AgentSession)::Dict{String, Any}
    d = Dict{String, Any}(
        "id" => session.id,
        "state" => session.state,
        "metadata" => session.metadata,
    )
    session.user_id !== nothing && (d["user_id"] = session.user_id)
    session.thread_id !== nothing && (d["thread_id"] = session.thread_id)
    return d
end

"""
    session_from_dict(d::Dict{String, Any}) -> AgentSession
"""
function session_from_dict(d::Dict{String, Any})::AgentSession
    AgentSession(
        id = get(d, "id", string(UUIDs.uuid4())),
        state = get(d, "state", Dict{String, Any}()),
        user_id = get(d, "user_id", nothing),
        thread_id = get(d, "thread_id", nothing),
        metadata = get(d, "metadata", Dict{String, Any}()),
    )
end

# ── SessionContext ───────────────────────────────────────────────────────────

"""
    SessionContext

Per-invocation state passed through the context provider pipeline.
Created fresh for each `run_agent()` call.

# Fields
- `session_id::Union{Nothing, String}`: ID of the current session.
- `service_session_id::Union{Nothing, String}`: Service-managed session ID.
- `input_messages::Vector{Message}`: New messages being sent to the agent.
- `context_messages::Dict{String, Vector{Message}}`: Messages added by providers, keyed by source_id.
- `instructions::Vector{String}`: Additional instructions from providers.
- `tools::Vector{FunctionTool}`: Additional tools from providers.
- `response::Union{Nothing, AgentResponse}`: Populated after invocation.
- `options::Dict{String, Any}`: Read-only options from the caller.
- `metadata::Dict{String, Any}`: Cross-provider communication channel.
"""
Base.@kwdef mutable struct SessionContext
    session_id::Union{Nothing, String} = nothing
    service_session_id::Union{Nothing, String} = nothing
    input_messages::Vector{Message} = Message[]
    context_messages::Dict{String, Vector{Message}} = Dict{String, Vector{Message}}()
    instructions::Vector{String} = String[]
    tools::Vector{FunctionTool} = FunctionTool[]
    response::Any = nothing  # Will hold AgentResponse after invocation
    options::Dict{String, Any} = Dict{String, Any}()
    metadata::Dict{String, Any} = Dict{String, Any}()
end

"""
    extend_messages!(ctx::SessionContext, source_id::String, messages::Vector{Message})

Add context messages attributed to a provider.
"""
function _resolve_context_source(source)::Tuple{String, Dict{String, String}}
    if source isa AbstractString
        source_id = String(source)
        return source_id, Dict{String, String}("source_id" => source_id)
    end

    source_id = if hasproperty(source, :source_id)
        String(getproperty(source, :source_id))
    else
        string(typeof(source))
    end

    return source_id, Dict{String, String}(
        "source_id" => source_id,
        "source_type" => string(nameof(typeof(source))),
    )
end

function extend_messages!(ctx::SessionContext, source, messages::Vector{Message})
    source_id, attribution = _resolve_context_source(source)
    if !haskey(ctx.context_messages, source_id)
        ctx.context_messages[source_id] = Message[]
    end

    copied = Message[]
    for message in messages
        msg_copy = Message(
            role = message.role,
            contents = copy(message.contents),
            author_name = message.author_name,
            message_id = message.message_id,
            additional_properties = copy(message.additional_properties),
            raw_representation = message.raw_representation,
        )
        if !haskey(msg_copy.additional_properties, "_attribution")
            msg_copy.additional_properties["_attribution"] = attribution
        end
        push!(copied, msg_copy)
    end

    append!(ctx.context_messages[source_id], copied)
end

"""
    extend_instructions!(ctx::SessionContext, instructions)

Add instructions to be prepended to the conversation.
"""
function extend_instructions!(ctx::SessionContext, instructions::Vector{String})
    append!(ctx.instructions, instructions)
end

function extend_instructions!(ctx::SessionContext, instruction::AbstractString)
    push!(ctx.instructions, String(instruction))
end

"""
    extend_tools!(ctx::SessionContext, tools::Vector{FunctionTool})

Add tools for this invocation.
"""
function extend_tools!(ctx::SessionContext, tools::Vector{FunctionTool})
    append!(ctx.tools, tools)
end

"""
    get_all_context_messages(ctx::SessionContext) -> Vector{Message}

Get all context messages flattened in provider insertion order.
"""
function get_all_context_messages(ctx::SessionContext)::Vector{Message}
    result = Message[]
    for (_, msgs) in ctx.context_messages
        append!(result, msgs)
    end
    return result
end

# ── Context Provider ─────────────────────────────────────────────────────────

"""
    BaseContextProvider

Base type for context providers. Subtype this and implement `before_run!` and/or
`after_run!` to participate in the context engineering pipeline.

# Required fields (convention)
- `source_id::String`: Unique identifier for this provider instance.
"""
abstract type BaseContextProvider <: AbstractContextProvider end

"""
    before_run!(provider, agent, session, context, state)

Called before model invocation. Override to add context (messages, instructions, tools)
to the SessionContext.
"""
function before_run! end

"""
    after_run!(provider, agent, session, context, state)

Called after model invocation. Override to process the response.
"""
function after_run! end

# Default no-ops
before_run!(::AbstractContextProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any}) = nothing
after_run!(::AbstractContextProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any}) = nothing

# ── History Provider ─────────────────────────────────────────────────────────

"""
    BaseHistoryProvider <: AbstractHistoryProvider

Base type for conversation history storage. Subtype and implement
`get_messages` and `save_messages!`.
"""
abstract type BaseHistoryProvider <: AbstractHistoryProvider end

"""
    get_messages(provider, session_id) -> Vector{Message}

Retrieve conversation history for a session.
"""
function get_messages end

"""
    save_messages!(provider, session_id, messages)

Persist messages to history storage.
"""
function save_messages! end

"""
    InMemoryHistoryProvider <: BaseHistoryProvider

In-memory conversation history store. Messages are lost when the process exits.

# Fields
- `source_id::String`: Provider identifier.
- `store::Dict{String, Vector{Message}}`: Session ID → message history.
"""
mutable struct InMemoryHistoryProvider <: BaseHistoryProvider
    source_id::String
    store::Dict{String, Vector{Message}}
end

InMemoryHistoryProvider(; source_id::String = "history") = InMemoryHistoryProvider(source_id, Dict{String, Vector{Message}}())

function get_messages(provider::InMemoryHistoryProvider, session_id::String)::Vector{Message}
    return get(provider.store, session_id, Message[])
end

function save_messages!(provider::InMemoryHistoryProvider, session_id::String, messages::Vector{Message})
    if !haskey(provider.store, session_id)
        provider.store[session_id] = Message[]
    end
    append!(provider.store[session_id], messages)
end

function before_run!(provider::InMemoryHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    history = get_messages(provider, session.id)
    if !isempty(history)
        extend_messages!(ctx, provider.source_id, history)
    end
end

function after_run!(provider::InMemoryHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    # Save input + response messages
    to_save = Message[]
    append!(to_save, ctx.input_messages)
    if ctx.response !== nothing && hasproperty(ctx.response, :messages)
        append!(to_save, ctx.response.messages)
    end
    if !isempty(to_save)
        save_messages!(provider, session.id, to_save)
    end
end

function Base.show(io::IO, p::InMemoryHistoryProvider)
    n = length(p.store)
    print(io, "InMemoryHistoryProvider(", n, " sessions)")
end

# ── Conversation Turn Management ─────────────────────────────────────────────

"""Represents a single conversation turn (user input + agent response)."""
Base.@kwdef mutable struct ConversationTurn
    index::Int
    user_messages::Vector{Message} = Message[]
    assistant_messages::Vector{Message} = Message[]
    timestamp::DateTime = Dates.now(Dates.UTC)
    metadata::Dict{String, Any} = Dict{String, Any}()
end

"""
    TurnTracker

Tracks conversation turns within a session.
"""
mutable struct TurnTracker
    turns::Vector{ConversationTurn}
    current_turn::Union{Nothing, ConversationTurn}
    lock::ReentrantLock
end

TurnTracker() = TurnTracker(ConversationTurn[], nothing, ReentrantLock())

"""Start a new turn with user messages."""
function start_turn!(tracker::TurnTracker, user_messages::Vector{Message})::ConversationTurn
    lock(tracker.lock) do
        idx = length(tracker.turns) + 1
        turn = ConversationTurn(index=idx, user_messages=copy(user_messages))
        tracker.current_turn = turn
        return turn
    end
end

"""Complete the current turn with assistant response."""
function complete_turn!(tracker::TurnTracker, assistant_messages::Vector{Message})::ConversationTurn
    lock(tracker.lock) do
        tracker.current_turn === nothing && throw(AgentError("No active turn to complete"))
        tracker.current_turn.assistant_messages = copy(assistant_messages)
        push!(tracker.turns, tracker.current_turn)
        turn = tracker.current_turn
        tracker.current_turn = nothing
        return turn
    end
end

"""Get the number of completed turns."""
turn_count(tracker::TurnTracker)::Int = length(tracker.turns)

"""Get a specific turn by index."""
function get_turn(tracker::TurnTracker, index::Int)::Union{Nothing, ConversationTurn}
    lock(tracker.lock) do
        1 <= index <= length(tracker.turns) ? tracker.turns[index] : nothing
    end
end

"""Get the last completed turn."""
function last_turn(tracker::TurnTracker)::Union{Nothing, ConversationTurn}
    lock(tracker.lock) do
        isempty(tracker.turns) ? nothing : tracker.turns[end]
    end
end

"""Get all messages from all turns in order."""
function all_turn_messages(tracker::TurnTracker)::Vector{Message}
    lock(tracker.lock) do
        msgs = Message[]
        for turn in tracker.turns
            append!(msgs, turn.user_messages)
            append!(msgs, turn.assistant_messages)
        end
        return msgs
    end
end
