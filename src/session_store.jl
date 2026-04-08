# Session store for persisting agent sessions.

"""Abstract session store for persisting agent sessions."""
abstract type AbstractSessionStore end

"""Get a session by ID, returns nothing if not found."""
function load_session end

"""Save/update a session."""
function save_session! end

"""Delete a session."""
function delete_session! end

"""List all session IDs."""
function list_sessions end

"""Check if a session exists."""
function has_session end

# ── InMemorySessionStore ─────────────────────────────────────────────────────

"""
    InMemorySessionStore <: AbstractSessionStore

Thread-safe in-memory session store. Sessions are lost when the process exits.
"""
mutable struct InMemorySessionStore <: AbstractSessionStore
    sessions::Dict{String, AgentSession}
    lock::ReentrantLock
end

InMemorySessionStore() = InMemorySessionStore(Dict{String, AgentSession}(), ReentrantLock())

function load_session(store::InMemorySessionStore, id::String)::Union{Nothing, AgentSession}
    lock(store.lock) do
        get(store.sessions, id, nothing)
    end
end

function save_session!(store::InMemorySessionStore, session::AgentSession)
    lock(store.lock) do
        store.sessions[session.id] = session
    end
end

function delete_session!(store::InMemorySessionStore, id::String)::Bool
    lock(store.lock) do
        haskey(store.sessions, id) ? (delete!(store.sessions, id); true) : false
    end
end

function list_sessions(store::InMemorySessionStore)::Vector{String}
    lock(store.lock) do
        collect(keys(store.sessions))
    end
end

function has_session(store::InMemorySessionStore, id::String)::Bool
    lock(store.lock) do
        haskey(store.sessions, id)
    end
end

# ── FileSessionStore ─────────────────────────────────────────────────────────

"""
    FileSessionStore <: AbstractSessionStore

Persists sessions as JSON files in a directory. Thread-safe via ReentrantLock.
"""
Base.@kwdef struct FileSessionStore <: AbstractSessionStore
    directory::String
    lock::ReentrantLock = ReentrantLock()
end

function FileSessionStore(directory::String)
    mkpath(directory)
    FileSessionStore(directory=directory)
end

function _session_path(store::FileSessionStore, id::String)::String
    joinpath(store.directory, "$(id).json")
end

function load_session(store::FileSessionStore, id::String)::Union{Nothing, AgentSession}
    path = _session_path(store, id)
    !isfile(path) && return nothing
    lock(store.lock) do
        json = read(path, String)
        raw = JSON3.read(json)
        dict = _materialize_json(raw)
        deserialize_from_dict(dict)
    end
end

function save_session!(store::FileSessionStore, session::AgentSession)
    isdir(store.directory) || mkpath(store.directory)
    path = _session_path(store, session.id)
    lock(store.lock) do
        json = serialize_to_json(session)
        write(path, json)
    end
end

function delete_session!(store::FileSessionStore, id::String)::Bool
    path = _session_path(store, id)
    isfile(path) ? (rm(path); true) : false
end

function list_sessions(store::FileSessionStore)::Vector{String}
    !isdir(store.directory) && return String[]
    [splitext(f)[1] for f in readdir(store.directory) if endswith(f, ".json")]
end

function has_session(store::FileSessionStore, id::String)::Bool
    isfile(_session_path(store, id))
end
