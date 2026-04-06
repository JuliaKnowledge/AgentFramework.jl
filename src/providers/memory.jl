# Memory-backed history providers for AgentFramework.jl
# Persistent conversation history across sessions via SQL databases, Redis, or files.

using Dates

# ── Helpers ──────────────────────────────────────────────────────────────────

"""
    _serialize_message(msg::Message) -> String

Serialize a Message to a JSON string using `message_to_dict`.
"""
function _serialize_message(msg::Message)::String
    return String(JSON3.write(message_to_dict(msg)))
end

"""
    _deserialize_message(json::AbstractString) -> Message

Deserialize a JSON string back to a Message using `message_from_dict`.
"""
function _deserialize_message(json::AbstractString)::Message
    d = JSON3.read(json, Dict{String, Any})
    return message_from_dict(d)
end

# ── DBInterfaceHistoryProvider ───────────────────────────────────────────────

"""
    DBInterfaceHistoryProvider <: BaseHistoryProvider

History provider that stores messages in any SQL database via DBInterface.jl.

Requires the user to install and load a DBInterface-compatible package (e.g.,
`SQLite.jl`, `LibPQ.jl`, `MySQL.jl`) and pass an active connection.

# Fields
- `source_id::String`: Provider identifier for context attribution.
- `conn::Any`: A DBInterface-compatible database connection.
- `table_name::String`: Table name for storing messages (default: `"agent_messages"`).
- `max_messages::Int`: Maximum messages to load per session (default: 100).
- `auto_create_table::Bool`: Create the table on first use (default: `true`).

# Usage
```julia
using SQLite, DBInterface
db = SQLite.DB("history.db")
provider = DBInterfaceHistoryProvider(conn=db)
```
"""
Base.@kwdef mutable struct DBInterfaceHistoryProvider <: BaseHistoryProvider
    source_id::String = "db_history"
    conn::Any = nothing
    table_name::String = "agent_messages"
    max_messages::Int = 100
    auto_create_table::Bool = true
    _table_created::Bool = false
end

"""
    _ensure_table!(provider::DBInterfaceHistoryProvider)

Create the messages table if `auto_create_table` is enabled and it hasn't been created yet.
Uses `DBInterface.execute` from the caller's loaded DBInterface module.
"""
function _ensure_table!(provider::DBInterfaceHistoryProvider)
    provider._table_created && return
    if provider.auto_create_table
        sql = """
        CREATE TABLE IF NOT EXISTS $(provider.table_name) (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
        _db_execute(provider.conn, sql)
        # Create index separately (INDEX inside CREATE TABLE is not standard SQL)
        idx_sql = """
        CREATE INDEX IF NOT EXISTS idx_$(provider.table_name)_session
        ON $(provider.table_name) (session_id)
        """
        _db_execute(provider.conn, idx_sql)
    end
    provider._table_created = true
end

"""
    _db_execute(conn, sql, params=())

Execute a SQL statement via DBInterface.execute.
"""
function _db_execute(conn, sql::String, params=())
    return DBInterface.execute(conn, sql, params)
end

"""
    get_create_table_sql(provider::DBInterfaceHistoryProvider) -> String

Return the SQL DDL for the messages table. Useful for manual table creation.
"""
function get_create_table_sql(provider::DBInterfaceHistoryProvider)::String
    return """
    CREATE TABLE IF NOT EXISTS $(provider.table_name) (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
end

function get_messages(provider::DBInterfaceHistoryProvider, session_id::String)::Vector{Message}
    _ensure_table!(provider)
    sql = """
    SELECT role, content FROM $(provider.table_name)
    WHERE session_id = ?
    ORDER BY created_at ASC, id ASC
    LIMIT ?
    """
    result = _db_execute(provider.conn, sql, (session_id, provider.max_messages))
    messages = Message[]
    for row in result
        content_json = row[2] isa AbstractString ? row[2] : String(row[2])
        push!(messages, _deserialize_message(content_json))
    end
    return messages
end

function save_messages!(provider::DBInterfaceHistoryProvider, session_id::String, messages::Vector{Message})
    _ensure_table!(provider)
    sql = """
    INSERT INTO $(provider.table_name) (session_id, role, content, created_at)
    VALUES (?, ?, ?, ?)
    """
    now_str = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-dd\THH:MM:SS\Z")
    for msg in messages
        role_str = String(msg.role)
        content_json = _serialize_message(msg)
        _db_execute(provider.conn, sql, (session_id, role_str, content_json, now_str))
    end
end

function before_run!(provider::DBInterfaceHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    history = get_messages(provider, session.id)
    if !isempty(history)
        extend_messages!(ctx, provider.source_id, history)
    end
end

function after_run!(provider::DBInterfaceHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    to_save = Message[]
    append!(to_save, ctx.input_messages)
    if ctx.response !== nothing && hasproperty(ctx.response, :messages)
        append!(to_save, ctx.response.messages)
    end
    if !isempty(to_save)
        save_messages!(provider, session.id, to_save)
    end
end

function Base.show(io::IO, p::DBInterfaceHistoryProvider)
    print(io, "DBInterfaceHistoryProvider(table=\"", p.table_name, "\")")
end

# ── RedisHistoryProvider ─────────────────────────────────────────────────────

"""
    RedisHistoryProvider <: BaseHistoryProvider

History provider using Redis for fast session storage.

Requires the user to install and load a Redis package (e.g., `Jedis.jl`) and
pass an active connection. The connection must support the standard Redis command
interface via a callable pattern or `execute` method.

# Fields
- `source_id::String`: Provider identifier for context attribution.
- `conn::Any`: A Redis connection object.
- `prefix::String`: Key prefix for session data (default: `"agentframework:history:"`).
- `max_messages::Int`: Maximum messages to retain per session (default: 100).
- `ttl::Int`: Key TTL in seconds (default: 86400 = 24 hours).

# Usage
```julia
using Jedis
Jedis.connect()
provider = RedisHistoryProvider(conn=Jedis)
```
"""
Base.@kwdef mutable struct RedisHistoryProvider <: BaseHistoryProvider
    source_id::String = "redis_history"
    conn::Any = nothing
    prefix::String = "agentframework:history:"
    max_messages::Int = 100
    ttl::Int = 86400
end

"""
    _redis_execute(conn, args...)

Execute a Redis command. Supports objects with an `execute` method or
callable modules (e.g., `Jedis.execute(args...)`).
"""
function _redis_execute(conn, args...)
    if applicable(conn, args...)
        return conn(args...)
    elseif hasproperty(conn, :execute)
        return conn.execute(args...)
    else
        # Try calling execute as a module-level function
        mod = typeof(conn) isa Module ? conn : parentmodule(typeof(conn))
        if isdefined(mod, :execute)
            return Base.invokelatest(getfield(mod, :execute), args...)
        end
    end
    error("Cannot execute Redis commands with the provided connection. " *
          "The connection must support `execute(args...)` or be callable.")
end

function _redis_key(provider::RedisHistoryProvider, session_id::String)::String
    return provider.prefix * session_id
end

function get_messages(provider::RedisHistoryProvider, session_id::String)::Vector{Message}
    key = _redis_key(provider, session_id)
    raw_messages = _redis_execute(provider.conn, "LRANGE", key, "0", string(provider.max_messages - 1))
    messages = Message[]
    if raw_messages !== nothing && !isempty(raw_messages)
        for raw in raw_messages
            json_str = raw isa AbstractString ? raw : String(raw)
            push!(messages, _deserialize_message(json_str))
        end
    end
    return messages
end

function save_messages!(provider::RedisHistoryProvider, session_id::String, messages::Vector{Message})
    key = _redis_key(provider, session_id)
    for msg in messages
        json_str = _serialize_message(msg)
        _redis_execute(provider.conn, "RPUSH", key, json_str)
    end
    # Trim to max_messages
    _redis_execute(provider.conn, "LTRIM", key, string(-provider.max_messages), "-1")
    # Set/refresh TTL
    _redis_execute(provider.conn, "EXPIRE", key, string(provider.ttl))
end

function before_run!(provider::RedisHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    history = get_messages(provider, session.id)
    if !isempty(history)
        extend_messages!(ctx, provider.source_id, history)
    end
end

function after_run!(provider::RedisHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    to_save = Message[]
    append!(to_save, ctx.input_messages)
    if ctx.response !== nothing && hasproperty(ctx.response, :messages)
        append!(to_save, ctx.response.messages)
    end
    if !isempty(to_save)
        save_messages!(provider, session.id, to_save)
    end
end

function Base.show(io::IO, p::RedisHistoryProvider)
    print(io, "RedisHistoryProvider(prefix=\"", p.prefix, "\")")
end

# ── FileHistoryProvider ──────────────────────────────────────────────────────

"""
    FileHistoryProvider <: BaseHistoryProvider

File-based history provider with no external dependencies. Each session is stored
as a JSON file in the specified directory.

# Fields
- `source_id::String`: Provider identifier for context attribution.
- `directory::String`: Directory path for storing session files.
- `max_messages::Int`: Maximum messages to retain per session (default: 100).

# Usage
```julia
provider = FileHistoryProvider(directory="/tmp/agent_history")
```

Sessions are stored as `{directory}/{session_id}.json`.
"""
Base.@kwdef mutable struct FileHistoryProvider <: BaseHistoryProvider
    source_id::String = "file_history"
    directory::String
    max_messages::Int = 100
end

function _file_path(provider::FileHistoryProvider, session_id::String)::String
    return joinpath(provider.directory, session_id * ".json")
end

function _ensure_directory!(provider::FileHistoryProvider)
    if !isdir(provider.directory)
        mkpath(provider.directory)
    end
end

function get_messages(provider::FileHistoryProvider, session_id::String)::Vector{Message}
    path = _file_path(provider, session_id)
    if !isfile(path)
        return Message[]
    end
    json_str = read(path, String)
    if isempty(strip(json_str))
        return Message[]
    end
    data = JSON3.read(json_str, Vector{Dict{String, Any}})
    messages = Message[message_from_dict(d) for d in data]
    # Return only the last max_messages
    if length(messages) > provider.max_messages
        return messages[end - provider.max_messages + 1:end]
    end
    return messages
end

function save_messages!(provider::FileHistoryProvider, session_id::String, messages::Vector{Message})
    _ensure_directory!(provider)
    # Load existing messages
    existing = get_messages(provider, session_id)
    # Append new messages
    all_messages = vcat(existing, messages)
    # Truncate to max_messages
    if length(all_messages) > provider.max_messages
        all_messages = all_messages[end - provider.max_messages + 1:end]
    end
    # Serialize and write
    data = [message_to_dict(msg) for msg in all_messages]
    json_str = JSON3.write(data)
    path = _file_path(provider, session_id)
    write(path, json_str)
end

function before_run!(provider::FileHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    history = get_messages(provider, session.id)
    if !isempty(history)
        extend_messages!(ctx, provider.source_id, history)
    end
end

function after_run!(provider::FileHistoryProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    to_save = Message[]
    append!(to_save, ctx.input_messages)
    if ctx.response !== nothing && hasproperty(ctx.response, :messages)
        append!(to_save, ctx.response.messages)
    end
    if !isempty(to_save)
        save_messages!(provider, session.id, to_save)
    end
end

function Base.show(io::IO, p::FileHistoryProvider)
    print(io, "FileHistoryProvider(\"", p.directory, "\")")
end
