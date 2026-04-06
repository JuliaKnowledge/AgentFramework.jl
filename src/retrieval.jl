# Retrieval-oriented memory stores and context providers for AgentFramework.jl.

const DEFAULT_MEMORY_CONTEXT_PROMPT = "## Memories\nConsider the following memories when answering the user:"
const MEMORY_TIMESTAMP_FORMAT = dateformat"yyyy-mm-dd\THH:MM:SS\Z"

abstract type AbstractMemoryStore end

"""
    MemoryRecord

Persistent memory item stored in a memory backend.
"""
Base.@kwdef mutable struct MemoryRecord
    id::String = string(UUIDs.uuid4())
    scope::String
    kind::Symbol = :episodic
    role::Symbol = ROLE_USER
    content::String
    created_at::DateTime = Dates.now(Dates.UTC)
    session_id::Union{Nothing, String} = nothing
    author_name::Union{Nothing, String} = nothing
    message_id::Union{Nothing, String} = nothing
    metadata::Dict{String, Any} = Dict{String, Any}()
end

"""
    MemorySearchResult

Result row returned by `search_memories`.
"""
Base.@kwdef struct MemorySearchResult
    record::MemoryRecord
    score::Float64 = 0.0
end

function Base.show(io::IO, result::MemorySearchResult)
    print(io, "MemorySearchResult(score=", round(result.score; digits=3), ", ", result.record, ")")
end

function Base.show(io::IO, record::MemoryRecord)
    preview = length(record.content) > 48 ? record.content[1:45] * "..." : record.content
    print(io, "MemoryRecord(kind=:",
        record.kind,
        ", \"",
        preview,
        "\", scope=\"",
        record.scope,
        "\")",
    )
end

function add_memories! end
function search_memories end
function get_memories end
function clear_memories! end
function load_ontology! end

load_ontology!(::AbstractMemoryStore, args...; kwargs...) = throw(
    AgentError("This memory store does not support ontology loading"),
)

function _normalized_role(role)::Symbol
    return role isa Symbol ? role : Symbol(lowercase(string(role)))
end

function _memory_tokens(text::AbstractString)::Vector{String}
    tokens = split(lowercase(String(text)), r"[^[:alnum:]]+")
    return unique(filter(!isempty, tokens))
end

function _json_safe(value)
    if value === nothing || value isa AbstractString || value isa Number || value isa Bool
        return value
    elseif value isa Symbol
        return String(value)
    elseif value isa DateTime
        return Dates.format(value, MEMORY_TIMESTAMP_FORMAT)
    elseif value isa Date
        return string(value)
    elseif value isa AbstractVector
        return [_json_safe(item) for item in value]
    elseif value isa AbstractDict
        return Dict{String, Any}(string(key) => _json_safe(val) for (key, val) in pairs(value))
    end

    return string(value)
end

function _json_object(value)::Dict{String, Any}
    if value isa Dict{String, Any}
        return value
    elseif value isa AbstractDict
        return Dict{String, Any}(string(key) => val for (key, val) in pairs(value))
    elseif value === nothing
        return Dict{String, Any}()
    end

    return Dict{String, Any}()
end

function _parse_datetime(value)::DateTime
    if value isa DateTime
        return value
    elseif value isa Date
        return DateTime(value)
    elseif value isa AbstractString
        stripped = strip(value)
        isempty(stripped) && return Dates.now(Dates.UTC)
        for fmt in (MEMORY_TIMESTAMP_FORMAT, dateformat"yyyy-mm-dd\THH:MM:SS")
            try
                return DateTime(stripped, fmt)
            catch
            end
        end
        try
            return DateTime(stripped)
        catch
        end
    end

    return Dates.now(Dates.UTC)
end

function _metadata_strings(metadata::Dict{String, Any}, key::String)::Vector{String}
    value = get(metadata, key, nothing)
    if value isa AbstractString
        return isempty(strip(value)) ? String[] : [String(strip(value))]
    elseif value isa Symbol
        str = String(value)
        return isempty(str) ? String[] : [str]
    elseif value isa AbstractVector
        strings = String[]
        for item in value
            if item === nothing
                continue
            end
            text = strip(string(item))
            isempty(text) || push!(strings, text)
        end
        return unique(strings)
    end

    return String[]
end

function _memory_search_text(record::MemoryRecord)::String
    parts = String[strip(record.content), String(record.kind)]
    for key in ("tags", "keywords", "concepts", "summary", "task", "subtask", "plan", "reflection", "agent_name", "tools")
        append!(parts, _metadata_strings(record.metadata, key))
    end
    unique_parts = unique(filter(!isempty, parts))
    return join(unique_parts, "\n")
end

function _record_to_dict(record::MemoryRecord)::Dict{String, Any}
    return Dict{String, Any}(
        "id" => record.id,
        "scope" => record.scope,
        "kind" => String(record.kind),
        "role" => String(record.role),
        "content" => record.content,
        "created_at" => Dates.format(record.created_at, MEMORY_TIMESTAMP_FORMAT),
        "session_id" => record.session_id,
        "author_name" => record.author_name,
        "message_id" => record.message_id,
        "metadata" => _json_safe(record.metadata),
    )
end

function _record_from_dict(data::Dict{String, Any})::MemoryRecord
    return MemoryRecord(
        id = string(get(data, "id", string(UUIDs.uuid4()))),
        scope = string(get(data, "scope", "")),
        kind = _normalized_role(get(data, "kind", :episodic)),
        role = _normalized_role(get(data, "role", ROLE_USER)),
        content = string(get(data, "content", "")),
        created_at = _parse_datetime(get(data, "created_at", Dates.now(Dates.UTC))),
        session_id = begin
            value = get(data, "session_id", nothing)
            value === nothing ? nothing : string(value)
        end,
        author_name = begin
            value = get(data, "author_name", nothing)
            value === nothing ? nothing : string(value)
        end,
        message_id = begin
            value = get(data, "message_id", nothing)
            value === nothing ? nothing : string(value)
        end,
        metadata = _json_object(get(data, "metadata", Dict{String, Any}())),
    )
end

function _score_memory_record(record::MemoryRecord, query::AbstractString, query_tokens::Vector{String})::Float64
    isempty(query_tokens) && return 0.0

    haystack = lowercase(_memory_search_text(record))
    record_tokens = Set(_memory_tokens(haystack))
    overlap = count(token -> token in record_tokens, query_tokens)
    overlap == 0 && return 0.0

    phrase = strip(lowercase(String(query)))
    phrase_bonus = (!isempty(phrase) && occursin(phrase, haystack)) ? 0.25 : 0.0
    recency_days = abs(Dates.value(Dates.now(Dates.UTC) - record.created_at)) / (1000 * 60 * 60 * 24)
    recency_bonus = 0.05 / (1 + recency_days)
    coverage = overlap / length(query_tokens)

    return coverage + phrase_bonus + recency_bonus
end

function _rank_memory_records(records::Vector{MemoryRecord}, query::AbstractString; limit::Int=5)::Vector{MemorySearchResult}
    limit > 0 || throw(ArgumentError("limit must be positive"))
    query_tokens = _memory_tokens(query)
    isempty(query_tokens) && return MemorySearchResult[]

    ranked = MemorySearchResult[]
    for record in records
        score = _score_memory_record(record, query, query_tokens)
        score > 0 && push!(ranked, MemorySearchResult(record = record, score = score))
    end

    sort!(ranked; lt = (a, b) -> a.score == b.score ? a.record.created_at > b.record.created_at : a.score > b.score)
    length(ranked) > limit && resize!(ranked, limit)
    return ranked
end

function _take_recent(records::Vector{MemoryRecord}, limit::Union{Nothing, Int})
    limit === nothing && return records
    limit > 0 || throw(ArgumentError("limit must be positive"))
    length(records) <= limit && return records
    return records[end - limit + 1:end]
end

function _trim_recent!(records::Vector{MemoryRecord}, max_records::Union{Nothing, Int})
    max_records === nothing && return records
    max_records > 0 || throw(ArgumentError("max_records_per_scope must be positive"))
    while length(records) > max_records
        popfirst!(records)
    end
    return records
end

function _validate_sql_identifier(name::AbstractString, what::AbstractString)
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", name) || throw(
        ArgumentError("$what must be a simple SQL identifier"),
    )
end

function _row_value(row, name::Symbol, index::Int)
    hasproperty(row, name) && return getproperty(row, name)
    return row[index]
end

function _memory_from_message(message::Message, scope::String, session::AgentSession)::MemoryRecord
    metadata = Dict{String, Any}()
    if !isempty(message.additional_properties)
        metadata["message_properties"] = _json_safe(message.additional_properties)
    end
    session.user_id !== nothing && (metadata["user_id"] = session.user_id)
    session.thread_id !== nothing && (metadata["thread_id"] = session.thread_id)

    return MemoryRecord(
        scope = scope,
        kind = :episodic,
        role = _normalized_role(message.role),
        content = strip(message.text),
        created_at = Dates.now(Dates.UTC),
        session_id = session.id,
        author_name = message.author_name,
        message_id = message.message_id,
        metadata = metadata,
    )
end

function _format_memory_context(
    results::Vector{MemorySearchResult},
    context_prompt::AbstractString;
    include_scores::Bool=false,
)::String
    lines = String[String(context_prompt)]
    for result in results
        prefix = include_scores ? "- [" * string(round(result.score; digits=3)) * "] " : "- "
        push!(lines, prefix * result.record.content)
    end
    return join(lines, "\n")
end

Base.@kwdef mutable struct InMemoryMemoryStore <: AbstractMemoryStore
    records_by_scope::Dict{String, Vector{MemoryRecord}} = Dict{String, Vector{MemoryRecord}}()
    max_records_per_scope::Union{Nothing, Int} = nothing
end

function add_memories!(store::InMemoryMemoryStore, records::Vector{MemoryRecord})
    for record in records
        scope_records = get!(store.records_by_scope, record.scope, MemoryRecord[])
        push!(scope_records, record)
        _trim_recent!(scope_records, store.max_records_per_scope)
    end
    return store
end

function get_memories(
    store::InMemoryMemoryStore;
    scope::Union{Nothing, String}=nothing,
    limit::Union{Nothing, Int}=nothing,
)::Vector{MemoryRecord}
    records = if scope === nothing
        collected = MemoryRecord[]
        for scope_records in values(store.records_by_scope)
            append!(collected, scope_records)
        end
        collected
    else
        copy(get(store.records_by_scope, scope, MemoryRecord[]))
    end
    sort!(records; by = record -> record.created_at)
    return _take_recent(records, limit)
end

function search_memories(
    store::InMemoryMemoryStore,
    query::AbstractString;
    scope::Union{Nothing, String}=nothing,
    limit::Int=5,
)::Vector{MemorySearchResult}
    return _rank_memory_records(get_memories(store; scope), query; limit)
end

function clear_memories!(store::InMemoryMemoryStore; scope::Union{Nothing, String}=nothing)
    if scope === nothing
        empty!(store.records_by_scope)
    else
        delete!(store.records_by_scope, scope)
    end
    return store
end

function Base.show(io::IO, store::InMemoryMemoryStore)
    print(io, "InMemoryMemoryStore(", length(store.records_by_scope), " scopes)")
end

Base.@kwdef mutable struct FileMemoryStore <: AbstractMemoryStore
    directory::String
    max_records_per_scope::Union{Nothing, Int} = nothing
end

function _ensure_memory_directory!(store::FileMemoryStore)
    isdir(store.directory) || mkpath(store.directory)
end

function _scope_file(store::FileMemoryStore, scope::AbstractString)::String
    digest = bytes2hex(SHA.sha1(scope))
    return joinpath(store.directory, digest * ".json")
end

function _load_scope_records(store::FileMemoryStore, scope::String)::Vector{MemoryRecord}
    path = _scope_file(store, scope)
    isfile(path) || return MemoryRecord[]
    payload = strip(read(path, String))
    isempty(payload) && return MemoryRecord[]
    rows = JSON3.read(payload, Vector{Dict{String, Any}})
    return [_record_from_dict(row) for row in rows]
end

function _save_scope_records(store::FileMemoryStore, scope::String, records::Vector{MemoryRecord})
    _ensure_memory_directory!(store)
    data = [_record_to_dict(record) for record in records]
    write(_scope_file(store, scope), JSON3.write(data))
    return store
end

function add_memories!(store::FileMemoryStore, records::Vector{MemoryRecord})
    isempty(records) && return store
    grouped = Dict{String, Vector{MemoryRecord}}()
    for record in records
        push!(get!(grouped, record.scope, MemoryRecord[]), record)
    end

    for (scope, pending) in grouped
        existing = _load_scope_records(store, scope)
        append!(existing, pending)
        sort!(existing; by = record -> record.created_at)
        _trim_recent!(existing, store.max_records_per_scope)
        _save_scope_records(store, scope, existing)
    end
    return store
end

function get_memories(
    store::FileMemoryStore;
    scope::Union{Nothing, String}=nothing,
    limit::Union{Nothing, Int}=nothing,
)::Vector{MemoryRecord}
    records = MemoryRecord[]
    if scope === nothing
        if !isdir(store.directory)
            return records
        end
        for file in readdir(store.directory; join=true)
            endswith(file, ".json") || continue
            payload = strip(read(file, String))
            isempty(payload) && continue
            rows = JSON3.read(payload, Vector{Dict{String, Any}})
            append!(records, (_record_from_dict(row) for row in rows))
        end
    else
        records = _load_scope_records(store, scope)
    end
    sort!(records; by = record -> record.created_at)
    return _take_recent(records, limit)
end

function search_memories(
    store::FileMemoryStore,
    query::AbstractString;
    scope::Union{Nothing, String}=nothing,
    limit::Int=5,
)::Vector{MemorySearchResult}
    return _rank_memory_records(get_memories(store; scope), query; limit)
end

function clear_memories!(store::FileMemoryStore; scope::Union{Nothing, String}=nothing)
    if scope === nothing
        isdir(store.directory) && rm(store.directory; recursive=true, force=true)
        return store
    end

    path = _scope_file(store, scope)
    isfile(path) && rm(path; force=true)
    return store
end

function Base.show(io::IO, store::FileMemoryStore)
    print(io, "FileMemoryStore(\"", store.directory, "\")")
end

Base.@kwdef mutable struct SQLiteMemoryStore <: AbstractMemoryStore
    db::SQLite.DB
    path::Union{Nothing, String} = nothing
    table_name::String = "agent_memories"
    fts_table_name::String = "agent_memories_fts"
    max_records_per_scope::Union{Nothing, Int} = nothing
    auto_create_table::Bool = true
    _initialized::Bool = false
end

SQLiteMemoryStore(path::AbstractString; kwargs...) = SQLiteMemoryStore(
    db = SQLite.DB(String(path)),
    path = String(path),
    kwargs...,
)

function _ensure_sqlite_memory_schema!(store::SQLiteMemoryStore)
    store._initialized && return
    _validate_sql_identifier(store.table_name, "table_name")
    _validate_sql_identifier(store.fts_table_name, "fts_table_name")

    if store.auto_create_table
        DBInterface.execute(store.db, """
        CREATE TABLE IF NOT EXISTS $(store.table_name) (
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            kind TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            search_text TEXT NOT NULL,
            session_id TEXT,
            author_name TEXT,
            message_id TEXT,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """)
        DBInterface.execute(store.db, """
        CREATE INDEX IF NOT EXISTS idx_$(store.table_name)_scope_created
        ON $(store.table_name) (scope, created_at)
        """)
        DBInterface.execute(store.db, """
        CREATE VIRTUAL TABLE IF NOT EXISTS $(store.fts_table_name)
        USING fts5(id UNINDEXED, scope UNINDEXED, search_text)
        """)
    end

    store._initialized = true
    return
end

function _sqlite_record_from_row(row)::MemoryRecord
    return MemoryRecord(
        id = string(_row_value(row, :id, 1)),
        scope = string(_row_value(row, :scope, 2)),
        kind = _normalized_role(_row_value(row, :kind, 3)),
        role = _normalized_role(_row_value(row, :role, 4)),
        content = string(_row_value(row, :content, 5)),
        session_id = begin
            value = _row_value(row, :session_id, 6)
            value === nothing ? nothing : string(value)
        end,
        author_name = begin
            value = _row_value(row, :author_name, 7)
            value === nothing ? nothing : string(value)
        end,
        message_id = begin
            value = _row_value(row, :message_id, 8)
            value === nothing ? nothing : string(value)
        end,
        created_at = _parse_datetime(_row_value(row, :created_at, 9)),
        metadata = begin
            raw = _row_value(row, :metadata_json, 10)
            if raw === nothing || isempty(strip(String(raw)))
                Dict{String, Any}()
            else
                JSON3.read(String(raw), Dict{String, Any})
            end
        end,
    )
end

function _sqlite_delete_ids!(store::SQLiteMemoryStore, ids::Vector{String})
    isempty(ids) && return
    for id in ids
        DBInterface.execute(store.db, "DELETE FROM $(store.table_name) WHERE id = ?", (id,))
        DBInterface.execute(store.db, "DELETE FROM $(store.fts_table_name) WHERE id = ?", (id,))
    end
end

function _sqlite_trim_scope!(store::SQLiteMemoryStore, scope::String)
    store.max_records_per_scope === nothing && return
    records = get_memories(store; scope)
    if length(records) <= store.max_records_per_scope
        return
    end

    overflow = records[1:(length(records) - store.max_records_per_scope)]
    _sqlite_delete_ids!(store, [record.id for record in overflow])
end

function add_memories!(store::SQLiteMemoryStore, records::Vector{MemoryRecord})
    isempty(records) && return store
    _ensure_sqlite_memory_schema!(store)

    DBInterface.execute(store.db, "BEGIN")
    try
        for record in records
            search_text = _memory_search_text(record)
            metadata_json = JSON3.write(_json_safe(record.metadata))
            DBInterface.execute(
                store.db,
                "INSERT OR REPLACE INTO $(store.table_name) (id, scope, kind, role, content, search_text, session_id, author_name, message_id, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.scope,
                    String(record.kind),
                    String(record.role),
                    record.content,
                    search_text,
                    record.session_id,
                    record.author_name,
                    record.message_id,
                    Dates.format(record.created_at, MEMORY_TIMESTAMP_FORMAT),
                    metadata_json,
                ),
            )
            DBInterface.execute(store.db, "DELETE FROM $(store.fts_table_name) WHERE id = ?", (record.id,))
            DBInterface.execute(
                store.db,
                "INSERT INTO $(store.fts_table_name) (id, scope, search_text) VALUES (?, ?, ?)",
                (record.id, record.scope, search_text),
            )
        end
        DBInterface.execute(store.db, "COMMIT")
    catch err
        DBInterface.execute(store.db, "ROLLBACK")
        rethrow(err)
    end

    if store.max_records_per_scope !== nothing
        for scope in unique(record.scope for record in records)
            _sqlite_trim_scope!(store, scope)
        end
    end
    return store
end

function get_memories(
    store::SQLiteMemoryStore;
    scope::Union{Nothing, String}=nothing,
    limit::Union{Nothing, Int}=nothing,
)::Vector{MemoryRecord}
    _ensure_sqlite_memory_schema!(store)
    rows = if scope === nothing
        if limit === nothing
            DBInterface.execute(
                store.db,
                "SELECT id, scope, kind, role, content, session_id, author_name, message_id, created_at, metadata_json FROM $(store.table_name) ORDER BY created_at ASC, id ASC",
            )
        else
            DBInterface.execute(
                store.db,
                "SELECT id, scope, kind, role, content, session_id, author_name, message_id, created_at, metadata_json FROM $(store.table_name) ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        end
    else
        if limit === nothing
            DBInterface.execute(
                store.db,
                "SELECT id, scope, kind, role, content, session_id, author_name, message_id, created_at, metadata_json FROM $(store.table_name) WHERE scope = ? ORDER BY created_at ASC, id ASC",
                (scope,),
            )
        else
            DBInterface.execute(
                store.db,
                "SELECT id, scope, kind, role, content, session_id, author_name, message_id, created_at, metadata_json FROM $(store.table_name) WHERE scope = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (scope, limit),
            )
        end
    end

    records = [_sqlite_record_from_row(row) for row in rows]
    if limit !== nothing
        reverse!(records)
    end
    return records
end

function _sqlite_fts_query(query::AbstractString)::Union{Nothing, String}
    tokens = _memory_tokens(query)
    isempty(tokens) && return nothing
    quoted = ["\"" * replace(token, "\"" => "\"\"") * "\"" for token in tokens]
    return join(quoted, " OR ")
end

function search_memories(
    store::SQLiteMemoryStore,
    query::AbstractString;
    scope::Union{Nothing, String}=nothing,
    limit::Int=5,
)::Vector{MemorySearchResult}
    _ensure_sqlite_memory_schema!(store)
    limit > 0 || throw(ArgumentError("limit must be positive"))
    candidate_limit = max(limit * 8, 20)
    fts_query = _sqlite_fts_query(query)
    rows = if fts_query === nothing
        like_query = "%" * strip(String(query)) * "%"
        if scope === nothing
            DBInterface.execute(
                store.db,
                "SELECT id, scope, kind, role, content, session_id, author_name, message_id, created_at, metadata_json FROM $(store.table_name) WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (like_query, candidate_limit),
            )
        else
            DBInterface.execute(
                store.db,
                "SELECT id, scope, kind, role, content, session_id, author_name, message_id, created_at, metadata_json FROM $(store.table_name) WHERE scope = ? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (scope, like_query, candidate_limit),
            )
        end
    else
        if scope === nothing
            DBInterface.execute(
                store.db,
                "SELECT m.id, m.scope, m.kind, m.role, m.content, m.session_id, m.author_name, m.message_id, m.created_at, m.metadata_json FROM $(store.table_name) AS m JOIN $(store.fts_table_name) AS f ON m.id = f.id WHERE f.search_text MATCH ? ORDER BY m.created_at DESC LIMIT ?",
                (fts_query, candidate_limit),
            )
        else
            DBInterface.execute(
                store.db,
                "SELECT m.id, m.scope, m.kind, m.role, m.content, m.session_id, m.author_name, m.message_id, m.created_at, m.metadata_json FROM $(store.table_name) AS m JOIN $(store.fts_table_name) AS f ON m.id = f.id WHERE m.scope = ? AND f.search_text MATCH ? ORDER BY m.created_at DESC LIMIT ?",
                (scope, fts_query, candidate_limit),
            )
        end
    end

    records = [_sqlite_record_from_row(row) for row in rows]
    return _rank_memory_records(records, query; limit)
end

function clear_memories!(store::SQLiteMemoryStore; scope::Union{Nothing, String}=nothing)
    _ensure_sqlite_memory_schema!(store)
    if scope === nothing
        DBInterface.execute(store.db, "DELETE FROM $(store.table_name)")
        DBInterface.execute(store.db, "DELETE FROM $(store.fts_table_name)")
        return store
    end

    ids = [record.id for record in get_memories(store; scope)]
    _sqlite_delete_ids!(store, ids)
    return store
end

function Base.show(io::IO, store::SQLiteMemoryStore)
    if store.path === nothing
        print(io, "SQLiteMemoryStore(<connection>)")
    else
        print(io, "SQLiteMemoryStore(\"", store.path, "\")")
    end
end

mutable struct RDFMemoryStore <: AbstractMemoryStore
    rdflib::Module
    graph::Any
    text_index::Any
    base_uri::String
    namespace_uri::String
    max_records_per_scope::Union{Nothing, Int}
end

function _resolve_rdflib_module(rdflib::Union{Nothing, Module}=nothing)::Module
    rdflib !== nothing && return rdflib

    if isdefined(Main, :RDFLib)
        mod = getfield(Main, :RDFLib)
        mod isa Module && return mod
    end

    try
        Base.require(Main, :RDFLib)
    catch
    end

    if isdefined(Main, :RDFLib)
        mod = getfield(Main, :RDFLib)
        mod isa Module && return mod
    end

    throw(AgentError(
        "RDFMemoryStore requires RDFLib.jl in the active environment. For local experimentation, `Pkg.develop(path=\"...\")` or `using RDFLib` before constructing the store.",
    ))
end

function _rdflib_is(mod::Module, value, typename::Symbol)::Bool
    value === nothing && return false
    return parentmodule(typeof(value)) === mod && nameof(typeof(value)) === typename
end

function _rdflib_namespace(store::RDFMemoryStore)
    return store.rdflib.Namespace(store.namespace_uri)
end

function _rdflib_predicate(store::RDFMemoryStore, name::AbstractString)
    return _rdflib_namespace(store)(String(name))
end

function _rdflib_subject(store::RDFMemoryStore, id::AbstractString)
    return store.rdflib.URIRef(store.base_uri * String(id))
end

function _rdflib_term_text(store::RDFMemoryStore, value)
    value === nothing && return nothing
    if _rdflib_is(store.rdflib, value, :Literal)
        try
            return convert(Any, value)
        catch
            return string(value)
        end
    elseif _rdflib_is(store.rdflib, value, :URIRef)
        return getproperty(value, :value)
    elseif _rdflib_is(store.rdflib, value, :BNode)
        return getproperty(value, :id)
    end
    return value
end

function _rdflib_lookup_label(store::RDFMemoryStore, concept_uri::AbstractString)
    concept = store.rdflib.URIRef(String(concept_uri))
    for predicate in (
        store.rdflib.RDFS.label,
        store.rdflib.SKOS.prefLabel,
        store.rdflib.SKOS.altLabel,
    )
        label = store.rdflib.value(store.graph, concept, predicate; default=nothing)
        label === nothing || return string(_rdflib_term_text(store, label))
    end
    return nothing
end

function _rdflib_record_search_text(store::RDFMemoryStore, record::MemoryRecord)::String
    parts = split(_memory_search_text(record), '\n')
    for concept in _metadata_strings(record.metadata, "concepts")
        if startswith(concept, "http://") || startswith(concept, "https://") || startswith(concept, "urn:")
            label = _rdflib_lookup_label(store, concept)
            label === nothing || push!(parts, label)
        end
    end
    return join(unique(filter(!isempty, parts)), "\n")
end

function _rdflib_remove_subject!(store::RDFMemoryStore, subject)
    for triple in collect(store.rdflib.triples(store.graph, (subject, nothing, nothing)))
        store.rdflib.remove!(store.graph, triple)
    end
end

function _rdflib_rebuild_index!(store::RDFMemoryStore)
    store.rdflib.build!(store.text_index, store.graph)
    store.rdflib.set_text_index!(store.text_index)
    return store
end

function _rdflib_record_from_subject(store::RDFMemoryStore, subject)
    scope = store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "scope"); default=nothing)
    content = store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "content"); default=nothing)
    scope === nothing && return nothing
    content === nothing && return nothing

    metadata_json = store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "metadataJson"); default=nothing)
    metadata = if metadata_json === nothing
        Dict{String, Any}()
    else
        JSON3.read(String(_rdflib_term_text(store, metadata_json)), Dict{String, Any})
    end

    return MemoryRecord(
        id = replace(String(_rdflib_term_text(store, subject)), store.base_uri => ""),
        scope = String(_rdflib_term_text(store, scope)),
        kind = _normalized_role(_rdflib_term_text(store, store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "kind"); default=store.rdflib.Literal("episodic")))),
        role = _normalized_role(_rdflib_term_text(store, store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "role"); default=store.rdflib.Literal("user")))),
        content = String(_rdflib_term_text(store, content)),
        created_at = _parse_datetime(_rdflib_term_text(store, store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "createdAt"); default=nothing))),
        session_id = begin
            value = store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "sessionId"); default=nothing)
            value === nothing ? nothing : String(_rdflib_term_text(store, value))
        end,
        author_name = begin
            value = store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "authorName"); default=nothing)
            value === nothing ? nothing : String(_rdflib_term_text(store, value))
        end,
        message_id = begin
            value = store.rdflib.value(store.graph, subject, _rdflib_predicate(store, "messageId"); default=nothing)
            value === nothing ? nothing : String(_rdflib_term_text(store, value))
        end,
        metadata = metadata,
    )
end

function _rdflib_memory_subjects(store::RDFMemoryStore)
    subjects = Any[]
    seen = Set{String}()
    for triple in store.rdflib.triples(
        store.graph,
        (nothing, store.rdflib.RDF.type, _rdflib_predicate(store, "Memory")),
    )
        subject = getproperty(triple, :subject)
        key = String(_rdflib_term_text(store, subject))
        key in seen && continue
        push!(seen, key)
        push!(subjects, subject)
    end
    return subjects
end

function _rdflib_trim_scope!(store::RDFMemoryStore, scope::String)
    store.max_records_per_scope === nothing && return
    records = get_memories(store; scope)
    if length(records) <= store.max_records_per_scope
        return
    end

    overflow = records[1:(length(records) - store.max_records_per_scope)]
    for record in overflow
        _rdflib_remove_subject!(store, _rdflib_subject(store, record.id))
    end
end

function _make_rdf_memory_store(;
    rdflib::Union{Nothing, Module}=nothing,
    base_uri::String="urn:agentframework:memory/",
    namespace_uri::String="urn:agentframework:memory#",
    max_records_per_scope::Union{Nothing, Int}=nothing,
)
    module_ref = _resolve_rdflib_module(rdflib)
    graph = module_ref.RDFGraph()
    module_ref.bind!(graph, "af", module_ref.Namespace(namespace_uri))
    text_index = module_ref.TextIndex()
    module_ref.build!(text_index, graph)
    module_ref.set_text_index!(text_index)
    return RDFMemoryStore(module_ref, graph, text_index, base_uri, namespace_uri, max_records_per_scope)
end

"""
    RDFMemoryStore(; rdflib=nothing, base_uri="urn:agentframework:memory/", namespace_uri="urn:agentframework:memory#", max_records_per_scope=nothing)

Experimental RDFLib-backed memory store. `RDFLib.jl` is loaded at runtime when it is
available in the active environment, so the core package remains registration-safe.
"""
RDFMemoryStore(; kwargs...) = _make_rdf_memory_store(; kwargs...)

function add_memories!(store::RDFMemoryStore, records::Vector{MemoryRecord})
    isempty(records) && return store
    for record in records
        subject = _rdflib_subject(store, record.id)
        _rdflib_remove_subject!(store, subject)

        store.rdflib.add!(store.graph, subject, store.rdflib.RDF.type, _rdflib_predicate(store, "Memory"))
        store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "scope"), store.rdflib.Literal(record.scope))
        store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "kind"), store.rdflib.Literal(String(record.kind)))
        store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "role"), store.rdflib.Literal(String(record.role)))
        store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "content"), store.rdflib.Literal(record.content))
        store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "searchText"), store.rdflib.Literal(_rdflib_record_search_text(store, record)))
        store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "createdAt"), store.rdflib.Literal(Dates.format(record.created_at, MEMORY_TIMESTAMP_FORMAT)))
        store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "metadataJson"), store.rdflib.Literal(String(JSON3.write(_json_safe(record.metadata)))))

        record.session_id === nothing || store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "sessionId"), store.rdflib.Literal(record.session_id))
        record.author_name === nothing || store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "authorName"), store.rdflib.Literal(record.author_name))
        record.message_id === nothing || store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "messageId"), store.rdflib.Literal(record.message_id))

        for tag in _metadata_strings(record.metadata, "tags")
            store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "tag"), store.rdflib.Literal(tag))
        end
        for keyword in _metadata_strings(record.metadata, "keywords")
            store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "keyword"), store.rdflib.Literal(keyword))
        end
        for tool_name in _metadata_strings(record.metadata, "tools")
            store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "tool"), store.rdflib.Literal(tool_name))
        end
        for concept in _metadata_strings(record.metadata, "concepts")
            if startswith(concept, "http://") || startswith(concept, "https://") || startswith(concept, "urn:")
                concept_uri = store.rdflib.URIRef(concept)
                store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "concept"), concept_uri)
                label = _rdflib_lookup_label(store, concept)
                label === nothing || store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "conceptLabel"), store.rdflib.Literal(label))
            else
                store.rdflib.add!(store.graph, subject, _rdflib_predicate(store, "conceptLabel"), store.rdflib.Literal(concept))
            end
        end
    end

    if store.max_records_per_scope !== nothing
        for scope in unique(record.scope for record in records)
            _rdflib_trim_scope!(store, scope)
        end
    end

    return _rdflib_rebuild_index!(store)
end

function get_memories(
    store::RDFMemoryStore;
    scope::Union{Nothing, String}=nothing,
    limit::Union{Nothing, Int}=nothing,
)::Vector{MemoryRecord}
    records = MemoryRecord[]
    for subject in _rdflib_memory_subjects(store)
        record = _rdflib_record_from_subject(store, subject)
        record === nothing && continue
        if scope !== nothing && record.scope != scope
            continue
        end
        push!(records, record)
    end
    sort!(records; by = record -> record.created_at)
    return _take_recent(records, limit)
end

function search_memories(
    store::RDFMemoryStore,
    query::AbstractString;
    scope::Union{Nothing, String}=nothing,
    limit::Int=5,
)::Vector{MemorySearchResult}
    limit > 0 || throw(ArgumentError("limit must be positive"))
    candidates = MemoryRecord[]
    seen = Set{String}()
    triples = store.rdflib.text_search(store.text_index, query; limit=max(limit * 12, 40))
    for triple in triples
        subject = getproperty(triple, :subject)
        record = _rdflib_record_from_subject(store, subject)
        record === nothing && continue
        if scope !== nothing && record.scope != scope
            continue
        end
        record.id in seen && continue
        push!(seen, record.id)
        push!(candidates, record)
    end
    return _rank_memory_records(candidates, query; limit)
end

function clear_memories!(store::RDFMemoryStore; scope::Union{Nothing, String}=nothing)
    for subject in _rdflib_memory_subjects(store)
        record = _rdflib_record_from_subject(store, subject)
        record === nothing && continue
        if scope !== nothing && record.scope != scope
            continue
        end
        _rdflib_remove_subject!(store, subject)
    end
    return _rdflib_rebuild_index!(store)
end

function load_ontology!(store::RDFMemoryStore, data::AbstractString; format::Symbol=:turtle)
    fmt = if format == :turtle
        store.rdflib.TurtleFormat()
    elseif format == :ntriples
        store.rdflib.NTriplesFormat()
    elseif format == :rdfxml
        store.rdflib.RDFXMLFormat()
    elseif format == :jsonld
        store.rdflib.JSONLDFormat()
    else
        throw(ArgumentError("Unsupported ontology format: $format"))
    end

    store.rdflib.parse_rdf!(store.graph, data, fmt)
    return _rdflib_rebuild_index!(store)
end

function Base.show(io::IO, store::RDFMemoryStore)
    print(io, "RDFMemoryStore(\"", store.base_uri, "\")")
end

Base.@kwdef mutable struct MemoryContextProvider{S<:AbstractMemoryStore} <: BaseContextProvider
    store::S
    source_id::String = "memory"
    scope::Union{Nothing, String} = nothing
    scope_metadata_key::String = "memory_scope"
    context_prompt::String = DEFAULT_MEMORY_CONTEXT_PROMPT
    max_results::Int = 5
    include_scores::Bool = false
    store_roles::Vector{Symbol} = [ROLE_USER, ROLE_ASSISTANT]
end

function _memory_scope(provider::MemoryContextProvider, session::AgentSession)::String
    provider.scope !== nothing && return provider.scope

    if haskey(session.metadata, provider.scope_metadata_key)
        value = session.metadata[provider.scope_metadata_key]
        value !== nothing && !isempty(strip(string(value))) && return strip(string(value))
    end

    session.user_id !== nothing && return session.user_id
    session.thread_id !== nothing && return session.thread_id
    return session.id
end

function before_run!(
    provider::MemoryContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    query_messages = [message for message in ctx.input_messages if !isempty(strip(message.text))]
    isempty(query_messages) && return nothing

    scope = _memory_scope(provider, session)
    query = join((message.text for message in query_messages), "\n")
    results = search_memories(provider.store, query; scope, limit = provider.max_results)
    state["last_query"] = query
    state["last_result_count"] = length(results)

    isempty(results) && return nothing
    extend_messages!(
        ctx,
        provider,
        [Message(ROLE_USER, _format_memory_context(results, provider.context_prompt; include_scores = provider.include_scores))],
    )
    return nothing
end

function after_run!(
    provider::MemoryContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    scope = _memory_scope(provider, session)
    allowed_roles = Set(provider.store_roles)
    memories = MemoryRecord[]

    for message in ctx.input_messages
        text = strip(message.text)
        isempty(text) && continue
        _normalized_role(message.role) in allowed_roles || continue
        push!(memories, _memory_from_message(message, scope, session))
    end

    if ctx.response !== nothing && hasproperty(ctx.response, :messages)
        for message in ctx.response.messages
            text = strip(message.text)
            isempty(text) && continue
            _normalized_role(message.role) in allowed_roles || continue
            push!(memories, _memory_from_message(message, scope, session))
        end
    end

    isempty(memories) && return nothing
    add_memories!(provider.store, memories)
    state["stored_count"] = get(state, "stored_count", 0) + length(memories)
    return nothing
end

function Base.show(io::IO, provider::MemoryContextProvider)
    print(io, "MemoryContextProvider(source_id=\"", provider.source_id, "\")")
end
