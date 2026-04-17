# ──────────────────────────────────────────────────────────────────────────────
# context_provider.jl — AgentFramework context provider backed by Neo4j
# ──────────────────────────────────────────────────────────────────────────────
#
# Retrieval strategy:
#
#   before_run!:
#     1. Take the latest user message text as the search query.
#     2. Resolve a "seed" uuid from `seed_uuid_fn(session)` — typically
#        a per-user or per-session anchor node.
#     3. Traverse the graph up to `depth` hops from the seed, filter by
#        `relationship_type` if supplied, drop expired edges.
#     4. Format the k-hop neighbourhood as a markdown block and inject
#        it as a user message ahead of the model call (matching the
#        Mem0 provider's approach).
#
# after_run!:
#   No-op by default. Pass a `writer_fn(client, session, ctx)` to persist
#   extracted entities/relationships — left open because extraction
#   strategies vary per application.

const DEFAULT_GRAPH_CONTEXT_PROMPT =
    "## Knowledge graph context\nConsider the following related facts from the knowledge graph:"

"""
    Neo4jContextProvider(; client, seed_uuid_fn, ...)

Context provider that injects graph neighbourhood context before each
model call. Required kwargs:

- `client` — a `Neo4jClient`.
- `seed_uuid_fn` — a callable `(session::AgentSession) -> Union{String,Nothing}`
  returning the uuid of an anchor node to traverse from. Returning
  `nothing` skips injection for this turn.

Optional kwargs:

- `relationship_type` — restrict traversal to one edge type.
- `depth` (default `2`) — k-hop radius.
- `limit` (default `10`) — maximum rows returned.
- `include_expired` (default `false`) — whether to follow expired edges.
- `context_prompt` — header inserted before the rendered facts.
- `writer_fn` — optional `(client, session, ctx) -> Nothing` called in
  `after_run!` to persist new entities/relationships.
"""
mutable struct Neo4jContextProvider <: BaseContextProvider
    client::Neo4jClient
    seed_uuid_fn::Function
    relationship_type::Union{Nothing, String}
    depth::Int
    limit::Int
    include_expired::Bool
    context_prompt::String
    writer_fn::Union{Nothing, Function}
end

function Neo4jContextProvider(;
    client::Neo4jClient,
    seed_uuid_fn::Function,
    relationship_type::Union{Nothing, AbstractString} = nothing,
    depth::Integer = 2,
    limit::Integer = 10,
    include_expired::Bool = false,
    context_prompt::AbstractString = DEFAULT_GRAPH_CONTEXT_PROMPT,
    writer_fn::Union{Nothing, Function} = nothing,
)
    depth >= 1 || throw(ArgumentError("depth must be >= 1"))
    limit >= 1 || throw(ArgumentError("limit must be >= 1"))

    return Neo4jContextProvider(
        client,
        seed_uuid_fn,
        _nonempty_string(relationship_type),
        Int(depth),
        Int(limit),
        include_expired,
        String(context_prompt),
        writer_fn,
    )
end

function Base.show(io::IO, provider::Neo4jContextProvider)
    print(io, "Neo4jContextProvider(client=", provider.client,
              ", depth=", provider.depth,
              ", limit=", provider.limit, ")")
end

function _format_entity(entity)::Union{Nothing, String}
    entity isa AbstractDict || return nothing
    name = get(entity, "name", nothing)
    uuid = get(entity, "uuid", nothing)
    label = get(entity, "__label__", nothing)
    bits = String[]
    name !== nothing && push!(bits, String(name))
    if uuid !== nothing
        push!(bits, "(uuid=$(String(uuid)))")
    end
    isempty(bits) && return nothing
    label === nothing ? join(bits, " ") : string(label, ": ", join(bits, " "))
end

function _format_relationship(rel)::Union{Nothing, String}
    rel isa AbstractDict || return nothing
    # Properties only; relationship type is not carried in {.*}.
    items = String[]
    for (k, v) in rel
        k == "uuid" && continue
        push!(items, string(k, "=", v))
    end
    isempty(items) && return nothing
    return join(items, ", ")
end

function _format_graph_context(rows::AbstractVector{<:AbstractDict},
                               prompt::AbstractString)::Union{Nothing, String}
    lines = String[]
    for row in rows
        entity = get(row, "entity", nothing)
        hop    = get(row, "hop", nothing)
        rel    = get(row, "relationship", nothing)

        entity_text = _format_entity(entity)
        entity_text === nothing && continue

        rel_text = _format_relationship(rel)
        hop_suffix = hop === nothing ? "" : " (hop $(hop))"
        rel_suffix = rel_text === nothing ? "" : " [$(rel_text)]"
        push!(lines, string("- ", entity_text, rel_suffix, hop_suffix))
    end
    isempty(lines) && return nothing
    return string(String(prompt), "\n", join(lines, "\n"))
end

function _latest_user_query(ctx::SessionContext)::Union{Nothing, String}
    for msg in Iterators.reverse(ctx.input_messages)
        text = strip(msg.text)
        isempty(text) && continue
        return String(text)
    end
    return nothing
end

function before_run!(
    provider::Neo4jContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    seed = provider.seed_uuid_fn(session)
    seed === nothing && return nothing
    seed_str = _nonempty_string(seed)
    seed_str === nothing && return nothing

    query = _latest_user_query(ctx)
    state["last_query"] = query
    state["seed_uuid"] = seed_str

    rows = find_related(
        provider.client, seed_str;
        relationship_type = provider.relationship_type,
        depth = provider.depth,
        limit = provider.limit,
        include_expired = provider.include_expired,
    )
    state["last_result_count"] = length(rows)

    context_text = _format_graph_context(rows, provider.context_prompt)
    context_text === nothing && return nothing

    extend_messages!(ctx, provider, [Message(ROLE_USER, context_text)])
    return nothing
end

function after_run!(
    provider::Neo4jContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    provider.writer_fn === nothing && return nothing
    provider.writer_fn(provider.client, session, ctx)
    state["persisted_turn"] = get(state, "persisted_turn", 0) + 1
    return nothing
end
