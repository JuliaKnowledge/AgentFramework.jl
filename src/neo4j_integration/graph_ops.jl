# ──────────────────────────────────────────────────────────────────────────────
# graph_ops.jl — GraphRAG-style helpers for entities, relationships, retrieval
# ──────────────────────────────────────────────────────────────────────────────
#
# Mirrors the surface of graphiti's Neo4j provider where it fits cleanly:
#   - Entities are nodes with a stable `uuid`, a primary `label`, and
#     arbitrary `properties`.
#   - Relationships carry a `relationship_type`, `valid_at`, optional
#     `expired_at`, and free-form `properties`.
#   - Reads: `find_entities` (label + property filters) and `find_related`
#     (k-hop traversal with optional relationship type filter).
#   - Writes: `add_entity!`, `add_relationship!` (MERGE semantics).

"""
    Neo4jEntity(; uuid, label, name=nothing, properties=Dict())

A labelled graph node, identified by its `uuid`. `label` maps to the
Neo4j node label. Extra `properties` are merged on write.
"""
Base.@kwdef struct Neo4jEntity
    uuid::String
    label::String
    name::Union{Nothing, String} = nothing
    properties::Dict{String, Any} = Dict{String, Any}()
end

"""
    Neo4jRelationship(; source_uuid, target_uuid, relationship_type,
                        valid_at=now(), expired_at=nothing,
                        properties=Dict())

A directed edge between two entities. `relationship_type` maps to the
Cypher relationship type (an alphanumeric identifier). `valid_at` and
`expired_at` support bi-temporal invalidation.
"""
Base.@kwdef struct Neo4jRelationship
    source_uuid::String
    target_uuid::String
    relationship_type::String
    valid_at::DateTime = Dates.now(UTC)
    expired_at::Union{Nothing, DateTime} = nothing
    properties::Dict{String, Any} = Dict{String, Any}()
end

# Sanitise a user-provided label / relationship-type identifier. Neo4j
# rejects anything that isn't a valid identifier, and raw interpolation
# into Cypher risks injection.
function _safe_identifier(text::AbstractString)::String
    s = String(text)
    if !occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", s)
        throw(ArgumentError("invalid Neo4j identifier: $(repr(s)) — expected [A-Za-z_][A-Za-z0-9_]*"))
    end
    return s
end

_to_iso(dt::DateTime) = Dates.format(dt, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"

function _entity_properties(entity::Neo4jEntity)::Dict{String, Any}
    props = Dict{String, Any}(k => v for (k, v) in entity.properties)
    entity.name !== nothing && (props["name"] = entity.name)
    props["uuid"] = entity.uuid
    return props
end

"""
    add_entity!(client, entity::Neo4jEntity) -> Dict

Idempotently upsert an entity by `uuid`. Returns a dict carrying the
stored properties (as read back from Neo4j).
"""
function add_entity!(client::Neo4jClient, entity::Neo4jEntity)::Dict{String, Any}
    label = _safe_identifier(entity.label)
    props = _entity_properties(entity)
    stmt = """
    MERGE (e:`$label` {uuid: \$uuid})
    SET e += \$properties
    RETURN e { .* } AS entity
    """
    rows = cypher_query(client, stmt;
                        parameters = Dict{String, Any}(
                            "uuid" => entity.uuid,
                            "properties" => props,
                        ))
    isempty(rows) && return props
    stored = get(first(rows), "entity", nothing)
    return stored isa AbstractDict ? Dict{String, Any}(stored) : props
end

"""
    add_relationship!(client, rel::Neo4jRelationship) -> Bool

Idempotently create (or re-activate) an edge between two entities.
Returns `true` if either endpoint was found and the edge was written.
"""
function add_relationship!(client::Neo4jClient, rel::Neo4jRelationship)::Bool
    rtype = _safe_identifier(rel.relationship_type)
    props = Dict{String, Any}(k => v for (k, v) in rel.properties)
    props["valid_at"] = _to_iso(rel.valid_at)
    if rel.expired_at !== nothing
        props["expired_at"] = _to_iso(rel.expired_at)
    end

    stmt = """
    MATCH (src {uuid: \$src_uuid}), (dst {uuid: \$dst_uuid})
    MERGE (src)-[r:`$rtype`]->(dst)
    SET r += \$properties
    RETURN r { .* } AS edge
    """
    rows = cypher_query(client, stmt;
                        parameters = Dict{String, Any}(
                            "src_uuid" => rel.source_uuid,
                            "dst_uuid" => rel.target_uuid,
                            "properties" => props,
                        ))
    return !isempty(rows)
end

"""
    find_entities(client; label=nothing, where=Dict(), limit=25)
        -> Vector{Dict{String,Any}}

Fetch entities by label and optional equality filters. Each filter
pair is passed to Neo4j as a bound parameter (no string interpolation).
"""
function find_entities(client::Neo4jClient;
                       label::Union{Nothing, AbstractString} = nothing,
                       where::AbstractDict = Dict{String, Any}(),
                       limit::Integer = 25)::Vector{Dict{String, Any}}
    label_frag = label === nothing ? "" : ":`" * _safe_identifier(label) * "`"
    params = Dict{String, Any}("__limit" => Int(limit))

    conditions = String[]
    for (k, v) in where
        key = _safe_identifier(k)
        pname = "p_" * key
        push!(conditions, "e.`$key` = \$$pname")
        params[pname] = v
    end

    where_clause = isempty(conditions) ? "" : ("WHERE " * join(conditions, " AND "))

    stmt = """
    MATCH (e$label_frag)
    $where_clause
    RETURN e { .* } AS entity
    LIMIT \$__limit
    """
    rows = cypher_query(client, stmt; parameters = params)
    out = Vector{Dict{String, Any}}()
    for r in rows
        entity = get(r, "entity", nothing)
        entity isa AbstractDict && push!(out, Dict{String, Any}(entity))
    end
    return out
end

"""
    find_related(client, uuid; relationship_type=nothing, depth=1,
                 limit=25, include_expired=false) -> Vector{Dict}

K-hop traversal starting at the entity identified by `uuid`.
Each returned dict carries:

- `"entity"`     : the neighbour's properties (as a `Dict`)
- `"hop"`        : 1..depth distance
- `"relationship"`: the edge on the final hop (as a `Dict`), or `nothing`

If `include_expired=false` (default), edges whose `expired_at` is set
are excluded — matching graphiti's "live view" semantics.
"""
function find_related(
    client::Neo4jClient,
    uuid::AbstractString;
    relationship_type::Union{Nothing, AbstractString} = nothing,
    depth::Integer = 1,
    limit::Integer = 25,
    include_expired::Bool = false,
)::Vector{Dict{String, Any}}
    depth >= 1 || throw(ArgumentError("depth must be >= 1"))
    depth <= 10 || throw(ArgumentError("depth must be <= 10 (sanity limit)"))

    rel_frag = relationship_type === nothing ? "" :
               ":`" * _safe_identifier(relationship_type) * "`"

    expired_filter = include_expired ? "" : "WHERE ALL(e IN relationships(path) WHERE e.expired_at IS NULL)"

    stmt = """
    MATCH path = (src {uuid: \$uuid})-[$rel_frag*1..$(Int(depth))]->(neighbour)
    $expired_filter
    WITH neighbour, path, length(path) AS hop, last(relationships(path)) AS last_rel
    RETURN neighbour { .* } AS entity,
           hop        AS hop,
           last_rel { .* } AS relationship
    LIMIT \$__limit
    """
    params = Dict{String, Any}("uuid" => String(uuid), "__limit" => Int(limit))
    rows = cypher_query(client, stmt; parameters = params)

    out = Vector{Dict{String, Any}}()
    for r in rows
        push!(out, Dict{String, Any}(
            "entity" => r["entity"],
            "hop" => r["hop"],
            "relationship" => get(r, "relationship", nothing),
        ))
    end
    return out
end

"""
    expire_relationships!(client; relationship_type, where, at=now())

Mark all matching relationships expired by setting `expired_at`.
Returns the number of edges updated.
"""
function expire_relationships!(
    client::Neo4jClient;
    relationship_type::AbstractString,
    where::AbstractDict = Dict{String, Any}(),
    at::DateTime = Dates.now(UTC),
)::Int
    rtype = _safe_identifier(relationship_type)
    params = Dict{String, Any}("__expired_at" => _to_iso(at))

    conditions = String["r.expired_at IS NULL"]
    for (k, v) in where
        key = _safe_identifier(k)
        pname = "p_" * key
        push!(conditions, "r.`$key` = \$$pname")
        params[pname] = v
    end

    stmt = """
    MATCH ()-[r:`$rtype`]->()
    WHERE $(join(conditions, " AND "))
    SET r.expired_at = \$__expired_at
    RETURN count(r) AS updated
    """
    rows = cypher_query(client, stmt; parameters = params)
    isempty(rows) && return 0
    return Int(get(first(rows), "updated", 0))
end

"""
    clear_entities!(client; label=nothing)

Remove every node (and attached relationships) with the given label,
or the entire graph when `label` is `nothing`. Intended for tests and
one-shot migrations — use with care.
"""
function clear_entities!(client::Neo4jClient; label::Union{Nothing, AbstractString} = nothing)::Int
    label_frag = label === nothing ? "" : (":`" * _safe_identifier(label) * "`")
    stmt = """
    MATCH (n$label_frag)
    DETACH DELETE n
    RETURN count(n) AS deleted
    """
    rows = cypher_query(client, stmt)
    isempty(rows) && return 0
    return Int(get(first(rows), "deleted", 0))
end
