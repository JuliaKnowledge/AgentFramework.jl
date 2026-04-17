# ──────────────────────────────────────────────────────────────────────────────
# cypher.jl — Cypher transaction execution + result decoding
# ──────────────────────────────────────────────────────────────────────────────
#
# The HTTP tx/commit endpoint returns:
#
#   { "results":  [ { "columns": [...], "data": [ { "row": [...], ... } ] } ],
#     "errors":   [ { "code": "...", "message": "..." } ] }
#
# We return a Vector{Dict{String,Any}} — one dict per row, keyed by
# column name — which matches the shape most Cypher callers want.

function _coerce_cypher_value(value)
    value === nothing && return nothing
    if value isa JSON3.Array
        return [_coerce_cypher_value(v) for v in value]
    elseif value isa AbstractVector
        return [_coerce_cypher_value(v) for v in value]
    elseif value isa JSON3.Object
        return Dict{String, Any}(String(k) => _coerce_cypher_value(v) for (k, v) in pairs(value))
    elseif value isa AbstractDict
        return Dict{String, Any}(String(k) => _coerce_cypher_value(v) for (k, v) in pairs(value))
    else
        return value
    end
end

function _decode_results(payload)
    payload === nothing && throw(Neo4jError("Neo4j returned an empty body."))

    if haskey(payload, "errors")
        errors = payload["errors"]
        if errors isa AbstractVector && !isempty(errors)
            err = first(errors)
            code = String(get(err, "code", "unknown"))
            msg  = String(get(err, "message", "unknown Cypher error"))
            throw(Neo4jError(msg, nothing, code, nothing))
        end
    end

    results = get(payload, "results", nothing)
    results isa AbstractVector || return Vector{Dict{String, Any}}[]
    isempty(results) && return Vector{Dict{String, Any}}[]

    decoded = Vector{Vector{Dict{String, Any}}}()
    for block in results
        cols = get(block, "columns", String[])
        column_names = String[String(c) for c in cols]
        rows = Vector{Dict{String, Any}}()
        data = get(block, "data", [])
        for entry in data
            row = get(entry, "row", nothing)
            row === nothing && continue
            dict = Dict{String, Any}()
            for (i, name) in enumerate(column_names)
                i <= length(row) || break
                dict[name] = _coerce_cypher_value(row[i])
            end
            push!(rows, dict)
        end
        push!(decoded, rows)
    end
    return decoded
end

# Override throw to attach status from Neo4jError raised by client
function _post_cypher(client::Neo4jClient, statements::AbstractVector)
    body = Dict{String, Any}("statements" => collect(statements))
    response = client.request_runner(
        "POST", _commit_url(client);
        headers = _neo4j_headers(client),
        body = body,
    )
    payload = _read_payload(response)
    return _decode_results(payload)
end

"""
    cypher_query(client, statement; parameters=nothing) -> Vector{Dict{String,Any}}
    cypher_query(client, statements::Vector) -> Vector{Vector{Dict{String,Any}}}

Execute one or more Cypher statements and return decoded rows.

With a single statement (second positional argument a `String`),
returns a `Vector{Dict}` — one entry per row, keyed by column name.

With a vector of statements (each a `NamedTuple` or a `Dict` carrying
`statement` and optional `parameters`), returns a `Vector{Vector{Dict}}`
— one rows block per statement, preserving submission order.

Raises [`Neo4jError`](@ref) if the transport, commit, or any individual
statement fails.
"""
function cypher_query(
    client::Neo4jClient,
    statement::AbstractString;
    parameters::Union{Nothing, AbstractDict} = nothing,
)::Vector{Dict{String, Any}}
    stmt = Dict{String, Any}("statement" => String(statement))
    parameters === nothing || (stmt["parameters"] = parameters)
    decoded = _post_cypher(client, [stmt])
    return isempty(decoded) ? Dict{String, Any}[] : first(decoded)
end

function cypher_query(
    client::Neo4jClient,
    statements::AbstractVector,
)::Vector{Vector{Dict{String, Any}}}
    normalized = Vector{Dict{String, Any}}()
    for stmt in statements
        if stmt isa AbstractDict
            entry = Dict{String, Any}("statement" => String(stmt["statement"]))
            haskey(stmt, "parameters") && (entry["parameters"] = stmt["parameters"])
            push!(normalized, entry)
        elseif stmt isa NamedTuple
            entry = Dict{String, Any}("statement" => String(stmt.statement))
            hasproperty(stmt, :parameters) && (entry["parameters"] = stmt.parameters)
            push!(normalized, entry)
        elseif stmt isa AbstractString
            push!(normalized, Dict{String, Any}("statement" => String(stmt)))
        else
            throw(ArgumentError("Unsupported statement shape: $(typeof(stmt))."))
        end
    end
    return _post_cypher(client, normalized)
end

"""
    cypher_write(client, statement; parameters=nothing) -> Int

Convenience wrapper for write queries that don't return useful rows.
Returns the number of result rows reported (usually 0 for pure writes).
"""
function cypher_write(
    client::Neo4jClient,
    statement::AbstractString;
    parameters::Union{Nothing, AbstractDict} = nothing,
)::Int
    rows = cypher_query(client, statement; parameters = parameters)
    return length(rows)
end
