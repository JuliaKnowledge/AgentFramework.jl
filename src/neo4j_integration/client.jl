# ──────────────────────────────────────────────────────────────────────────────
# client.jl — HTTP-based Neo4j driver (transactional Cypher endpoint)
# ──────────────────────────────────────────────────────────────────────────────
#
# Targets Neo4j 4.x / 5.x HTTP API:
#
#   POST {base_url}/db/{database}/tx/commit
#
# Request body:
#   { "statements": [ { "statement": "...", "parameters": {...} } ] }
#
# Authentication: HTTP Basic (user:password) or Bearer token.

const DEFAULT_NEO4J_BASE_URL = "http://localhost:7474"
const DEFAULT_NEO4J_DATABASE = "neo4j"

function _nonempty_string(value)::Union{Nothing, String}
    value === nothing && return nothing
    text = strip(String(value))
    isempty(text) && return nothing
    return text
end

_env_string(name::AbstractString) = _nonempty_string(get(ENV, String(name), nothing))

function _strip_trailing_slashes(text::AbstractString)::String
    return replace(String(text), r"/+$" => "")
end

"""
    Neo4jClient(; base_url=nothing, database="neo4j", user=nothing,
                  password=nothing, token=nothing,
                  request_runner=_default_neo4j_request)

HTTP driver for the Neo4j transactional Cypher endpoint.

Resolution order for connection settings:

- `base_url` ← `NEO4J_URL` env var ← `http://localhost:7474`.
- `database` ← `NEO4J_DATABASE` env var ← `"neo4j"`.
- `user`/`password` ← `NEO4J_USERNAME`/`NEO4J_PASSWORD` env vars.
- `token` (bearer) ← `NEO4J_TOKEN` env var. Bearer takes precedence
  over basic when both are supplied.

`request_runner` is an injection seam for tests; replace with a
function that captures calls and returns synthetic `HTTP.Response`s.
"""
mutable struct Neo4jClient
    base_url::String
    database::String
    user::Union{Nothing, String}
    password::Union{Nothing, String}
    token::Union{Nothing, String}
    request_runner::Function
end

function _default_neo4j_request(
    method::AbstractString,
    url::AbstractString;
    headers::AbstractVector{<:Pair} = Pair{String, String}[],
    body = nothing,
)
    payload = body === nothing ? nothing : JSON3.write(body)
    response = HTTP.request(
        String(method), String(url), collect(headers);
        body = payload, status_exception = false,
    )
    if response.status < 200 || response.status >= 300
        body_text = isempty(response.body) ? nothing : String(response.body)
        throw(Neo4jError("Neo4j HTTP request failed for $(method) $(url).",
                         response.status, body_text))
    end
    return response
end

function Neo4jClient(;
    base_url::Union{Nothing, AbstractString} = nothing,
    database::AbstractString = DEFAULT_NEO4J_DATABASE,
    user::Union{Nothing, AbstractString} = nothing,
    password::Union{Nothing, AbstractString} = nothing,
    token::Union{Nothing, AbstractString} = nothing,
    request_runner::Function = _default_neo4j_request,
)
    resolved_url = _nonempty_string(base_url)
    resolved_url === nothing && (resolved_url = _env_string("NEO4J_URL"))
    resolved_url === nothing && (resolved_url = DEFAULT_NEO4J_BASE_URL)
    resolved_url = _strip_trailing_slashes(resolved_url)

    resolved_db = _nonempty_string(database)
    resolved_db === nothing && (resolved_db = _env_string("NEO4J_DATABASE"))
    resolved_db === nothing && (resolved_db = DEFAULT_NEO4J_DATABASE)

    resolved_user = _nonempty_string(user)
    resolved_user === nothing && (resolved_user = _env_string("NEO4J_USERNAME"))

    resolved_password = _nonempty_string(password)
    resolved_password === nothing && (resolved_password = _env_string("NEO4J_PASSWORD"))

    resolved_token = _nonempty_string(token)
    resolved_token === nothing && (resolved_token = _env_string("NEO4J_TOKEN"))

    return Neo4jClient(
        resolved_url,
        resolved_db,
        resolved_user,
        resolved_password,
        resolved_token,
        request_runner,
    )
end

function Base.show(io::IO, client::Neo4jClient)
    auth = client.token !== nothing ? "bearer" :
           client.user !== nothing ? "basic" : "none"
    print(io, "Neo4jClient(\"", client.base_url,
              "\", database=\"", client.database,
              "\", auth=", auth, ")")
end

function _neo4j_headers(client::Neo4jClient)::Vector{Pair{String, String}}
    headers = Pair{String, String}[
        "Content-Type" => "application/json",
        "Accept" => "application/json",
    ]
    if client.token !== nothing
        push!(headers, "Authorization" => "Bearer " * client.token)
    elseif client.user !== nothing
        pwd = client.password === nothing ? "" : client.password
        encoded = base64encode(string(client.user, ":", pwd))
        push!(headers, "Authorization" => "Basic " * encoded)
    end
    return headers
end

function _commit_url(client::Neo4jClient)::String
    return string(client.base_url, "/db/", client.database, "/tx/commit")
end

function _neo4j_request(client::Neo4jClient, method::AbstractString, path::AbstractString;
                       body = nothing)
    url = startswith(path, "http") ? String(path) :
          string(client.base_url, startswith(path, "/") ? path : string("/", path))
    return client.request_runner(
        String(method), url;
        headers = _neo4j_headers(client),
        body = body,
    )
end

function _read_payload(response::HTTP.Response)
    body_text = strip(String(response.body))
    isempty(body_text) && return nothing
    return JSON3.read(body_text)
end

"""
    ping(client::Neo4jClient) -> Bool

Issue a trivial `RETURN 1` against the configured database. Returns
`true` on success, or throws a [`Neo4jError`](@ref) carrying the
underlying transport or driver error.
"""
function ping(client::Neo4jClient)::Bool
    result = cypher_query(client, "RETURN 1 AS one")
    return !isempty(result) && get(first(result), "one", nothing) == 1
end
