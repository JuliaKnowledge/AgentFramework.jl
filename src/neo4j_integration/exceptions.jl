"""
    Neo4jError(message[, status[, body]])

Raised when a Neo4j HTTP request fails, or when a Cypher statement
returns a driver-level error (e.g. a `Neo.ClientError.*` code).
"""
struct Neo4jError <: Exception
    message::String
    status::Union{Nothing, Int}
    code::Union{Nothing, String}
    body::Union{Nothing, String}
end

Neo4jError(message::AbstractString) =
    Neo4jError(String(message), nothing, nothing, nothing)
Neo4jError(message::AbstractString, status::Integer) =
    Neo4jError(String(message), Int(status), nothing, nothing)
Neo4jError(message::AbstractString, status::Integer, body) =
    Neo4jError(String(message), Int(status), nothing, body === nothing ? nothing : String(body))

function Base.showerror(io::IO, err::Neo4jError)
    print(io, "Neo4jError: ", err.message)
    err.code !== nothing && print(io, " [code=", err.code, "]")
    err.status !== nothing && print(io, " [status=", err.status, "]")
    err.body !== nothing && print(io, "\n  body: ", err.body)
end
