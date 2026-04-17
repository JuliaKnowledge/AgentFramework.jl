using AgentFramework
using AgentFramework.Neo4jIntegration
using AgentFramework.Neo4jIntegration: _coerce_cypher_value, _safe_identifier, _to_iso, _commit_url
using Dates
using HTTP
using JSON3
using Test

# ── Mock HTTP runner ────────────────────────────────────────────────────────

"""
Create a capturing `request_runner` backed by a queue of pre-canned
responses. Each call pops one response and appends a NamedTuple record
of the request to `calls`.
"""
function _capture_runner(responses::Vector{HTTP.Response})
    calls = NamedTuple[]
    index = Ref(0)
    runner = function(method, url; headers = Pair{String, String}[], body = nothing)
        push!(calls, (
            method = String(method),
            url = String(url),
            headers = collect(headers),
            body = body,
        ))
        index[] += 1
        index[] <= length(responses) || error("mock runner: no more responses queued")
        return responses[index[]]
    end
    return calls, runner
end

"""Build the HTTP-API-shaped response body for a block of rows."""
function _tx_response(rows_blocks::Vector{<:Vector}; columns::Vector{Vector{String}} = Vector{String}[])
    if isempty(columns)
        columns = [collect(keys(isempty(block) ? Dict{String,Any}() : first(block))) for block in rows_blocks]
    end
    results = map(zip(columns, rows_blocks)) do (cols, rows)
        Dict(
            "columns" => cols,
            "data" => [Dict("row" => [row[c] for c in cols]) for row in rows],
        )
    end
    return HTTP.Response(200, JSON3.write(Dict("results" => results, "errors" => [])))
end

_tx_response(rows::Vector{<:Dict}; columns::Vector{String} = collect(keys(isempty(rows) ? Dict{String,Any}() : first(rows)))) =
    _tx_response([rows]; columns = [columns])

_error_response(code::String, message::String) =
    HTTP.Response(200, JSON3.write(Dict(
        "results" => [],
        "errors"  => [Dict("code" => code, "message" => message)],
    )))

function _header_value(headers, name::String)
    for (k, v) in headers
        lowercase(String(k)) == lowercase(name) && return String(v)
    end
    return nothing
end

# ── Tests ───────────────────────────────────────────────────────────────────

@testset "AgentFramework.Neo4jIntegration.jl" begin

    @testset "Neo4jClient configuration" begin
        client = Neo4jClient(;
            base_url = "http://neo.local:7474/",
            database = "prod",
            user = "neo4j",
            password = "secret",
        )
        @test client.base_url == "http://neo.local:7474"
        @test client.database == "prod"
        @test occursin("/db/prod/tx/commit", _commit_url(client))

        bearer = Neo4jClient(;
            base_url = "http://neo.local:7474",
            token = "tkn-123",
        )
        headers = AgentFramework.Neo4jIntegration._neo4j_headers(bearer)
        @test _header_value(headers, "Authorization") == "Bearer tkn-123"

        basic = Neo4jClient(; user = "u", password = "p")
        headers = AgentFramework.Neo4jIntegration._neo4j_headers(basic)
        @test startswith(_header_value(headers, "Authorization"), "Basic ")
    end

    @testset "cypher_query decodes rows" begin
        calls, runner = _capture_runner([
            _tx_response([Dict{String,Any}("n" => 1), Dict{String,Any}("n" => 2)]; columns = ["n"]),
        ])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)

        rows = cypher_query(client, "MATCH (n) RETURN n LIMIT 2")
        @test length(rows) == 2
        @test rows[1]["n"] == 1
        @test rows[2]["n"] == 2

        @test length(calls) == 1
        @test calls[1].method == "POST"
        @test endswith(calls[1].url, "/db/neo4j/tx/commit")
        @test calls[1].body isa AbstractDict
        @test calls[1].body["statements"][1]["statement"] == "MATCH (n) RETURN n LIMIT 2"
    end

    @testset "cypher_query surfaces driver errors" begin
        _, runner = _capture_runner([_error_response("Neo.ClientError.Statement.SyntaxError", "boom")])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)
        @test_throws Neo4jError cypher_query(client, "BAD CYPHER")
    end

    @testset "ping succeeds on RETURN 1" begin
        _, runner = _capture_runner([
            _tx_response([Dict{String,Any}("one" => 1)]; columns = ["one"]),
        ])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)
        @test ping(client) == true
    end

    @testset "add_entity! MERGEs by uuid" begin
        calls, runner = _capture_runner([
            _tx_response([Dict{String,Any}("entity" => Dict("uuid" => "u1", "name" => "Alice"))]; columns = ["entity"]),
        ])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)

        entity = Neo4jEntity(uuid = "u1", label = "Person", name = "Alice",
                             properties = Dict{String,Any}("age" => 30))
        stored = add_entity!(client, entity)

        @test stored["uuid"] == "u1"
        @test stored["name"] == "Alice"

        stmt = calls[1].body["statements"][1]["statement"]
        @test occursin("MERGE (e:`Person`", stmt)
        params = calls[1].body["statements"][1]["parameters"]
        @test params["uuid"] == "u1"
        @test params["properties"]["name"] == "Alice"
        @test params["properties"]["age"] == 30
    end

    @testset "add_entity! rejects unsafe labels" begin
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p",
                             request_runner = (args...; kwargs...) -> error("should not run"))
        bad = Neo4jEntity(uuid = "u1", label = "Person; DROP *", name = "Eve")
        @test_throws ArgumentError add_entity!(client, bad)
    end

    @testset "add_relationship! writes ISO timestamps" begin
        calls, runner = _capture_runner([
            _tx_response([Dict{String,Any}("edge" => Dict("valid_at" => "x"))]; columns = ["edge"]),
        ])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)

        rel = Neo4jRelationship(
            source_uuid = "u1", target_uuid = "u2",
            relationship_type = "KNOWS",
            valid_at = DateTime(2026, 4, 17, 1, 0, 0),
        )
        @test add_relationship!(client, rel) == true

        stmt = calls[1].body["statements"][1]["statement"]
        @test occursin("MERGE (src)-[r:`KNOWS`]->(dst)", stmt)
        params = calls[1].body["statements"][1]["parameters"]
        @test params["src_uuid"] == "u1"
        @test params["dst_uuid"] == "u2"
        @test params["properties"]["valid_at"] == "2026-04-17T01:00:00.000Z"
        @test !haskey(params["properties"], "expired_at")
    end

    @testset "find_entities applies label + filters" begin
        calls, runner = _capture_runner([
            _tx_response([Dict{String,Any}("entity" => Dict("uuid" => "u1", "name" => "Alice"))]; columns = ["entity"]),
        ])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)

        results = find_entities(client;
                                label = "Person",
                                where = Dict{String,Any}("name" => "Alice"),
                                limit = 5)
        @test length(results) == 1
        @test results[1]["uuid"] == "u1"

        stmt = calls[1].body["statements"][1]["statement"]
        @test occursin("MATCH (e:`Person`)", stmt)
        @test occursin("e.`name` = \$p_name", stmt)
        @test calls[1].body["statements"][1]["parameters"]["p_name"] == "Alice"
        @test calls[1].body["statements"][1]["parameters"]["__limit"] == 5
    end

    @testset "find_related builds traversal" begin
        row = Dict{String,Any}(
            "entity"       => Dict("uuid" => "u2", "name" => "Bob"),
            "hop"          => 1,
            "relationship" => Dict("valid_at" => "2026-01-01T00:00:00.000Z"),
        )
        _, runner = _capture_runner([_tx_response([row]; columns = ["entity", "hop", "relationship"])])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)

        out = find_related(client, "u1"; relationship_type = "KNOWS", depth = 2, limit = 7)
        @test length(out) == 1
        @test out[1]["entity"]["name"] == "Bob"
        @test out[1]["hop"] == 1
    end

    @testset "expire_relationships! sets expired_at" begin
        calls, runner = _capture_runner([
            _tx_response([Dict{String,Any}("updated" => 3)]; columns = ["updated"]),
        ])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)

        n = expire_relationships!(client;
                                  relationship_type = "KNOWS",
                                  at = DateTime(2026, 4, 17, 1, 0, 0))
        @test n == 3

        stmt = calls[1].body["statements"][1]["statement"]
        @test occursin("MATCH ()-[r:`KNOWS`]->()", stmt)
        @test occursin("r.expired_at IS NULL", stmt)
        @test calls[1].body["statements"][1]["parameters"]["__expired_at"] == "2026-04-17T01:00:00.000Z"
    end

    @testset "Neo4jContextProvider.before_run! injects neighbourhood" begin
        row = Dict{String,Any}(
            "entity"       => Dict("uuid" => "u2", "name" => "Bob"),
            "hop"          => 1,
            "relationship" => Dict("valid_at" => "2026-01-01T00:00:00.000Z"),
        )
        _, runner = _capture_runner([_tx_response([row]; columns = ["entity", "hop", "relationship"])])
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p", request_runner = runner)

        provider = Neo4jContextProvider(
            client = client,
            seed_uuid_fn = session -> "u1",
            depth = 1,
            limit = 5,
        )

        session = AgentSession(id = "s1")
        ctx = SessionContext(input_messages = [Message(:user, "Who does Alice know?")])
        state = Dict{String, Any}()

        before_run!(provider, nothing, session, ctx, state)

        @test state["seed_uuid"] == "u1"
        @test state["last_query"] == "Who does Alice know?"
        @test state["last_result_count"] == 1
        @test haskey(ctx.context_messages, "Neo4jContextProvider")
        ctx_msgs = ctx.context_messages["Neo4jContextProvider"]
        @test length(ctx_msgs) == 1
        @test occursin("Bob", ctx_msgs[1].text)
        @test occursin("hop 1", ctx_msgs[1].text)
    end

    @testset "Neo4jContextProvider skips when seed returns nothing" begin
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p",
                             request_runner = (args...; kwargs...) -> error("should not run"))
        provider = Neo4jContextProvider(
            client = client,
            seed_uuid_fn = session -> nothing,
        )
        session = AgentSession(id = "s1")
        ctx = SessionContext(input_messages = [Message(:user, "hi")])
        state = Dict{String, Any}()

        before_run!(provider, nothing, session, ctx, state)

        @test isempty(ctx.context_messages)
        @test !haskey(state, "seed_uuid")
    end

    @testset "after_run! invokes writer_fn once per turn" begin
        client = Neo4jClient(; base_url = "http://x", user = "u", password = "p",
                             request_runner = (args...; kwargs...) -> error("should not run"))
        calls = Ref(0)
        provider = Neo4jContextProvider(
            client = client,
            seed_uuid_fn = session -> nothing,
            writer_fn = (c, s, ctx) -> (calls[] += 1; nothing),
        )
        session = AgentSession(id = "s1")
        ctx = SessionContext(input_messages = Message[])
        state = Dict{String, Any}()

        after_run!(provider, nothing, session, ctx, state)
        after_run!(provider, nothing, session, ctx, state)

        @test calls[] == 2
        @test state["persisted_turn"] == 2
    end

    @testset "_safe_identifier + _to_iso helpers" begin
        @test _safe_identifier("Person_1") == "Person_1"
        @test_throws ArgumentError _safe_identifier("bad name")
        @test _to_iso(DateTime(2026, 1, 2, 3, 4, 5, 6)) == "2026-01-02T03:04:05.006Z"
    end
end
