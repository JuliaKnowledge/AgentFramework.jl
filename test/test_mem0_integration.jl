using AgentFramework
using AgentFramework.Mem0Integration
using HTTP
using JSON3
using Test

mutable struct Mem0MockChatClient <: AbstractChatClient
    response::String
    call_count::Ref{Int}
end

Mem0MockChatClient(response::String) = Mem0MockChatClient(response, Ref(0))

function AgentFramework.get_response(
    client::Mem0MockChatClient,
    messages::Vector{Message},
    options::ChatOptions,
)::ChatResponse
    client.call_count[] += 1
    return ChatResponse(
        messages = [Message(:assistant, client.response)],
        finish_reason = STOP,
        model_id = "mem0-mock",
    )
end

function _json_response(payload)
    return HTTP.Response(200, JSON3.write(payload))
end

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
        return responses[index[]]
    end

    return calls, runner
end

function _header_value(headers, name::String)
    for (key, value) in headers
        lowercase(String(key)) == lowercase(name) && return String(value)
    end
    return nothing
end

@testset "Mem0Integration.jl" begin
    @testset "platform search injects context" begin
        calls, runner = _capture_runner([
            _json_response([
                Dict("memory" => "User likes Julia."),
                Dict("memory" => "User prefers concise answers."),
            ]),
        ])

        provider = Mem0ContextProvider(
            api_key = "test-key",
            user_id = "user-1",
            request_runner = runner,
        )
        session = AgentSession(id = "session-1")
        ctx = SessionContext(input_messages = [Message(:user, "Hello there")])
        state = Dict{String, Any}()

        before_run!(provider, nothing, session, ctx, state)

        @test length(calls) == 1
        @test calls[1].method == "POST"
        @test calls[1].url == "https://api.mem0.ai/v2/memories/search"
        @test calls[1].body["query"] == "Hello there"
        @test calls[1].body["filters"]["user_id"] == "user-1"
        @test calls[1].body["top_k"] == 5
        @test _header_value(calls[1].headers, "Authorization") == "Token test-key"
        @test occursin("User likes Julia.", only(ctx.context_messages["mem0"]).text)
        @test occursin("User prefers concise answers.", only(ctx.context_messages["mem0"]).text)
        @test state["last_query"] == "Hello there"
        @test state["last_result_count"] == 2
    end

    @testset "search uses session and agent fallbacks" begin
        calls, runner = _capture_runner([
            _json_response(Dict("results" => [Dict("memory" => "remembered fact")])),
        ])

        provider = Mem0ContextProvider(
            api_key = "test-key",
            application_id = "app-1",
            request_runner = runner,
        )
        session = AgentSession(id = "session-1", user_id = "session-user")
        ctx = SessionContext(input_messages = [Message(:user, "Question")])

        before_run!(provider, (name = "helper-agent",), session, ctx, Dict{String, Any}())

        filters = calls[1].body["filters"]
        @test filters["user_id"] == "session-user"
        @test filters["agent_id"] == "helper-agent"
        @test filters["app_id"] == "app-1"
        @test occursin("remembered fact", only(ctx.context_messages["mem0"]).text)
    end

    @testset "empty input skips search" begin
        calls, runner = _capture_runner([_json_response(Any[])])
        provider = Mem0ContextProvider(
            api_key = "test-key",
            user_id = "user-1",
            request_runner = runner,
        )
        session = AgentSession(id = "session-1")
        ctx = SessionContext(input_messages = [Message(:user, "   ")])

        before_run!(provider, nothing, session, ctx, Dict{String, Any}())

        @test isempty(calls)
        @test isempty(ctx.context_messages)
    end

    @testset "after_run stores supported roles and metadata" begin
        calls, runner = _capture_runner([
            _json_response(Dict("id" => "mem-1")),
        ])

        provider = Mem0ContextProvider(
            api_key = "test-key",
            user_id = "user-1",
            application_id = "app-1",
            request_runner = runner,
        )
        session = AgentSession(id = "session-1")
        ctx = SessionContext(input_messages = [
            Message(:system, "Rules"),
            Message(:user, "Hi"),
            Message(:tool, "tool output"),
        ])
        ctx.response = AgentResponse(messages = [
            Message(:assistant, "Hello"),
            Message(:tool, "ignored"),
        ])
        state = Dict{String, Any}()

        after_run!(provider, (name = "helper-agent",), session, ctx, state)

        @test length(calls) == 1
        @test calls[1].url == "https://api.mem0.ai/v1/memories"
        @test calls[1].body["user_id"] == "user-1"
        @test calls[1].body["agent_id"] == "helper-agent"
        @test calls[1].body["metadata"]["application_id"] == "app-1"
        @test calls[1].body["messages"] == [
            Dict("role" => "system", "content" => "Rules"),
            Dict("role" => "user", "content" => "Hi"),
            Dict("role" => "assistant", "content" => "Hello"),
        ]
        @test state["stored_count"] == 3
    end

    @testset "run_agent executes Mem0 hooks end-to-end" begin
        calls, runner = _capture_runner([
            _json_response([Dict("memory" => "User likes short answers.")]),
            _json_response(Dict("id" => "mem-2")),
        ])

        provider = Mem0ContextProvider(
            api_key = "test-key",
            request_runner = runner,
        )
        client = Mem0MockChatClient("Certainly.")
        agent = Agent(
            name = "HelpfulBot",
            instructions = "Be helpful.",
            client = client,
            context_providers = [provider],
        )
        session = AgentSession(
            id = "session-1",
            user_id = "user-42",
            metadata = Dict{String, Any}("application_id" => "app-9"),
        )

        response = run_agent(agent, "What do you remember?"; session = session)

        @test response.text == "Certainly."
        @test client.call_count[] == 1
        @test length(calls) == 2
        @test calls[1].body["filters"]["user_id"] == "user-42"
        @test calls[1].body["filters"]["agent_id"] == "HelpfulBot"
        @test calls[1].body["filters"]["app_id"] == "app-9"
        @test calls[2].body["user_id"] == "user-42"
        @test calls[2].body["agent_id"] == "HelpfulBot"
        @test calls[2].body["metadata"]["application_id"] == "app-9"
        @test calls[2].body["messages"] == [
            Dict("role" => "user", "content" => "What do you remember?"),
            Dict("role" => "assistant", "content" => "Certainly."),
        ]
    end

    @testset "oss deployment uses local endpoints and X-API-Key" begin
        calls, runner = _capture_runner([
            _json_response([Dict("memory" => "Local memory")]),
        ])

        provider = Mem0ContextProvider(
            api_key = "local-key",
            deployment = :oss,
            base_url = "http://localhost:8000/",
            user_id = "user-1",
            request_runner = runner,
        )
        session = AgentSession(id = "session-1")
        ctx = SessionContext(input_messages = [Message(:user, "Search locally")])

        before_run!(provider, nothing, session, ctx, Dict{String, Any}())

        @test calls[1].url == "http://localhost:8000/search"
        @test _header_value(calls[1].headers, "X-API-Key") == "local-key"
        @test _header_value(calls[1].headers, "Authorization") === nothing
        @test !haskey(calls[1].body, "filters")
        @test calls[1].body["user_id"] == "user-1"
        @test calls[1].body["query"] == "Search locally"
    end

    @testset "missing filters raise an error" begin
        calls, runner = _capture_runner([_json_response(Any[])])
        provider = Mem0ContextProvider(
            api_key = "test-key",
            request_runner = runner,
        )
        session = AgentSession(id = "session-1")
        ctx = SessionContext(input_messages = [Message(:user, "Hello")])

        @test_throws ArgumentError before_run!(provider, nothing, session, ctx, Dict{String, Any}())
        @test isempty(calls)
    end

    @testset "client and direct connection settings are mutually exclusive" begin
        client = Mem0Client(api_key = "test-key")
        @test_throws ArgumentError Mem0ContextProvider(client = client, api_key = "other-key")
    end
end
