using AgentFramework
using AzureIdentity
using Dates
using HTTP
using JSON3
using Test

Base.@kwdef struct MockAzureCredential <: AzureIdentity.AbstractAzureCredential
    token::String = "mock-token"
end

function AzureIdentity.get_token_info(
    credential::MockAzureCredential,
    scopes::Vararg{String};
    kwargs...,
)
    return AzureIdentity.AzureAccessTokenInfo(
        token = credential.token,
        expires_on = now() + Hour(1),
        scopes = collect(scopes),
    )
end

@testset "AzureIdentity extension" begin
    @test Base.get_extension(AgentFramework, :AgentFrameworkAzureIdentityExt) !== nothing

    @testset "credential-backed Azure headers" begin
        client = AzureOpenAIChatClient(
            model = "gpt-4o",
            endpoint = "https://myresource.openai.azure.com",
            credential = MockAzureCredential(),
        )
        headers = Dict(AgentFramework._build_headers(client))
        @test headers["Authorization"] == "Bearer mock-token"
        @test !haskey(headers, "api-key")
    end

    @testset "unsupported credential type throws" begin
        client = AzureOpenAIChatClient(
            model = "gpt-4o",
            endpoint = "https://myresource.openai.azure.com",
            credential = Ref("not-a-credential"),
        )
        err = try
            AgentFramework._build_headers(client)
            nothing
        catch exc
            exc
        end
        @test err isa ChatClientInvalidAuthError
        @test occursin("AzureIdentity.jl", err.message)
    end

    @testset "Azure embeddings accept AzureIdentity credentials" begin
        received_headers = Channel{Dict{String, String}}(1)
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany=true)
        server = HTTP.serve!(listener; verbose=false) do request::HTTP.Request
            header_dict = Dict(
                lowercase(String(key)) => String(value)
                for (key, value) in request.headers
            )
            put!(received_headers, header_dict)
            return HTTP.Response(
                200,
                ["Content-Type" => "application/json"],
                JSON3.write(
                    Dict(
                        "data" => [
                            Dict("index" => 0, "embedding" => [0.25, 0.5, 0.75]),
                        ],
                    ),
                ),
            )
        end

        try
            client = AzureOpenAIChatClient(
                model = "text-embedding-3-small",
                endpoint = "http://127.0.0.1:$(HTTP.port(server))",
                credential = MockAzureCredential(token="embed-token"),
            )
            embeddings = get_embeddings(client, ["hello"])
            @test embeddings == [Float64[0.25, 0.5, 0.75]]

            headers = take!(received_headers)
            @test headers["authorization"] == "Bearer embed-token"
            @test !haskey(headers, "api-key")
        finally
            close(server)
        end
    end
end
