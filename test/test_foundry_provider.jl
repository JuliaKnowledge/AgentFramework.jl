using AgentFramework
using AzureIdentity
using Dates
using HTTP
using JSON3
using Test

Base.@kwdef struct FoundryMockAzureCredential <: AzureIdentity.AbstractAzureCredential
    token::String = "foundry-token"
end

function AzureIdentity.get_token_info(
    credential::FoundryMockAzureCredential,
    scopes::Vararg{String};
    kwargs...,
)
    return AzureIdentity.AzureAccessTokenInfo(
        token = credential.token,
        expires_on = now() + Hour(1),
        scopes = collect(scopes),
    )
end

@testset "Foundry Provider" begin
    @testset "FoundryChatClient env defaults" begin
        old_model = get(ENV, "FOUNDRY_MODEL", nothing)
        old_endpoint = get(ENV, "FOUNDRY_PROJECT_ENDPOINT", nothing)
        try
            ENV["FOUNDRY_MODEL"] = "env-foundry-model"
            ENV["FOUNDRY_PROJECT_ENDPOINT"] = "https://acct.services.ai.azure.com/api/projects/_project"
            client = FoundryChatClient(credential = FoundryMockAzureCredential())
            @test AgentFramework._resolve_foundry_model(client) == "env-foundry-model"
            @test AgentFramework._resolve_foundry_project_endpoint(client) ==
                "https://acct.services.ai.azure.com/api/projects/_project"
            @test client.token_scope == AgentFramework.DEFAULT_FOUNDRY_PROJECT_TOKEN_SCOPE
        finally
            if old_model === nothing
                delete!(ENV, "FOUNDRY_MODEL")
            else
                ENV["FOUNDRY_MODEL"] = old_model
            end
            if old_endpoint === nothing
                delete!(ENV, "FOUNDRY_PROJECT_ENDPOINT")
            else
                ENV["FOUNDRY_PROJECT_ENDPOINT"] = old_endpoint
            end
        end
    end

    @testset "FoundryChatClient validation" begin
        client = FoundryChatClient(project_endpoint = "https://acct.services.ai.azure.com/api/projects/_project")
        @test_throws ChatClientInvalidRequestError AgentFramework._resolve_foundry_model(client)

        client = FoundryChatClient(model = "gpt-5.3", credential = FoundryMockAzureCredential())
        @test_throws ChatClientInvalidRequestError AgentFramework._resolve_foundry_project_endpoint(client)

        client = FoundryChatClient(
            model = "gpt-5.3",
            project_endpoint = "https://acct.services.ai.azure.com/api/projects/_project",
        )
        @test_throws ChatClientInvalidAuthError AgentFramework._build_headers(client)
    end

    @testset "FoundryChatClient headers and URL" begin
        client = FoundryChatClient(
            model = "gpt-5.3",
            project_endpoint = "https://acct.services.ai.azure.com/api/projects/_project/",
            credential = FoundryMockAzureCredential(token = "project-token"),
            default_headers = Dict("x-test" => "1"),
        )
        headers = Dict(AgentFramework._build_headers(client))
        @test headers["Authorization"] == "Bearer project-token"
        @test headers["x-test"] == "1"
        @test AgentFramework._chat_completions_url(client) ==
            "https://acct.services.ai.azure.com/api/projects/_project/openai/v1/chat/completions"
        curl_headers = AgentFramework._build_curl_headers(client)
        @test "Authorization: Bearer project-token" in curl_headers
        @test "x-test: 1" in curl_headers
        @test sprint(show, client) == "FoundryChatClient(\"gpt-5.3\")"
    end

    @testset "FoundryChatClient request/response" begin
        received = Channel{Tuple{String, Dict{String, String}, Dict{String, Any}}}(1)
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            headers = Dict(lowercase(String(key)) => String(value) for (key, value) in request.headers)
            body = JSON3.read(String(request.body), Dict{String, Any})
            put!(received, (String(request.target), headers, body))
            return HTTP.Response(
                200,
                ["Content-Type" => "application/json"],
                JSON3.write(
                    Dict(
                        "id" => "resp-foundry",
                        "model" => "gpt-5.3",
                        "choices" => [
                            Dict(
                                "message" => Dict("role" => "assistant", "content" => "Hello from Foundry"),
                                "finish_reason" => "stop",
                            ),
                        ],
                        "usage" => Dict(
                            "prompt_tokens" => 3,
                            "completion_tokens" => 4,
                            "total_tokens" => 7,
                        ),
                    ),
                ),
            )
        end

        try
            client = FoundryChatClient(
                model = "gpt-5.3",
                project_endpoint = "http://127.0.0.1:$(HTTP.port(server))/api/projects/_project",
                token_provider = () -> "runtime-token",
                options = Dict{String, Any}("seed" => 7),
            )
            response = get_response(
                client,
                [Message(role = :user, contents = [text_content("Hi")])],
                ChatOptions(temperature = 0.2),
            )
            target, headers, body = take!(received)
            @test target == "/api/projects/_project/openai/v1/chat/completions"
            @test headers["authorization"] == "Bearer runtime-token"
            @test body["model"] == "gpt-5.3"
            @test body["temperature"] == 0.2
            @test body["seed"] == 7
            @test response.response_id == "resp-foundry"
            @test response.model_id == "gpt-5.3"
            @test get_text(response.messages[1].contents[1]) == "Hello from Foundry"
        finally
            close(server)
        end
    end

    @testset "FoundryChatClient streaming" begin
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            @test String(request.target) == "/api/projects/_project/openai/v1/chat/completions"
            body = join(
                [
                    "data: " * JSON3.write(
                        Dict(
                            "id" => "resp-stream",
                            "model" => "gpt-5.3",
                            "choices" => [
                                Dict("delta" => Dict("role" => "assistant", "content" => "Hello"), "finish_reason" => nothing),
                            ],
                        ),
                    ),
                    "",
                    "data: " * JSON3.write(
                        Dict(
                            "id" => "resp-stream",
                            "model" => "gpt-5.3",
                            "choices" => [
                                Dict("delta" => Dict("content" => " world"), "finish_reason" => "stop"),
                            ],
                            "usage" => Dict(
                                "prompt_tokens" => 2,
                                "completion_tokens" => 2,
                                "total_tokens" => 4,
                            ),
                        ),
                    ),
                    "",
                    "data: [DONE]",
                    "",
                ],
                "\n",
            )
            return HTTP.Response(200, ["Content-Type" => "text/event-stream"], body)
        end

        try
            client = FoundryChatClient(
                model = "gpt-5.3",
                project_endpoint = "http://127.0.0.1:$(HTTP.port(server))/api/projects/_project",
                token_provider = () -> "stream-token",
            )
            updates = collect(
                get_response_streaming(
                    client,
                    [Message(role = :user, contents = [text_content("Hi")])],
                    ChatOptions(),
                ),
            )
            @test length(updates) == 2
            response = ChatResponse(updates)
            @test get_text(response.messages[1].contents[1]) == "Hello world"
            @test response.finish_reason == STOP
            @test response.response_id == "resp-stream"
        finally
            close(server)
        end
    end

    @testset "FoundryEmbeddingClient headers and URL" begin
        client = FoundryEmbeddingClient(
            model = "text-embedding-3-large",
            endpoint = "https://acct.eastus2.models.ai.azure.com/",
            api_key = "embed-key",
            default_headers = Dict("x-embed" => "1"),
        )
        headers = Dict(AgentFramework._build_headers(client))
        @test headers["api-key"] == "embed-key"
        @test headers["x-embed"] == "1"
        @test AgentFramework._embeddings_url(client) ==
            "https://acct.eastus2.models.ai.azure.com/embeddings?api-version=2024-05-01-preview"
        @test sprint(show, client) == "FoundryEmbeddingClient(\"text-embedding-3-large\")"
    end

    @testset "FoundryEmbeddingClient credential auth and embeddings" begin
        received = Channel{Tuple{String, Dict{String, String}, Dict{String, Any}}}(1)
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            headers = Dict(lowercase(String(key)) => String(value) for (key, value) in request.headers)
            body = JSON3.read(String(request.body), Dict{String, Any})
            put!(received, (String(request.target), headers, body))
            return HTTP.Response(
                200,
                ["Content-Type" => "application/json"],
                JSON3.write(
                    Dict(
                        "data" => [
                            Dict("index" => 1, "embedding" => [0.4, 0.5, 0.6]),
                            Dict("index" => 0, "embedding" => [0.1, 0.2, 0.3]),
                        ],
                    ),
                ),
            )
        end

        try
            client = FoundryEmbeddingClient(
                model = "text-embedding-3-large",
                endpoint = "http://127.0.0.1:$(HTTP.port(server))",
                credential = FoundryMockAzureCredential(token = "embed-token"),
                options = Dict{String, Any}("dimensions" => 1536),
            )
            embeddings = get_embeddings(client, ["hello", "world"])
            target, headers, body = take!(received)
            @test target == "/embeddings?api-version=2024-05-01-preview"
            @test headers["authorization"] == "Bearer embed-token"
            @test body["model"] == "text-embedding-3-large"
            @test body["dimensions"] == 1536
            @test embeddings == [
                Float64[0.1, 0.2, 0.3],
                Float64[0.4, 0.5, 0.6],
            ]
        finally
            close(server)
        end
    end
end
