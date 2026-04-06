@testset "BedrockEmbeddingClient request and response" begin
    requests = Channel{Tuple{String, Dict{String, String}, Dict{String, Any}}}(2)
    listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
    server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
        headers = Dict(lowercase(String(key)) => String(value) for (key, value) in request.headers)
        body = JSON3.read(String(request.body), Dict{String, Any})
        put!(requests, (String(request.target), headers, body))

        embedding = body["inputText"] == "hello" ? [0.1, 0.2, 0.3] : [0.4, 0.5, 0.6]
        return HTTP.Response(
            200,
            ["Content-Type" => "application/json"],
            JSON3.write(Dict("embedding" => embedding, "inputTextTokenCount" => 3)),
        )
    end

    try
        client = BedrockEmbeddingClient(
            model = "amazon.titan-embed-text-v2:0",
            endpoint = "http://127.0.0.1:$(HTTP.port(server))",
            access_key_id = "embed-access",
            secret_access_key = "embed-secret",
            default_headers = Dict("x-embed" => "1"),
            options = Dict{String, Any}("dimensions" => 1024, "normalize" => true),
        )
        embeddings = get_embeddings(client, ["hello", "world"])

        request_one = take!(requests)
        request_two = take!(requests)
        request_bodies = [request_one[3], request_two[3]]
        request_headers = [request_one[2], request_two[2]]
        request_targets = [request_one[1], request_two[1]]

        @test all(target -> target == "/model/amazon.titan-embed-text-v2%3A0/invoke", request_targets)
        @test all(headers -> haskey(headers, "authorization"), request_headers)
        @test all(headers -> headers["x-embed"] == "1", request_headers)
        @test sort([body["inputText"] for body in request_bodies]) == ["hello", "world"]
        @test all(body -> body["dimensions"] == 1024, request_bodies)
        @test all(body -> body["normalize"] == true, request_bodies)
        @test embeddings == [
            Float64[0.1, 0.2, 0.3],
            Float64[0.4, 0.5, 0.6],
        ]
    finally
        close(server)
    end
end

@testset "BedrockEmbeddingClient env fallback" begin
    saved = [(name, get(ENV, name, nothing)) for name in ("BEDROCK_EMBEDDING_MODEL", "BEDROCK_REGION")]

    try
        ENV["BEDROCK_EMBEDDING_MODEL"] = "amazon.titan-embed-text-v2:0"
        ENV["BEDROCK_REGION"] = "ap-southeast-2"

        client = BedrockEmbeddingClient(access_key_id = "env-access", secret_access_key = "env-secret")
        @test Bedrock._resolve_embedding_model(client) == "amazon.titan-embed-text-v2:0"
        @test Bedrock._resolve_bedrock_region(client) == "ap-southeast-2"
        @test sprint(show, client) == "BedrockEmbeddingClient(\"amazon.titan-embed-text-v2:0\")"
    finally
        _restore_env!(saved)
    end
end
