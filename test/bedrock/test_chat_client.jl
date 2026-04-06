function _bedrock_test_client(; kwargs...)
    return BedrockChatClient(
        model = "anthropic.claude-3-haiku-20240307-v1:0",
        endpoint = "http://example.test",
        access_key_id = "test-access",
        secret_access_key = "test-secret",
        ; kwargs...,
    )
end

function _weather_tool()
    return FunctionTool(
        name = "get_weather",
        description = "Get the weather for a location",
        func = identity,
        parameters = Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "location" => Dict{String, Any}("type" => "string"),
            ),
            "required" => ["location"],
        ),
    )
end

@testset "BedrockChatClient request building" begin
    client = _bedrock_test_client(options = Dict{String, Any}("requestMetadata" => Dict("source" => "client-default")))
    tool = _weather_tool()
    messages = [
        Message(:system, "Be helpful."),
        Message(:user, "How is the weather?"),
        Message(:assistant, [function_call_content("call-1", "get_weather", "{\"location\":\"SEA\"}")]),
        Message(:tool, [function_result_content("call-1", Dict("answer" => "72F"))]),
    ]

    body = Bedrock._build_converse_request(
        client,
        messages,
        ChatOptions(
            tools = [tool],
            tool_choice = "get_weather",
            temperature = 0.2,
            max_tokens = 64,
            additional = Dict{String, Any}("guardrailConfig" => Dict("trace" => "enabled")),
        ),
    )

    @test body["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    @test body["system"][1]["text"] == "Be helpful."
    @test body["messages"][1]["role"] == "user"
    @test body["messages"][2]["role"] == "assistant"
    @test body["messages"][3]["role"] == "user"
    @test body["inferenceConfig"]["temperature"] == 0.2
    @test body["inferenceConfig"]["maxTokens"] == 64
    @test body["toolConfig"]["tools"][1]["toolSpec"]["name"] == "get_weather"
    @test body["toolConfig"]["toolChoice"] == Dict("tool" => Dict("name" => "get_weather"))
    @test body["messages"][2]["content"][1]["toolUse"]["input"]["location"] == "SEA"
    @test body["messages"][3]["content"][1]["toolResult"]["content"][1]["json"]["answer"] == "72F"
    @test body["requestMetadata"]["source"] == "client-default"
    @test body["guardrailConfig"]["trace"] == "enabled"
end

@testset "BedrockChatClient omits tool config for tool_choice none" begin
    client = _bedrock_test_client()
    tool = _weather_tool()
    body = Bedrock._build_converse_request(
        client,
        [Message(:user, "Hi")],
        ChatOptions(tools = [tool], tool_choice = "none"),
    )
    @test !haskey(body, "toolConfig")
end

@testset "BedrockChatClient parses tool use responses" begin
    client = _bedrock_test_client()
    response = Bedrock._parse_converse_response(
        Dict{String, Any}(
            "modelId" => "anthropic.claude-3-haiku-20240307-v1:0",
            "responseId" => "resp-tool",
            "usage" => Dict("inputTokens" => 3, "outputTokens" => 4, "totalTokens" => 7),
            "output" => Dict{String, Any}(
                "completionReason" => "tool_use",
                "message" => Dict{String, Any}(
                    "id" => "msg-1",
                    "content" => Any[
                        Dict("toolUse" => Dict("toolUseId" => "call-1", "name" => "get_weather", "input" => Dict("location" => "NYC"))),
                        Dict("text" => "Calling tool"),
                    ],
                ),
            ),
        ),
        client,
    )

    @test response.response_id == "resp-tool"
    @test response.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
    @test response.finish_reason == TOOL_CALLS
    @test response.usage_details.input_tokens == 3
    @test is_function_call(response.messages[1].contents[1])
    @test parse_arguments(response.messages[1].contents[1])["location"] == "NYC"
    @test get_text(response.messages[1].contents[2]) == "Calling tool"
end

@testset "BedrockChatClient parses tool result responses" begin
    client = _bedrock_test_client()
    response = Bedrock._parse_converse_response(
        Dict{String, Any}(
            "modelId" => "anthropic.claude-3-haiku-20240307-v1:0",
            "output" => Dict{String, Any}(
                "completionReason" => "end_turn",
                "message" => Dict{String, Any}(
                    "content" => Any[
                        Dict("toolResult" => Dict("toolUseId" => "call-1", "status" => "success", "content" => [Dict("json" => Dict("answer" => 42))])),
                    ],
                ),
            ),
        ),
        client,
    )

    @test response.finish_reason == STOP
    @test is_function_result(response.messages[1].contents[1])
    @test response.messages[1].contents[1].result == Dict("answer" => 42)
end

@testset "BedrockChatClient live request/response" begin
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
                    "responseId" => "resp-live",
                    "modelId" => "anthropic.claude-3-haiku-20240307-v1:0",
                    "usage" => Dict("inputTokens" => 2, "outputTokens" => 3, "totalTokens" => 5),
                    "output" => Dict(
                        "completionReason" => "end_turn",
                        "message" => Dict(
                            "content" => [Dict("text" => "Hello from Bedrock")],
                        ),
                    ),
                ),
            ),
        )
    end

    try
        client = BedrockChatClient(
            model = "anthropic.claude-3-haiku-20240307-v1:0",
            endpoint = "http://127.0.0.1:$(HTTP.port(server))",
            access_key_id = "live-access",
            secret_access_key = "live-secret",
            session_token = "live-session",
            default_headers = Dict("x-test" => "1"),
            options = Dict{String, Any}("requestMetadata" => Dict("trace" => "true")),
        )
        response = get_response(client, [Message(:user, "Hi")], ChatOptions(temperature = 0.3, max_tokens = 22))
        target, headers, body = take!(received)

        @test target == "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
        @test haskey(headers, "authorization")
        @test headers["x-amz-security-token"] == "live-session"
        @test headers["x-test"] == "1"
        @test body["inferenceConfig"]["temperature"] == 0.3
        @test body["inferenceConfig"]["maxTokens"] == 22
        @test body["requestMetadata"]["trace"] == "true"
        @test response.response_id == "resp-live"
        @test response.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
        @test response.finish_reason == STOP
        @test response.usage_details.total_tokens == 5
        @test response.text == "Hello from Bedrock"
    finally
        close(server)
    end
end

@testset "BedrockChatClient streaming" begin
    listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
    server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
        @test String(request.target) == "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
        return HTTP.Response(
            200,
            ["Content-Type" => "application/json"],
            JSON3.write(
                Dict(
                    "responseId" => "resp-stream",
                    "modelId" => "anthropic.claude-3-haiku-20240307-v1:0",
                    "usage" => Dict("inputTokens" => 1, "outputTokens" => 2, "totalTokens" => 3),
                    "output" => Dict(
                        "completionReason" => "end_turn",
                        "message" => Dict(
                            "content" => [Dict("text" => "Hello stream")],
                        ),
                    ),
                ),
            ),
        )
    end

    try
        client = BedrockChatClient(
            model = "anthropic.claude-3-haiku-20240307-v1:0",
            endpoint = "http://127.0.0.1:$(HTTP.port(server))",
            access_key_id = "stream-access",
            secret_access_key = "stream-secret",
        )
        updates = collect(get_response_streaming(client, [Message(:user, "Hi")], ChatOptions()))
        @test length(updates) == 1
        @test updates[1].role == :assistant
        @test updates[1].finish_reason == STOP

        response = ChatResponse(updates)
        @test response.response_id == "resp-stream"
        @test response.finish_reason == STOP
        @test response.text == "Hello stream"
    finally
        close(server)
    end
end

@testset "BedrockChatClient agent integration" begin
    received = Channel{Dict{String, Any}}(1)
    listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
    server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
        body = JSON3.read(String(request.body), Dict{String, Any})
        put!(received, body)
        return HTTP.Response(
            200,
            ["Content-Type" => "application/json"],
            JSON3.write(
                Dict(
                    "responseId" => "resp-agent",
                    "modelId" => "anthropic.claude-3-haiku-20240307-v1:0",
                    "output" => Dict(
                        "completionReason" => "end_turn",
                        "message" => Dict("content" => [Dict("text" => "Agent hello")]),
                    ),
                ),
            ),
        )
    end

    try
        client = BedrockChatClient(
            model = "anthropic.claude-3-haiku-20240307-v1:0",
            endpoint = "http://127.0.0.1:$(HTTP.port(server))",
            access_key_id = "agent-access",
            secret_access_key = "agent-secret",
        )
        agent = Agent(name = "BedrockBot", instructions = "Be helpful.", client = client)
        response = run_agent(agent, "Hi")
        body = take!(received)

        @test body["system"][1]["text"] == "Be helpful."
        @test response.text == "Agent hello"
    finally
        close(server)
    end
end
