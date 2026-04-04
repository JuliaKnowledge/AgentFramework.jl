if !@isdefined(AgentFramework)
    using AgentFramework
    using Test
end

@testset "Anthropic Provider" begin

    # ── AnthropicChatClient Construction ─────────────────────────────────────

    @testset "AnthropicChatClient defaults" begin
        client = AnthropicChatClient(api_key="sk-ant-test")
        @test client.model == "claude-sonnet-4-20250514"
        @test client.base_url == "https://api.anthropic.com"
        @test client.api_key == "sk-ant-test"
        @test client.api_version == "2023-06-01"
        @test client.read_timeout == 120
        @test isempty(client.options)
    end

    @testset "AnthropicChatClient custom fields" begin
        client = AnthropicChatClient(
            model="claude-haiku-3-20240307",
            api_key="sk-custom",
            base_url="http://localhost:8080",
            api_version="2024-01-01",
            read_timeout=60,
            options=Dict{String, Any}("metadata" => Dict("user_id" => "abc")),
        )
        @test client.model == "claude-haiku-3-20240307"
        @test client.base_url == "http://localhost:8080"
        @test client.api_version == "2024-01-01"
        @test client.read_timeout == 60
        @test client.options["metadata"] isa Dict
    end

    @testset "AnthropicChatClient show method" begin
        client = AnthropicChatClient(api_key="sk-test")
        @test sprint(show, client) == "AnthropicChatClient(\"claude-sonnet-4-20250514\")"
        client2 = AnthropicChatClient(model="claude-haiku-3-20240307", api_key="sk-test")
        @test sprint(show, client2) == "AnthropicChatClient(\"claude-haiku-3-20240307\")"
    end

    # ── API Key Resolution ───────────────────────────────────────────────────

    @testset "AnthropicChatClient env var fallback" begin
        old_key = get(ENV, "ANTHROPIC_API_KEY", nothing)
        try
            ENV["ANTHROPIC_API_KEY"] = "sk-ant-from-env"
            client = AnthropicChatClient()
            @test client.api_key == ""
            key = AgentFramework._resolve_api_key(client)
            @test key == "sk-ant-from-env"
        finally
            if old_key === nothing
                delete!(ENV, "ANTHROPIC_API_KEY")
            else
                ENV["ANTHROPIC_API_KEY"] = old_key
            end
        end
    end

    @testset "AnthropicChatClient explicit key overrides env" begin
        old_key = get(ENV, "ANTHROPIC_API_KEY", nothing)
        try
            ENV["ANTHROPIC_API_KEY"] = "sk-ant-from-env"
            client = AnthropicChatClient(api_key="sk-ant-explicit")
            key = AgentFramework._resolve_api_key(client)
            @test key == "sk-ant-explicit"
        finally
            if old_key === nothing
                delete!(ENV, "ANTHROPIC_API_KEY")
            else
                ENV["ANTHROPIC_API_KEY"] = old_key
            end
        end
    end

    @testset "AnthropicChatClient missing key throws" begin
        old_key = get(ENV, "ANTHROPIC_API_KEY", nothing)
        try
            delete!(ENV, "ANTHROPIC_API_KEY")
            client = AnthropicChatClient()
            @test_throws ChatClientError AgentFramework._resolve_api_key(client)
        finally
            if old_key !== nothing
                ENV["ANTHROPIC_API_KEY"] = old_key
            end
        end
    end

    # ── System Message Splitting ─────────────────────────────────────────────

    @testset "_split_system_anthropic extracts system messages" begin
        msgs = [
            Message(:system, "You are helpful."),
            Message(:user, "Hello"),
            Message(:system, "Be concise."),
            Message(:assistant, "Hi!"),
        ]
        sys, remaining = AgentFramework._split_system_anthropic(msgs)
        @test sys == "You are helpful.\n\nBe concise."
        @test length(remaining) == 2
        @test remaining[1].role == :user
        @test remaining[2].role == :assistant
    end

    @testset "_split_system_anthropic handles no system messages" begin
        msgs = [
            Message(:user, "Hello"),
            Message(:assistant, "Hi!"),
        ]
        sys, remaining = AgentFramework._split_system_anthropic(msgs)
        @test sys === nothing
        @test length(remaining) == 2
    end

    @testset "_split_system_anthropic handles all system messages" begin
        msgs = [Message(:system, "Only system here.")]
        sys, remaining = AgentFramework._split_system_anthropic(msgs)
        @test sys == "Only system here."
        @test isempty(remaining)
    end

    # ── Message Conversion ───────────────────────────────────────────────────

    @testset "user text → Anthropic format" begin
        msgs = [Message(:user, "Hello")]
        result = AgentFramework._messages_to_anthropic(msgs)
        @test length(result) == 1
        @test result[1]["role"] == "user"
        @test result[1]["content"][1]["type"] == "text"
        @test result[1]["content"][1]["text"] == "Hello"
    end

    @testset "assistant text → Anthropic format" begin
        msgs = [Message(:assistant, "Hi there!")]
        result = AgentFramework._messages_to_anthropic(msgs)
        @test length(result) == 1
        @test result[1]["role"] == "assistant"
        @test result[1]["content"][1]["type"] == "text"
        @test result[1]["content"][1]["text"] == "Hi there!"
    end

    @testset "tool calls → tool_use blocks" begin
        fc = function_call_content("toolu_123", "get_weather", "{\"city\":\"London\"}")
        msgs = [Message(:assistant, [fc])]
        result = AgentFramework._messages_to_anthropic(msgs)
        @test length(result) == 1
        @test result[1]["role"] == "assistant"
        block = result[1]["content"][1]
        @test block["type"] == "tool_use"
        @test block["id"] == "toolu_123"
        @test block["name"] == "get_weather"
        @test block["input"]["city"] == "London"
    end

    @testset "tool results → tool_result in user message" begin
        fr = function_result_content("toolu_123", "Sunny, 22°C")
        msgs = [Message(:tool, [fr])]
        result = AgentFramework._messages_to_anthropic(msgs)
        @test length(result) == 1
        @test result[1]["role"] == "user"  # :tool maps to user
        block = result[1]["content"][1]
        @test block["type"] == "tool_result"
        @test block["tool_use_id"] == "toolu_123"
        @test block["content"] == "Sunny, 22°C"
    end

    @testset "merges adjacent same-role messages" begin
        msgs = [
            Message(:user, "Part 1"),
            Message(:user, "Part 2"),
            Message(:assistant, "Response"),
        ]
        result = AgentFramework._messages_to_anthropic(msgs)
        @test length(result) == 2
        @test result[1]["role"] == "user"
        @test length(result[1]["content"]) == 2
        @test result[1]["content"][1]["text"] == "Part 1"
        @test result[1]["content"][2]["text"] == "Part 2"
        @test result[2]["role"] == "assistant"
    end

    @testset "tool message after assistant merges into user" begin
        # assistant with tool_use, then tool with result → should merge tool results into user
        fc = function_call_content("toolu_1", "search", "{\"q\":\"julia\"}")
        fr = function_result_content("toolu_1", "Found 42 results")
        msgs = [
            Message(:assistant, [fc]),
            Message(:tool, [fr]),
        ]
        result = AgentFramework._messages_to_anthropic(msgs)
        @test length(result) == 2
        @test result[1]["role"] == "assistant"
        @test result[2]["role"] == "user"
        @test result[2]["content"][1]["type"] == "tool_result"
    end

    @testset "mixed text and tool_use in assistant message" begin
        contents = [
            text_content("Let me check the weather."),
            function_call_content("toolu_abc", "get_weather", "{\"city\":\"Paris\"}")
        ]
        msgs = [Message(:assistant, contents)]
        result = AgentFramework._messages_to_anthropic(msgs)
        @test length(result) == 1
        @test length(result[1]["content"]) == 2
        @test result[1]["content"][1]["type"] == "text"
        @test result[1]["content"][2]["type"] == "tool_use"
    end

    # ── Tool Conversion ──────────────────────────────────────────────────────

    @testset "_tools_to_anthropic returns nothing for empty" begin
        @test AgentFramework._tools_to_anthropic(FunctionTool[]) === nothing
    end

    @testset "_tools_to_anthropic converts tools" begin
        tool = FunctionTool(
            name="get_weather",
            description="Get weather for a city",
            func=identity,
            parameters=Dict{String, Any}(
                "type" => "object",
                "properties" => Dict("city" => Dict("type" => "string")),
                "required" => ["city"],
            ),
        )
        result = AgentFramework._tools_to_anthropic([tool])
        @test length(result) == 1
        @test result[1]["name"] == "get_weather"
        @test result[1]["description"] == "Get weather for a city"
        @test result[1]["input_schema"]["type"] == "object"
        @test haskey(result[1]["input_schema"]["properties"], "city")
    end

    # ── Request Body Construction ────────────────────────────────────────────

    @testset "_build_anthropic_request includes system, model, max_tokens" begin
        client = AnthropicChatClient(api_key="sk-test", model="claude-sonnet-4-20250514")
        api_messages = [Dict{String, Any}("role" => "user", "content" => [Dict{String, Any}("type" => "text", "text" => "Hello")])]
        opts = ChatOptions(max_tokens=2048, temperature=0.7)

        body = AgentFramework._build_anthropic_request(client, api_messages, "Be helpful", opts)
        @test body["model"] == "claude-sonnet-4-20250514"
        @test body["max_tokens"] == 2048
        @test body["system"] == "Be helpful"
        @test body["temperature"] == 0.7
        @test body["stream"] == false
        @test length(body["messages"]) == 1
    end

    @testset "_build_anthropic_request no system when nil" begin
        client = AnthropicChatClient(api_key="sk-test")
        api_messages = [Dict{String, Any}("role" => "user", "content" => [Dict{String, Any}("type" => "text", "text" => "Hi")])]
        opts = ChatOptions()

        body = AgentFramework._build_anthropic_request(client, api_messages, nothing, opts)
        @test !haskey(body, "system")
    end

    @testset "_build_anthropic_request includes tools" begin
        client = AnthropicChatClient(api_key="sk-test")
        api_messages = [Dict{String, Any}("role" => "user", "content" => [Dict{String, Any}("type" => "text", "text" => "Hi")])]
        tool = FunctionTool(
            name="calc",
            description="Calculate",
            func=identity,
            parameters=Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}()),
        )
        opts = ChatOptions(tools=[tool])

        body = AgentFramework._build_anthropic_request(client, api_messages, nothing, opts)
        @test haskey(body, "tools")
        @test body["tools"][1]["name"] == "calc"
    end

    @testset "_build_anthropic_request respects model override" begin
        client = AnthropicChatClient(api_key="sk-test", model="claude-sonnet-4-20250514")
        api_messages = [Dict{String, Any}("role" => "user", "content" => [])]
        opts = ChatOptions(model="claude-haiku-3-20240307")

        body = AgentFramework._build_anthropic_request(client, api_messages, nothing, opts)
        @test body["model"] == "claude-haiku-3-20240307"
    end

    @testset "_build_anthropic_request default max_tokens" begin
        client = AnthropicChatClient(api_key="sk-test")
        api_messages = [Dict{String, Any}("role" => "user", "content" => [])]
        opts = ChatOptions()

        body = AgentFramework._build_anthropic_request(client, api_messages, nothing, opts)
        @test body["max_tokens"] == 4096
    end

    @testset "_build_anthropic_request includes client options" begin
        client = AnthropicChatClient(
            api_key="sk-test",
            options=Dict{String, Any}("metadata" => Dict("user_id" => "u1")),
        )
        api_messages = [Dict{String, Any}("role" => "user", "content" => [])]
        opts = ChatOptions()

        body = AgentFramework._build_anthropic_request(client, api_messages, nothing, opts)
        @test body["metadata"]["user_id"] == "u1"
    end

    # ── Response Parsing ─────────────────────────────────────────────────────

    @testset "_parse_anthropic_response for text response" begin
        data = Dict{String, Any}(
            "id" => "msg_test123",
            "type" => "message",
            "role" => "assistant",
            "content" => [
                Dict{String, Any}("type" => "text", "text" => "Hello! How can I help?"),
            ],
            "model" => "claude-sonnet-4-20250514",
            "stop_reason" => "end_turn",
            "usage" => Dict{String, Any}("input_tokens" => 10, "output_tokens" => 15),
        )
        resp = AgentFramework._parse_anthropic_response(data)
        @test length(resp.messages) == 1
        @test resp.messages[1].role == :assistant
        @test get_text(resp) == "Hello! How can I help?"
        @test resp.finish_reason == STOP
        @test resp.model_id == "claude-sonnet-4-20250514"
        @test resp.response_id == "msg_test123"
    end

    @testset "_parse_anthropic_response for tool_use response" begin
        data = Dict{String, Any}(
            "id" => "msg_tool456",
            "type" => "message",
            "role" => "assistant",
            "content" => [
                Dict{String, Any}("type" => "text", "text" => "Let me look that up."),
                Dict{String, Any}(
                    "type" => "tool_use",
                    "id" => "toolu_xyz",
                    "name" => "get_weather",
                    "input" => Dict{String, Any}("city" => "London"),
                ),
            ],
            "model" => "claude-sonnet-4-20250514",
            "stop_reason" => "tool_use",
            "usage" => Dict{String, Any}("input_tokens" => 20, "output_tokens" => 30),
        )
        resp = AgentFramework._parse_anthropic_response(data)
        @test length(resp.messages) == 1
        contents = resp.messages[1].contents
        @test length(contents) == 2
        @test is_text(contents[1])
        @test get_text(contents[1]) == "Let me look that up."
        @test is_function_call(contents[2])
        @test contents[2].call_id == "toolu_xyz"
        @test contents[2].name == "get_weather"
        args = parse_arguments(contents[2])
        @test args["city"] == "London"
        @test resp.finish_reason == TOOL_CALLS
    end

    @testset "_parse_anthropic_response maps stop_reasons correctly" begin
        for (reason, expected) in [
            ("end_turn", STOP),
            ("stop_sequence", STOP),
            ("max_tokens", LENGTH),
            ("tool_use", TOOL_CALLS),
        ]
            data = Dict{String, Any}(
                "role" => "assistant",
                "content" => [Dict{String, Any}("type" => "text", "text" => "test")],
                "stop_reason" => reason,
            )
            resp = AgentFramework._parse_anthropic_response(data)
            @test resp.finish_reason == expected
        end
    end

    @testset "_parse_anthropic_response extracts usage" begin
        data = Dict{String, Any}(
            "role" => "assistant",
            "content" => [Dict{String, Any}("type" => "text", "text" => "hi")],
            "stop_reason" => "end_turn",
            "usage" => Dict{String, Any}("input_tokens" => 100, "output_tokens" => 50),
        )
        resp = AgentFramework._parse_anthropic_response(data)
        @test resp.usage_details !== nothing
        @test resp.usage_details.input_tokens == 100
        @test resp.usage_details.output_tokens == 50
        @test resp.usage_details.total_tokens == 150
    end

    @testset "_parse_anthropic_response handles missing usage" begin
        data = Dict{String, Any}(
            "role" => "assistant",
            "content" => [Dict{String, Any}("type" => "text", "text" => "hi")],
            "stop_reason" => "end_turn",
        )
        resp = AgentFramework._parse_anthropic_response(data)
        @test resp.usage_details === nothing
    end

    # ── Capability Traits ────────────────────────────────────────────────────

    @testset "capability traits registered correctly" begin
        client = AnthropicChatClient(api_key="sk-test")
        @test supports_streaming(client)
        @test supports_tool_calling(client)
        @test supports_structured_output(client)
        @test !supports_embeddings(client)
        @test !supports_image_generation(client)
        @test !supports_web_search(client)
    end

    @testset "list_capabilities returns expected set" begin
        client = AnthropicChatClient(api_key="sk-test")
        caps = list_capabilities(client)
        @test :streaming in caps
        @test :tool_calling in caps
        @test :structured_output in caps
        @test :embeddings ∉ caps
    end

    # ── get_response throws without API key ──────────────────────────────────

    @testset "get_response throws without API key" begin
        old_key = get(ENV, "ANTHROPIC_API_KEY", nothing)
        try
            delete!(ENV, "ANTHROPIC_API_KEY")
            client = AnthropicChatClient()
            msgs = [Message(:user, "Hello")]
            opts = ChatOptions()
            @test_throws ChatClientError get_response(client, msgs, opts)
        finally
            if old_key !== nothing
                ENV["ANTHROPIC_API_KEY"] = old_key
            end
        end
    end

    @testset "get_response_streaming throws without API key" begin
        old_key = get(ENV, "ANTHROPIC_API_KEY", nothing)
        try
            delete!(ENV, "ANTHROPIC_API_KEY")
            client = AnthropicChatClient()
            msgs = [Message(:user, "Hello")]
            opts = ChatOptions()
            @test_throws ChatClientError get_response_streaming(client, msgs, opts)
        finally
            if old_key !== nothing
                ENV["ANTHROPIC_API_KEY"] = old_key
            end
        end
    end

    # ── Header Construction ──────────────────────────────────────────────────

    @testset "Anthropic headers" begin
        client = AnthropicChatClient(api_key="sk-ant-test-key")
        headers = AgentFramework._build_headers(client)
        header_dict = Dict(headers)
        @test header_dict["x-api-key"] == "sk-ant-test-key"
        @test header_dict["anthropic-version"] == "2023-06-01"
        @test header_dict["Content-Type"] == "application/json"
        @test !haskey(header_dict, "Authorization")
    end

    @testset "Anthropic curl headers" begin
        client = AnthropicChatClient(api_key="sk-ant-curl-key", api_version="2024-01-01")
        curl_headers = AgentFramework._build_curl_headers(client)
        @test "x-api-key: sk-ant-curl-key" in curl_headers
        @test "anthropic-version: 2024-01-01" in curl_headers
        @test "Content-Type: application/json" in curl_headers
    end

    # ── Types exported ───────────────────────────────────────────────────────

    @testset "AnthropicChatClient is exported" begin
        @test isdefined(AgentFramework, :AnthropicChatClient)
        @test AnthropicChatClient <: AbstractChatClient
    end
end
