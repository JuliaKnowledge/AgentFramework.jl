using Test
using AgentFramework

@testset "Tool Lifecycle" begin
    @testset "Declaration-only tools" begin
        tool = FunctionTool(name="search", description="Search the web",
            parameters=Dict{String,Any}("type" => "object"))
        @test is_declaration_only(tool)
        @test tool.func === nothing
        @test_throws ToolExecutionError invoke_tool(tool, "{}")
    end

    @testset "Tool with function" begin
        tool = FunctionTool(name="add", description="Add two numbers",
            func=(a::Int, b::Int) -> a + b,
            parameters=Dict{String,Any}("type" => "object"))
        @test !is_declaration_only(tool)
    end

    @testset "Max invocations limit" begin
        call_count = Ref(0)
        tool = FunctionTool(
            name="limited",
            description="Limited tool",
            func=() -> begin call_count[] += 1; "ok" end,
            parameters=Dict{String,Any}("type" => "object"),
            max_invocations=3,
        )
        @test tool.invocation_count == 0

        # First 3 calls succeed
        for i in 1:3
            result = invoke_tool(tool, "{}")
            @test result == "ok"
            @test tool.invocation_count == i
        end

        # 4th call should throw
        @test_throws ToolExecutionError invoke_tool(tool, "{}")
        @test call_count[] == 3  # function was NOT called a 4th time
    end

    @testset "Exception counting" begin
        tool = FunctionTool(
            name="flaky",
            description="Flaky tool",
            func=() -> error("boom"),
            parameters=Dict{String,Any}("type" => "object"),
            max_invocation_exceptions=2,
        )

        # First exception is tolerated (rethrown)
        @test_throws ErrorException invoke_tool(tool, "{}")
        @test tool.invocation_exception_count == 1

        # Second exception hits the limit and throws ToolExecutionError
        @test_throws ToolExecutionError invoke_tool(tool, "{}")
        @test tool.invocation_exception_count == 2
    end

    @testset "Result parser" begin
        tool = FunctionTool(
            name="json_tool",
            description="Returns JSON",
            func=() -> """{"value": 42}""",
            parameters=Dict{String,Any}("type" => "object"),
            result_parser=(r) -> "Parsed: $r",
        )
        result = invoke_tool(tool, "{}")
        @test result == """Parsed: {"value": 42}"""
    end

    @testset "Reset invocation count" begin
        tool = FunctionTool(
            name="resettable",
            description="Resettable tool",
            func=() -> "ok",
            parameters=Dict{String,Any}("type" => "object"),
            max_invocations=2,
        )
        invoke_tool(tool, "{}")
        invoke_tool(tool, "{}")
        @test tool.invocation_count == 2

        reset_invocation_count!(tool)
        @test tool.invocation_count == 0
        @test tool.invocation_exception_count == 0

        # Can invoke again after reset
        result = invoke_tool(tool, "{}")
        @test result == "ok"
        @test tool.invocation_count == 1
    end

    @testset "Kind and additional properties" begin
        tool = FunctionTool(
            name="custom_tool",
            description="Custom tool",
            parameters=Dict{String,Any}("type" => "object"),
            kind="custom",
            additional_properties=Dict{String,Any}("source" => "mcp"),
        )
        @test tool.kind == "custom"
        @test tool.additional_properties["source"] == "mcp"
    end

    @testset "@tool macro still works with mutable struct" begin
        tool = FunctionTool(
            name="manual_greet",
            description="Greet by name",
            func=(x::String) -> string("Hello, ", x, "!"),
            parameters=Dict{String,Any}(
                "type" => "object",
                "properties" => Dict{String,Any}(
                    "x" => Dict{String,Any}("type" => "string")
                ),
                "required" => ["x"],
            ),
        )
        @test tool isa FunctionTool
        @test !is_declaration_only(tool)
        result = invoke_tool(tool, """{"x": "Julia"}""")
        @test result == "Hello, Julia!"
        # Verify mutability works
        tool.invocation_count = 0
        @test tool.invocation_count == 0
    end
end

@testset "Agent as Tool" begin
    # Create a mock client for the inner agent
    mutable struct AsToolMockClient <: AgentFramework.AbstractChatClient
        response_text::String
    end

    function AgentFramework.get_response(client::AsToolMockClient, messages::Vector{Message}, options=nothing)
        msg = Message(role=:assistant, contents=[text_content(client.response_text)])
        ChatResponse(messages=[msg])
    end

    @testset "Basic as_tool" begin
        inner_agent = Agent(
            name="Helper Agent",
            client=AsToolMockClient("I found the answer: 42"),
            instructions="You are a helpful assistant",
        )
        tool = as_tool(inner_agent)
        @test tool isa FunctionTool
        @test tool.name == "helper_agent"
        @test occursin("Helper Agent", tool.description)
        @test haskey(tool.parameters, "properties")
        @test haskey(tool.parameters["properties"], "input")
    end

    @testset "as_tool with custom description" begin
        inner_agent = Agent(
            name="Researcher",
            client=AsToolMockClient("Research result"),
            instructions="Research topics",
        )
        tool = as_tool(inner_agent; description="Performs research on any topic")
        @test tool.description == "Performs research on any topic"
    end

    @testset "as_tool name normalization" begin
        inner_agent = Agent(
            name="My Cool Agent 2.0!",
            client=AsToolMockClient("result"),
            instructions="test",
        )
        tool = as_tool(inner_agent)
        @test !occursin(" ", tool.name)
        @test !occursin(".", tool.name)
        @test !occursin("!", tool.name)
    end
end

@testset "Middleware Termination" begin
    @testset "MiddlewareTermination struct" begin
        mt = MiddlewareTermination("cached_result"; message="cache hit")
        @test mt.result == "cached_result"
        @test mt.message == "cache hit"
    end

    @testset "terminate_pipeline throws" begin
        @test_throws MiddlewareTermination terminate_pipeline("result")
    end

    @testset "Agent middleware termination" begin
        cache_mw = (ctx, next) -> begin
            terminate_pipeline("cached"; message="found in cache")
        end

        ctx = AgentContext()
        handler = (_) -> "from_handler"
        result = apply_agent_middleware([cache_mw], ctx, handler)
        @test result == "cached"
    end

    @testset "Chat middleware termination" begin
        block_mw = (ctx, next) -> begin
            terminate_pipeline(nothing; message="blocked")
        end

        ctx = ChatContext()
        handler = (_) -> "chat_result"
        result = apply_chat_middleware([block_mw], ctx, handler)
        @test result === nothing
    end

    @testset "Function middleware termination" begin
        override_mw = (ctx, next) -> begin
            terminate_pipeline("override_result")
        end

        ctx = FunctionInvocationContext()
        handler = (_) -> "original"
        result = apply_function_middleware([override_mw], ctx, handler)
        @test result == "override_result"
    end

    @testset "Termination in second middleware" begin
        first_mw = (ctx, next) -> begin
            result = next(ctx)
            return "wrapped: $result"
        end
        second_mw = (ctx, next) -> begin
            terminate_pipeline("short_circuited")
        end

        ctx = AgentContext()
        handler = (_) -> "handler_result"
        result = apply_agent_middleware([first_mw, second_mw], ctx, handler)
        @test result == "short_circuited"
    end

    @testset "Non-termination exceptions propagate" begin
        error_mw = (ctx, next) -> error("real error")
        ctx = AgentContext()
        handler = (_) -> "result"
        @test_throws ErrorException apply_agent_middleware([error_mw], ctx, handler)
    end

    @testset "showerror" begin
        mt = MiddlewareTermination("r"; message="test msg")
        buf = IOBuffer()
        showerror(buf, mt)
        @test occursin("test msg", String(take!(buf)))
    end
end

@testset "Tokenizer" begin
    @testset "CharacterEstimatorTokenizer" begin
        tok = CharacterEstimatorTokenizer()
        @test tok.chars_per_token == 4.0

        @test count_tokens(tok, "Hello, world!") == ceil(Int, 13 / 4.0)
        @test count_tokens(tok, "") == 1  # min 1
        @test count_tokens(tok, "ab") == 1
        @test count_tokens(tok, "abcdefgh") == 2
    end

    @testset "Custom chars_per_token" begin
        tok = CharacterEstimatorTokenizer(chars_per_token=2.0)
        @test count_tokens(tok, "Hello") == 3  # ceil(5/2)
    end

    @testset "WordEstimatorTokenizer" begin
        tok = WordEstimatorTokenizer()
        @test count_tokens(tok, "Hello world") == ceil(Int, 2 * 1.3)  # 3
        @test count_tokens(tok, "a") == ceil(Int, 1 * 1.3)  # 2
    end

    @testset "count_message_tokens" begin
        tok = CharacterEstimatorTokenizer()
        msgs = [
            Message(role=:user, contents=[text_content("Hello, how are you?")]),
            Message(role=:assistant, contents=[text_content("I'm fine!")]),
        ]
        total = count_message_tokens(tok, msgs)
        # 4 overhead per msg + tokens for text
        @test total > 0
        @test total == (4 + ceil(Int, 19/4.0)) + (4 + ceil(Int, 9/4.0))
    end

    @testset "count_message_tokens with tool calls" begin
        tok = CharacterEstimatorTokenizer()
        msgs = [
            Message(role=:assistant, contents=[
                function_call_content("call_1", "search", """{"q": "julia"}""")
            ]),
        ]
        total = count_message_tokens(tok, msgs)
        @test total > 4  # at least overhead + name + args tokens
    end
end

@testset "Message Group Annotations" begin
    @testset "group_messages by role" begin
        msgs = [
            Message(role=:system, contents=[text_content("sys")]),
            Message(role=:user, contents=[text_content("hi")]),
            Message(role=:user, contents=[text_content("there")]),
            Message(role=:assistant, contents=[text_content("hello")]),
        ]
        groups = group_messages(msgs; by=:role)
        @test length(groups) == 3
        @test groups[1].label == "system"
        @test groups[1].start_index == 1
        @test groups[1].end_index == 1
        @test groups[2].label == "user"
        @test groups[2].start_index == 2
        @test groups[2].end_index == 3
        @test groups[3].label == "assistant"
    end

    @testset "group_messages empty" begin
        groups = group_messages(Message[]; by=:role)
        @test isempty(groups)
    end

    @testset "group_messages by tool_calls" begin
        msgs = [
            Message(role=:user, contents=[text_content("search for X")]),
            Message(role=:assistant, contents=[
                function_call_content("c1", "search", """{"q":"X"}""")
            ]),
            Message(role=:tool, contents=[function_result_content("c1", "found X")]),
            Message(role=:assistant, contents=[text_content("Here's the result")]),
        ]
        groups = group_messages(msgs; by=:tool_calls)
        @test length(groups) == 3
        @test groups[1].label == "user"
        @test groups[2].label == "tool_calls"
        @test groups[2].start_index == 2
        @test groups[2].end_index == 3
        @test groups[3].label == "assistant"
    end

    @testset "group_messages by turns" begin
        msgs = [
            Message(role=:system, contents=[text_content("sys")]),
            Message(role=:user, contents=[text_content("hi")]),
            Message(role=:assistant, contents=[text_content("hello")]),
            Message(role=:user, contents=[text_content("bye")]),
            Message(role=:assistant, contents=[text_content("goodbye")]),
        ]
        groups = group_messages(msgs; by=:turns)
        @test length(groups) == 3  # system + 2 turns
        @test groups[1].label == "system"
        @test groups[2].label == "turn_1"
        @test groups[2].start_index == 2
        @test groups[2].end_index == 3
        @test groups[3].label == "turn_2"
        @test groups[3].start_index == 4
        @test groups[3].end_index == 5
    end

    @testset "MessageGroup length" begin
        g = MessageGroup(label="test", start_index=3, end_index=7)
        @test length(g) == 5
    end

    @testset "annotate_message_groups" begin
        tok = CharacterEstimatorTokenizer()
        msgs = [
            Message(role=:user, contents=[text_content("Hello")]),
            Message(role=:assistant, contents=[text_content("Hi there")]),
        ]
        groups = annotate_message_groups(msgs, tok)
        @test length(groups) >= 1
        @test haskey(groups[1].metadata, "token_count")
        @test groups[1].metadata["token_count"] > 0
    end

    @testset "Invalid grouping strategy" begin
        msgs = [Message(role=:user, contents=[text_content("hi")])]
        @test_throws ErrorException group_messages(msgs; by=:invalid)
    end
end

@testset "detect_media_type_from_base64" begin
    @testset "PNG detection" begin
        @test detect_media_type_from_base64("iVBORw0KGgoAAAANSU") == "image/png"
    end

    @testset "JPEG detection" begin
        @test detect_media_type_from_base64("/9j/4AAQSkZJRg") == "image/jpeg"
    end

    @testset "GIF detection" begin
        @test detect_media_type_from_base64("R0lGODlh") == "image/gif"
    end

    @testset "PDF detection" begin
        @test detect_media_type_from_base64("JVBERi0xLjQ=") == "application/pdf"
    end

    @testset "MP3 detection" begin
        @test detect_media_type_from_base64("SUQzBAAAAAAAI1RTU0UAAA") == "audio/mpeg"
    end

    @testset "OGG detection" begin
        @test detect_media_type_from_base64("T2dnUwACAAAAAAA") == "audio/ogg"
    end

    @testset "WebP detection" begin
        @test detect_media_type_from_base64("UklGRiAAAABXRUJQV") == "image/webp"
    end

    @testset "ZIP detection" begin
        @test detect_media_type_from_base64("UEsDBBQAAAAI") == "application/zip"
    end

    @testset "Unknown data" begin
        @test detect_media_type_from_base64("dW5rbm93bg==") === nothing
    end

    @testset "Empty string" begin
        @test detect_media_type_from_base64("") === nothing
    end
end
