using AgentFramework
using AgentFramework: TEXT, FUNCTION_CALL
using Test

# Mock chat client for Agent serialization test
mutable struct SerMockChatClient <: AbstractChatClient end
function AgentFramework.get_response(::SerMockChatClient, ::Vector{Message}, ::ChatOptions)::ChatResponse
    ChatResponse()
end

Base.@kwdef struct RegisteredSessionState
    name::String
    count::Int
end

@testset "Serialization" begin

    # ── Content serialization ────────────────────────────────────────────────

    @testset "Content serialize_to_dict" begin
        # Text content
        c = text_content("hello world")
        d = serialize_to_dict(c)
        @test d["_type"] == "Content"
        @test d["type"] == "text"
        @test d["text"] == "hello world"

        # Function call content
        c = function_call_content("call1", "get_weather", """{"location": "London"}""")
        d = serialize_to_dict(c)
        @test d["_type"] == "Content"
        @test d["type"] == "function_call"
        @test d["call_id"] == "call1"
        @test d["name"] == "get_weather"
        @test d["arguments"] == """{"location": "London"}"""

        # Function result content
        c = function_result_content("call1", "Sunny, 22°C")
        d = serialize_to_dict(c)
        @test d["_type"] == "Content"
        @test d["type"] == "function_result"
        @test d["call_id"] == "call1"
        @test d["result"] == "Sunny, 22°C"

        # Data content (base64 string payload)
        c = data_content("aGVsbG8=", "image/png")
        d = serialize_to_dict(c)
        @test d["_type"] == "Content"
        @test d["type"] == "data"
        @test d["text"] == "aGVsbG8="
        @test d["media_type"] == "image/png"
    end

    # ── Content roundtrip ────────────────────────────────────────────────────

    @testset "Content roundtrip" begin
        for (label, c) in [
            ("text", text_content("hello world")),
            ("function_call", function_call_content("c1", "fn", "{}")),
            ("function_result", function_result_content("c1", "ok"; name="fn")),
            ("data", data_content("aGVsbG8=", "image/png")),
            ("uri", uri_content("https://example.com"; media_type="text/html")),
            ("error", error_content("oops"; error_code="E01")),
        ]
            d = serialize_to_dict(c)
            c2 = deserialize_from_dict(d)
            @test c2 isa Content
            @test c2.type == c.type
            @test c2.text == c.text
            @test c2.call_id == c.call_id
            @test c2.name == c.name
            @test c2.arguments == c.arguments
        end
    end

    # ── Message serialization ────────────────────────────────────────────────

    @testset "Message serialize_to_dict" begin
        msg = Message(:user, "What's the weather?")
        d = serialize_to_dict(msg)
        @test d["_type"] == "Message"
        @test d["role"] == "user"
        @test length(d["contents"]) == 1
        @test d["contents"][1]["type"] == "text"
        @test d["contents"][1]["text"] == "What's the weather?"
        @test !haskey(d, "author_name")
        @test !haskey(d, "message_id")
    end

    # ── Message roundtrip ────────────────────────────────────────────────────

    @testset "Message roundtrip" begin
        # Simple message
        msg = Message(:user, "What's the weather?")
        d = serialize_to_dict(msg)
        msg2 = deserialize_from_dict(d)
        @test msg2 isa Message
        @test msg2.role == :user
        @test length(msg2.contents) == 1
        @test get_text(msg2) == "What's the weather?"

        # With optional fields
        msg = Message(
            role = :assistant,
            contents = [text_content("Hello!")],
            author_name = "bot",
            message_id = "msg-123",
            additional_properties = Dict{String, Any}("custom" => "value"),
        )
        d = serialize_to_dict(msg)
        @test d["author_name"] == "bot"
        @test d["message_id"] == "msg-123"
        @test d["additional_properties"]["custom"] == "value"

        msg2 = deserialize_from_dict(d)
        @test msg2 isa Message
        @test msg2.author_name == "bot"
        @test msg2.message_id == "msg-123"
        @test msg2.additional_properties["custom"] == "value"
    end

    # ── ChatOptions serialization ────────────────────────────────────────────

    @testset "ChatOptions serialize_to_dict" begin
        opts = ChatOptions(
            model = "gpt-4",
            temperature = 0.7,
            max_tokens = 100,
        )
        d = serialize_to_dict(opts)
        @test d["_type"] == "ChatOptions"
        @test d["model"] == "gpt-4"
        @test d["temperature"] == 0.7
        @test d["max_tokens"] == 100
        @test !haskey(d, "top_p")
        @test !haskey(d, "stop")
    end

    # ── ChatOptions roundtrip ────────────────────────────────────────────────

    @testset "ChatOptions roundtrip" begin
        opts = ChatOptions(
            model = "gpt-4",
            temperature = 0.7,
            top_p = 0.9,
            max_tokens = 100,
            stop = ["END", "DONE"],
            tool_choice = "auto",
            additional = Dict{String, Any}("seed" => 42),
        )
        d = serialize_to_dict(opts)
        opts2 = deserialize_from_dict(d)
        @test opts2 isa ChatOptions
        @test opts2.model == "gpt-4"
        @test opts2.temperature ≈ 0.7
        @test opts2.top_p ≈ 0.9
        @test opts2.max_tokens == 100
        @test opts2.stop == ["END", "DONE"]
        @test opts2.tool_choice == "auto"
        @test opts2.additional["seed"] == 42
    end

    # ── AgentSession serialization ───────────────────────────────────────────

    @testset "AgentSession serialize_to_dict" begin
        session = AgentSession(
            id = "test-session",
            state = Dict{String, Any}("counter" => 1),
            user_id = "user-123",
            thread_id = "thread-456",
            metadata = Dict{String, Any}("source" => "test"),
        )
        d = serialize_to_dict(session)
        @test d["_type"] == "AgentSession"
        @test d["id"] == "test-session"
        @test d["state"]["counter"] == 1
        @test d["user_id"] == "user-123"
        @test d["thread_id"] == "thread-456"
        @test d["metadata"]["source"] == "test"
    end

    # ── AgentSession roundtrip ───────────────────────────────────────────────

    @testset "AgentSession roundtrip" begin
        session = AgentSession(
            id = "test-session",
            state = Dict{String, Any}("counter" => 1),
            user_id = "user-123",
            thread_id = "thread-456",
            metadata = Dict{String, Any}("env" => "staging"),
        )
        d = serialize_to_dict(session)
        session2 = deserialize_from_dict(d)
        @test session2 isa AgentSession
        @test session2.id == "test-session"
        @test session2.state["counter"] == 1
        @test session2.user_id == "user-123"
        @test session2.thread_id == "thread-456"
        @test session2.metadata["env"] == "staging"
    end

    # ── Agent serialization (partial) ────────────────────────────────────────

    @testset "Agent serialize_to_dict (partial)" begin
        @tool function ser_test_tool(x::String)
            "A test tool"
            return x
        end

        agent = Agent(
            name = "TestAgent",
            description = "A test agent",
            instructions = "Be helpful.",
            client = SerMockChatClient(),
            tools = [ser_test_tool],
            max_tool_iterations = 5,
            options = ChatOptions(model = "gpt-4", temperature = 0.5),
        )

        d = serialize_to_dict(agent)
        @test d["_type"] == "Agent"
        @test d["name"] == "TestAgent"
        @test d["description"] == "A test agent"
        @test d["instructions"] == "Be helpful."
        @test d["max_tool_iterations"] == 5
        @test d["tool_names"] == ["ser_test_tool"]
        @test d["options"]["model"] == "gpt-4"
        @test d["options"]["temperature"] == 0.5
        # Client and middlewares are not serialized
        @test !haskey(d, "client")
        @test !haskey(d, "middlewares")
    end

    # ── serialize_messages / deserialize_messages ────────────────────────────

    @testset "serialize_messages / deserialize_messages roundtrip" begin
        msgs = [
            Message(:user, "Hello"),
            Message(:assistant, "Hi there!"),
            Message(:user, [
                text_content("Follow-up"),
                function_call_content("c1", "search", """{"q":"test"}"""),
            ]),
        ]
        json = serialize_messages(msgs)
        msgs2 = deserialize_messages(json)
        @test length(msgs2) == 3
        @test msgs2[1].role == :user
        @test get_text(msgs2[1]) == "Hello"
        @test msgs2[2].role == :assistant
        @test get_text(msgs2[2]) == "Hi there!"
        @test msgs2[3].role == :user
        @test length(msgs2[3].contents) == 2
        @test msgs2[3].contents[2].type == FUNCTION_CALL
        @test msgs2[3].contents[2].name == "search"
    end

    # ── serialize_to_json / deserialize_from_json ────────────────────────────

    @testset "serialize_to_json / deserialize_from_json roundtrip" begin
        # Message
        msg = Message(:user, "test message")
        json = serialize_to_json(msg)
        msg2 = deserialize_from_json(json)
        @test msg2 isa Message
        @test msg2.role == :user
        @test get_text(msg2) == "test message"

        # ChatOptions
        opts = ChatOptions(model = "gpt-4", temperature = 0.8)
        json = serialize_to_json(opts)
        opts2 = deserialize_from_json(json)
        @test opts2 isa ChatOptions
        @test opts2.model == "gpt-4"
        @test opts2.temperature ≈ 0.8

        # AgentSession
        session = AgentSession(id = "s1", user_id = "u1")
        json = serialize_to_json(session)
        session2 = deserialize_from_json(json)
        @test session2 isa AgentSession
        @test session2.id == "s1"
        @test session2.user_id == "u1"
    end

    # ── Type registry ────────────────────────────────────────────────────────

    @testset "Type registry: custom type" begin
        register_type!("TestCustomType", d -> Dict{String, Any}(
            "deserialized" => true,
            "value" => d["value"],
        ))
        d = Dict{String, Any}("_type" => "TestCustomType", "value" => 42)
        result = deserialize_from_dict(d)
        @test result["deserialized"] == true
        @test result["value"] == 42
    end

    @testset "Registered state type roundtrip" begin
        register_state_type!(RegisteredSessionState)

        session = AgentSession(
            id = "typed-state",
            state = Dict{String, Any}(
                "registered" => RegisteredSessionState(name="demo", count=3),
                "nested" => Dict{String, Any}("items" => [RegisteredSessionState(name="inner", count=1)]),
            ),
        )

        session2 = deserialize_from_dict(serialize_to_dict(session))
        @test session2.state["registered"] isa RegisteredSessionState
        @test session2.state["registered"].name == "demo"
        @test session2.state["registered"].count == 3
        @test session2.state["nested"]["items"][1] isa RegisteredSessionState
        @test session2.state["nested"]["items"][1].name == "inner"
    end

    # ── Unknown type ─────────────────────────────────────────────────────────

    @testset "Unknown type returns raw dict" begin
        d = Dict{String, Any}("_type" => "UnknownType99", "foo" => "bar")
        result = deserialize_from_dict(d)
        @test result isa Dict
        @test result["foo"] == "bar"

        # No _type field at all
        d2 = Dict{String, Any}("foo" => "bar")
        result2 = deserialize_from_dict(d2)
        @test result2 isa Dict
        @test result2["foo"] == "bar"
    end

    # ── Edge cases ───────────────────────────────────────────────────────────

    @testset "Edge cases" begin
        # Empty message contents
        msg = Message(role=:user, contents=Content[])
        d = serialize_to_dict(msg)
        @test d["contents"] == []
        msg2 = deserialize_from_dict(d)
        @test msg2 isa Message
        @test isempty(msg2.contents)

        # Content with nil text
        c = Content(type=TEXT)
        d = serialize_to_dict(c)
        @test d["type"] == "text"
        c2 = deserialize_from_dict(d)
        @test c2 isa Content
        @test c2.type == TEXT

        # Special characters in text survive JSON roundtrip
        special = "Hello \"world\" \n\ttab & <special> 🎉"
        msg = Message(:user, special)
        json = serialize_to_json(msg)
        msg2 = deserialize_from_json(json)
        @test get_text(msg2) == special

        # Empty ChatOptions roundtrip
        opts = ChatOptions()
        d = serialize_to_dict(opts)
        @test d["_type"] == "ChatOptions"
        opts2 = deserialize_from_dict(d)
        @test opts2 isa ChatOptions
        @test opts2.model === nothing
        @test opts2.temperature === nothing

        # Empty AgentSession (state/metadata omitted when empty)
        session = AgentSession(id="empty")
        d = serialize_to_dict(session)
        @test d["_type"] == "AgentSession"
        @test d["id"] == "empty"
        @test !haskey(d, "state")
        @test !haskey(d, "metadata")
        session2 = deserialize_from_dict(d)
        @test session2 isa AgentSession
        @test session2.id == "empty"
        @test isempty(session2.state)
        @test isempty(session2.metadata)
    end
end
