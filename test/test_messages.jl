using AgentFramework
using Test

@testset "Messages" begin
    @testset "Message from string" begin
        msg = Message(:user, "Hello, world!")
        @test msg.role == :user
        @test length(msg.contents) == 1
        @test msg.text == "Hello, world!"
    end

    @testset "Message from Content vector" begin
        msg = Message(:assistant, [text_content("Part 1"), text_content("Part 2")])
        @test msg.role == :assistant
        @test length(msg.contents) == 2
        @test msg.text == "Part 1 Part 2"
    end

    @testset "Message from mixed vector" begin
        msg = Message(:user, ["text input", text_content("content input")])
        @test length(msg.contents) == 2
        @test all(c -> is_text(c), msg.contents)
    end

    @testset "Message equality" begin
        a = Message(:user, "hello")
        b = Message(:user, "hello")
        c = Message(:assistant, "hello")
        @test a == b
        @test a != c
    end

    @testset "Message show" begin
        msg = Message(:user, "What is the weather?")
        s = sprint(show, msg)
        @test contains(s, "user")
        @test contains(s, "weather")
    end

    @testset "normalize_messages" begin
        # String
        msgs = normalize_messages("hello")
        @test length(msgs) == 1
        @test msgs[1].role == :user
        @test msgs[1].text == "hello"

        # Content
        msgs = normalize_messages(text_content("hello"))
        @test length(msgs) == 1

        # Message passthrough
        msg = Message(:assistant, "hi")
        msgs = normalize_messages(msg)
        @test msgs[1] === msg

        # Vector of mixed
        msgs = normalize_messages(["hello", Message(:user, "world")])
        @test length(msgs) == 2

        # Nothing
        msgs = normalize_messages(nothing)
        @test isempty(msgs)
    end

    @testset "prepend_instructions" begin
        msgs = [Message(:user, "What time is it?")]
        result = prepend_instructions(msgs, "You are a clock.")
        @test length(result) == 2
        @test result[1].role == :system
        @test result[1].text == "You are a clock."
        @test result[2].role == :user

        # Empty instructions → no change
        result2 = prepend_instructions(msgs, String[])
        @test length(result2) == 1
    end

    @testset "Message serialization round-trip" begin
        original = Message(:user, "Hello!")
        d = message_to_dict(original)
        @test d["role"] == "user"
        @test d["type"] == "chat_message"

        restored = message_from_dict(d)
        @test restored.role == :user
        @test restored.text == "Hello!"
    end
end
