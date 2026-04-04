using AgentFramework
using Test

@testset "Chat Client Types" begin
    @testset "FinishReason" begin
        @test parse_finish_reason("stop") == STOP
        @test parse_finish_reason("tool_calls") == TOOL_CALLS
        @test parse_finish_reason("length") == LENGTH
        @test parse_finish_reason("unknown") == STOP  # defaults to STOP
    end

    @testset "ChatOptions" begin
        opts = ChatOptions(temperature=0.7, max_tokens=100)
        @test opts.temperature == 0.7
        @test opts.max_tokens == 100
        @test opts.model === nothing
    end

    @testset "merge_chat_options" begin
        base = ChatOptions(model="gpt-4", temperature=0.5)
        override = ChatOptions(temperature=0.9, max_tokens=200)
        merged = merge_chat_options(base, override)
        @test merged.model == "gpt-4"       # kept from base
        @test merged.temperature == 0.9      # overridden
        @test merged.max_tokens == 200       # added from override
    end

    @testset "ChatResponseUpdate" begin
        update = ChatResponseUpdate(
            role = :assistant,
            contents = [text_content("Hello")],
        )
        @test get_text(update) == "Hello"
    end

    @testset "ChatResponse from updates" begin
        updates = [
            ChatResponseUpdate(role=:assistant, contents=[text_content("Hello")]),
            ChatResponseUpdate(contents=[text_content(" world")]),
            ChatResponseUpdate(finish_reason=STOP, model_id="test-model"),
        ]
        response = ChatResponse(updates)
        @test length(response.messages) == 1
        @test response.text == "Hello world"
        @test response.finish_reason == STOP
        @test response.model_id == "test-model"
    end

    @testset "ChatResponse empty" begin
        response = ChatResponse()
        @test isempty(response.messages)
        @test response.text == ""
    end

    @testset "AgentResponse" begin
        resp = AgentResponse(
            messages = [Message(:assistant, "Done.")],
            finish_reason = STOP,
        )
        @test resp.text == "Done."
        @test resp.finish_reason == STOP
    end
end
