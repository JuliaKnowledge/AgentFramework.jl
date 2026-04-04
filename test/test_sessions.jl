using AgentFramework
using Test

@testset "Sessions" begin
    @testset "AgentSession creation" begin
        s = AgentSession()
        @test !isempty(s.id)
        @test isempty(s.state)
        @test s.user_id === nothing
        @test s.thread_id === nothing

        s2 = AgentSession(id="test-session", user_id="user-1")
        @test s2.id == "test-session"
        @test s2.user_id == "user-1"
    end

    @testset "AgentSession serialization" begin
        s = AgentSession(id="s1", user_id="u1")
        s.state["counter"] = 42
        d = session_to_dict(s)
        @test d["id"] == "s1"
        @test d["user_id"] == "u1"
        @test d["state"]["counter"] == 42

        restored = session_from_dict(d)
        @test restored.id == "s1"
        @test restored.state["counter"] == 42
    end

    @testset "SessionContext" begin
        ctx = SessionContext(
            session_id = "s1",
            input_messages = [Message(:user, "hello")],
        )
        @test ctx.session_id == "s1"
        @test length(ctx.input_messages) == 1
        @test isempty(ctx.instructions)
        @test isempty(ctx.tools)
    end

    @testset "SessionContext extend_messages!" begin
        ctx = SessionContext(input_messages = Message[])
        history = [Message(:user, "old message"), Message(:assistant, "old response")]
        extend_messages!(ctx, "history", history)
        @test haskey(ctx.context_messages, "history")
        @test length(ctx.context_messages["history"]) == 2

        # Second call appends
        extend_messages!(ctx, "history", [Message(:user, "another")])
        @test length(ctx.context_messages["history"]) == 3

        # Different source
        extend_messages!(ctx, "rag", [Message(:system, "context from RAG")])
        @test haskey(ctx.context_messages, "rag")
    end

    @testset "SessionContext extend_instructions!" begin
        ctx = SessionContext(input_messages = Message[])
        extend_instructions!(ctx, "Be helpful.")
        extend_instructions!(ctx, ["Be concise.", "Be accurate."])
        @test length(ctx.instructions) == 3
    end

    @testset "get_all_context_messages" begin
        ctx = SessionContext(input_messages = Message[])
        extend_messages!(ctx, "a", [Message(:system, "from A")])
        extend_messages!(ctx, "b", [Message(:system, "from B")])
        all = get_all_context_messages(ctx)
        @test length(all) == 2
    end

    @testset "InMemoryHistoryProvider" begin
        provider = InMemoryHistoryProvider()
        @test isempty(get_messages(provider, "s1"))

        save_messages!(provider, "s1", [Message(:user, "hi"), Message(:assistant, "hello")])
        msgs = get_messages(provider, "s1")
        @test length(msgs) == 2
        @test msgs[1].text == "hi"

        # Append more
        save_messages!(provider, "s1", [Message(:user, "how are you?")])
        @test length(get_messages(provider, "s1")) == 3

        # Different session
        @test isempty(get_messages(provider, "s2"))
    end
end
