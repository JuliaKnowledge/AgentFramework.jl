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

    # ═══════════════════════════════════════════════════════════════════════
    #  Per-Service-Call History Persistence
    # ═══════════════════════════════════════════════════════════════════════

    @testset "LOCAL_HISTORY_CONVERSATION_ID" begin
        @test is_local_history_conversation_id(LOCAL_HISTORY_CONVERSATION_ID) == true
        @test is_local_history_conversation_id("other") == false
        @test is_local_history_conversation_id(nothing) == false
    end

    @testset "PerServiceCallHistoryMiddleware construction" begin
        provider = InMemoryHistoryProvider()
        session = AgentSession()
        mw = PerServiceCallHistoryMiddleware(providers=[provider], session=session)
        @test length(mw.providers) == 1
        @test mw.session === session
    end

    @testset "PerServiceCallHistoryMiddleware persists per call" begin
        provider = InMemoryHistoryProvider()
        session = AgentSession(id="test-session")
        mw = PerServiceCallHistoryMiddleware(providers=[provider], session=session)

        # Simulate a chat context (duck-typed)
        ctx = (
            messages = Message[Message(:user, "hello")],
            options = nothing,
            result = nothing,
        )

        # Make it mutable via Ref-like pattern
        mutable_ctx = Base.@kwdef mutable struct _TestCtx
            messages::Vector{Message} = Message[]
            options::Any = nothing
            result::Any = nothing
        end
        test_ctx = _TestCtx(
            messages=[Message(:user, "hello")],
            options=nothing,
            result=nothing,
        )

        # Simulate the LLM returning a response
        function fake_next()
            test_ctx.result = (messages=[Message(:assistant, "world")],)
        end

        mw(test_ctx, fake_next)

        # History should have been persisted
        saved = get_messages(provider, "test-session")
        @test length(saved) == 1
        @test saved[1].role == :assistant
    end

    @testset "with_per_service_call_history helper" begin
        provider = InMemoryHistoryProvider()
        session = AgentSession()
        mw = with_per_service_call_history(session, [provider])
        @test mw isa PerServiceCallHistoryMiddleware
        @test length(mw.providers) == 1
    end
end
