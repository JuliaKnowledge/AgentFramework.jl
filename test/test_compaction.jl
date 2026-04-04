# Tests for message compaction

using AgentFramework
using Test

@testset "Message Compaction" begin

    # Helper to create messages
    function make_msg(role::Symbol, text::String)
        Message(role=role, contents=[text_content(text)])
    end

    function make_messages(n::Int; prefix="Message")
        msgs = Message[]
        for i in 1:n
            role = isodd(i) ? :user : :assistant
            push!(msgs, make_msg(role, "$prefix $i " * "x"^100))
        end
        return msgs
    end

    @testset "Token Estimation" begin
        @test estimate_tokens("hello world") > 0
        @test estimate_tokens("") == 1  # min 1
        @test estimate_tokens("a"^100) == 25  # 100/4
        @test estimate_tokens("a"^100; chars_per_token=2.0) == 50
    end

    @testset "Message Token Estimation" begin
        msg = make_msg(:user, "Hello world")
        tokens = estimate_message_tokens(msg)
        @test tokens > 4  # at least overhead
    end

    @testset "Messages Token Estimation" begin
        msgs = make_messages(5)
        tokens = estimate_messages_tokens(msgs)
        @test tokens > 0
        @test tokens > estimate_messages_tokens(msgs[1:2])
    end

    @testset "No Compaction Strategy" begin
        msgs = make_messages(10)
        config = CompactionConfig(strategy=NO_COMPACTION, max_tokens=1)
        result = compact_messages(msgs, config)
        @test length(result) == length(msgs)
        @test !needs_compaction(msgs, config)
    end

    @testset "Under Budget — No Change" begin
        msgs = make_messages(2)
        config = CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=100000)
        result = compact_messages(msgs, config)
        @test length(result) == length(msgs)
        @test !needs_compaction(msgs, config)
    end

    @testset "Sliding Window" begin
        msgs = make_messages(20)
        config = CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=200)
        result = compact_messages(msgs, config)
        @test length(result) < length(msgs)
        @test length(result) > 0
        # Last message should be preserved
        @test result[end] === msgs[end]
    end

    @testset "Sliding Window Preserves System Messages" begin
        sys_msg = make_msg(:system, "You are a helpful assistant.")
        msgs = vcat([sys_msg], make_messages(20))
        config = CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=200, keep_system=true)
        result = compact_messages(msgs, config)
        @test result[1].role == :system
        @test length(result) < length(msgs)
    end

    @testset "Drop Oldest" begin
        msgs = make_messages(20)
        config = CompactionConfig(strategy=DROP_OLDEST, max_tokens=200, keep_recent=3)
        result = compact_messages(msgs, config)
        @test length(result) < length(msgs)
        # Recent messages preserved
        @test result[end] === msgs[end]
    end

    @testset "Drop Oldest Keeps System" begin
        sys_msg = make_msg(:system, "System instructions")
        msgs = vcat([sys_msg], make_messages(20))
        config = CompactionConfig(strategy=DROP_OLDEST, max_tokens=200, keep_recent=2, keep_system=true)
        result = compact_messages(msgs, config)
        @test result[1].role == :system
    end

    @testset "Summarize Oldest" begin
        msgs = make_messages(20)
        config = CompactionConfig(strategy=SUMMARIZE_OLDEST, max_tokens=200, keep_recent=3)
        result = compact_messages(msgs, config)
        @test length(result) < length(msgs)
        # Should have a summary message
        has_summary = any(m -> m.role == :system && any(c -> c.text !== nothing && occursin("Conversation summary", c.text), m.contents), result)
        @test has_summary
        # Recent messages preserved
        @test result[end] === msgs[end]
    end

    @testset "Summarize Oldest — Few Messages" begin
        msgs = make_messages(2)
        config = CompactionConfig(strategy=SUMMARIZE_OLDEST, max_tokens=1, keep_recent=5)
        result = compact_messages(msgs, config)
        # When keep_recent >= total, no summarization needed
        @test length(result) == length(msgs)
    end

    @testset "needs_compaction" begin
        msgs = make_messages(50)
        config = CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=100)
        @test needs_compaction(msgs, config)

        config2 = CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=1000000)
        @test !needs_compaction(msgs, config2)
    end

    @testset "CompactionConfig Defaults" begin
        config = CompactionConfig()
        @test config.strategy == SLIDING_WINDOW
        @test config.max_tokens == 4096
        @test config.keep_system == true
        @test config.keep_recent == 4
        @test config.chars_per_token == 4.0
    end

    @testset "Empty Messages" begin
        config = CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=100)
        result = compact_messages(Message[], config)
        @test isempty(result)
    end
end
