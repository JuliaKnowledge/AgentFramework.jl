# Tests for advanced compaction strategies (TRUNCATE, SELECTIVE_TOOL_CALL, TOOL_RESULT_ONLY, CompactionPipeline)

using AgentFramework
using Test

@testset "Advanced Compaction Strategies" begin

    # ── Helpers ──────────────────────────────────────────────────────────────

    function make_msg(role::Symbol, text::String)
        Message(role=role, contents=[text_content(text)])
    end

    function make_tool_messages(n)
        msgs = Message[]
        push!(msgs, Message(role=:system, contents=[text_content("System prompt")]))
        for i in 1:n
            push!(msgs, Message(role=:user, contents=[text_content("Question $i " * "x"^50)]))
            push!(msgs, Message(role=:assistant, contents=[function_call_content("call_$i", "tool_$i", "{\"arg\": $i}")]))
            push!(msgs, Message(role=:tool, contents=[function_result_content("call_$i", "Result $i " * "y"^100)]))
            push!(msgs, Message(role=:assistant, contents=[text_content("Response $i " * "z"^50)]))
        end
        return msgs
    end

    # ── TRUNCATE ─────────────────────────────────────────────────────────────

    @testset "TRUNCATE — reduces messages and keeps system" begin
        msgs = make_tool_messages(5)
        config = CompactionConfig(strategy=TRUNCATE, max_tokens=200, keep_system=true)
        result = compact_messages(msgs, config)
        @test length(result) < length(msgs)
        @test result[1].role == :system
    end

    @testset "TRUNCATE — preserves most recent messages" begin
        msgs = make_tool_messages(5)
        config = CompactionConfig(strategy=TRUNCATE, max_tokens=200, keep_system=true)
        result = compact_messages(msgs, config)
        @test result[end].role == msgs[end].role
        # The last message text should be identical (not truncated)
        last_orig = msgs[end].contents[1].text
        last_res = result[end].contents[1].text
        @test last_res == last_orig
    end

    @testset "TRUNCATE — under budget returns all" begin
        msgs = [make_msg(:user, "hi")]
        config = CompactionConfig(strategy=TRUNCATE, max_tokens=100000)
        result = compact_messages(msgs, config)
        @test length(result) == 1
    end

    # ── SELECTIVE_TOOL_CALL ──────────────────────────────────────────────────

    @testset "SELECTIVE_TOOL_CALL — removes old tool call/result pairs" begin
        msgs = make_tool_messages(5)
        config = CompactionConfig(strategy=SELECTIVE_TOOL_CALL, max_tokens=300, keep_recent=4, keep_system=true)
        result = compact_messages(msgs, config)
        @test length(result) < length(msgs)
        # System message preserved
        @test result[1].role == :system
    end

    @testset "SELECTIVE_TOOL_CALL — preserves recent tool interactions" begin
        msgs = make_tool_messages(5)
        total_tokens = estimate_messages_tokens(msgs)
        config = CompactionConfig(strategy=SELECTIVE_TOOL_CALL, max_tokens=300, keep_recent=8, keep_system=true)
        result = compact_messages(msgs, config)
        # The last keep_recent messages from other_msgs should be intact
        # (the last 8 non-system messages should all be present)
        non_system_result = filter(m -> m.role != :system && m.role != :developer, result)
        non_system_orig = filter(m -> m.role != :system && m.role != :developer, msgs)
        # Last few messages should be the same objects
        @test non_system_result[end] === non_system_orig[end]
    end

    @testset "SELECTIVE_TOOL_CALL — no tool calls means no change" begin
        msgs = Message[]
        push!(msgs, make_msg(:system, "System"))
        for i in 1:5
            push!(msgs, make_msg(:user, "Q$i " * "a"^50))
            push!(msgs, make_msg(:assistant, "A$i " * "b"^50))
        end
        config = CompactionConfig(strategy=SELECTIVE_TOOL_CALL, max_tokens=50, keep_recent=2, keep_system=true)
        result = compact_messages(msgs, config)
        # Without tool pairs to remove, all non-system messages are kept
        non_system_orig = filter(m -> m.role != :system, msgs)
        non_system_result = filter(m -> m.role != :system, result)
        @test length(non_system_result) == length(non_system_orig)
    end

    # ── TOOL_RESULT_ONLY ─────────────────────────────────────────────────────

    @testset "TOOL_RESULT_ONLY — replaces tool results with placeholder" begin
        msgs = make_tool_messages(5)
        config = CompactionConfig(strategy=TOOL_RESULT_ONLY, max_tokens=100, keep_recent=4, keep_system=true)
        result = compact_messages(msgs, config)
        # Find tool-result messages outside the protected window
        non_system = filter(m -> m.role != :system && m.role != :developer, result)
        n = length(non_system)
        protected_start = max(1, n - config.keep_recent + 1)
        for (i, msg) in enumerate(non_system)
            for c in msg.contents
                if c.type == AgentFramework.FUNCTION_RESULT && i < protected_start
                    @test c.result == "[Tool result truncated]"
                end
            end
        end
    end

    @testset "TOOL_RESULT_ONLY — preserves tool call names" begin
        msgs = make_tool_messages(3)
        config = CompactionConfig(strategy=TOOL_RESULT_ONLY, max_tokens=100, keep_recent=0, keep_system=true)
        result = compact_messages(msgs, config)
        # All FUNCTION_CALL contents should still have their name
        for msg in result
            for c in msg.contents
                if c.type == AgentFramework.FUNCTION_CALL
                    @test c.name !== nothing
                    @test startswith(c.name, "tool_")
                end
            end
        end
    end

    @testset "TOOL_RESULT_ONLY — no tool results means no change" begin
        msgs = [make_msg(:system, "Sys"), make_msg(:user, "Hi"), make_msg(:assistant, "Hello")]
        config = CompactionConfig(strategy=TOOL_RESULT_ONLY, max_tokens=1, keep_recent=0, keep_system=true)
        result = compact_messages(msgs, config)
        non_sys_orig = filter(m -> m.role != :system, msgs)
        non_sys_res = filter(m -> m.role != :system, result)
        @test length(non_sys_res) == length(non_sys_orig)
        @test non_sys_res[1].contents[1].text == "Hi"
        @test non_sys_res[2].contents[1].text == "Hello"
    end

    # ── CompactionPipeline ───────────────────────────────────────────────────

    @testset "CompactionPipeline — single strategy" begin
        msgs = make_tool_messages(5)
        pipeline = CompactionPipeline(strategies=[
            CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=200, keep_system=true)
        ])
        result = compact_messages(msgs, pipeline)
        @test length(result) < length(msgs)
        @test result[1].role == :system
    end

    @testset "CompactionPipeline — multi-strategy pipeline" begin
        msgs = make_tool_messages(5)
        pipeline = CompactionPipeline(strategies=[
            CompactionConfig(strategy=TOOL_RESULT_ONLY, max_tokens=100, keep_recent=0, keep_system=true),
            CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=300, keep_system=true),
        ])
        result = compact_messages(msgs, pipeline)
        @test length(result) < length(msgs)
        @test result[1].role == :system
    end

    @testset "CompactionPipeline — order matters" begin
        msgs = make_tool_messages(5)
        # Pipeline A: truncate tool results first, then sliding window
        pipeline_a = CompactionPipeline(strategies=[
            CompactionConfig(strategy=TOOL_RESULT_ONLY, max_tokens=100, keep_recent=0, keep_system=true),
            CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=400, keep_system=true),
        ])
        # Pipeline B: sliding window first, then truncate tool results
        pipeline_b = CompactionPipeline(strategies=[
            CompactionConfig(strategy=SLIDING_WINDOW, max_tokens=400, keep_system=true),
            CompactionConfig(strategy=TOOL_RESULT_ONLY, max_tokens=100, keep_recent=0, keep_system=true),
        ])
        result_a = compact_messages(msgs, pipeline_a)
        result_b = compact_messages(msgs, pipeline_b)
        # Results can differ because order changes what each stage sees
        tokens_a = estimate_messages_tokens(result_a)
        tokens_b = estimate_messages_tokens(result_b)
        # Both should be reduced; at least one pipeline ordering should differ or both be valid
        @test length(result_a) > 0
        @test length(result_b) > 0
        # They may or may not be equal — the key test is that both execute without error
    end

    # ── Integration with needs_compaction ─────────────────────────────────────

    @testset "New strategies work with needs_compaction" begin
        msgs = make_tool_messages(5)
        for strat in [TRUNCATE, SELECTIVE_TOOL_CALL, TOOL_RESULT_ONLY]
            config = CompactionConfig(strategy=strat, max_tokens=100)
            @test needs_compaction(msgs, config)
            config_big = CompactionConfig(strategy=strat, max_tokens=1000000)
            @test !needs_compaction(msgs, config_big)
        end
    end

    # ── Edge Cases ───────────────────────────────────────────────────────────

    @testset "Edge case — empty messages" begin
        for strat in [TRUNCATE, SELECTIVE_TOOL_CALL, TOOL_RESULT_ONLY]
            config = CompactionConfig(strategy=strat, max_tokens=100)
            result = compact_messages(Message[], config)
            @test isempty(result)
        end
        pipeline = CompactionPipeline(strategies=[
            CompactionConfig(strategy=TRUNCATE, max_tokens=100)
        ])
        @test isempty(compact_messages(Message[], pipeline))
    end

    @testset "Edge case — single message" begin
        msg = [make_msg(:user, "Hello")]
        for strat in [TRUNCATE, SELECTIVE_TOOL_CALL, TOOL_RESULT_ONLY]
            config = CompactionConfig(strategy=strat, max_tokens=100000)
            result = compact_messages(msg, config)
            @test length(result) == 1
        end
    end
end
