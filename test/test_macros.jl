# @executor, @handler, @middleware, and @pipeline macro tests

using AgentFramework
using Test

@testset "Macros" begin

    # ── @executor ────────────────────────────────────────────────────────────

    @testset "@executor with typed first arg" begin
        spec = @executor "upper" function(msg::String, ctx)
            send_message(ctx, uppercase(msg))
        end

        @test spec isa ExecutorSpec
        @test spec.id == "upper"
        @test spec.input_types == DataType[String]
        @test spec.output_types == DataType[Any]
        @test spec.description == ""
    end

    @testset "@executor with untyped arg defaults to Any" begin
        spec = @executor "passthrough" function(msg, ctx)
            send_message(ctx, msg)
        end

        @test spec isa ExecutorSpec
        @test spec.input_types == DataType[Any]
    end

    @testset "@executor with description string" begin
        spec = @executor "desc_test" "A helpful description" function(msg::String, ctx)
            send_message(ctx, msg)
        end

        @test spec.id == "desc_test"
        @test spec.description == "A helpful description"
        @test spec.input_types == DataType[String]
    end

    @testset "@executor handler executes correctly" begin
        spec = @executor "double" function(msg::Int, ctx)
            send_message(ctx, msg * 2)
        end

        result_ctx = execute_handler(spec, 21, String[], Dict{String, Any}())
        @test length(result_ctx._sent_messages) == 1
        @test result_ctx._sent_messages[1].data == 42
    end

    @testset "@executor input_types set from annotation" begin
        spec_float = @executor "float_handler" function(data::Float64, ctx)
            send_message(ctx, string(data))
        end

        @test spec_float.input_types == DataType[Float64]

        spec_int = @executor "int_handler" function(n::Int, ctx)
            send_message(ctx, n + 1)
        end

        @test spec_int.input_types == DataType[Int]
    end

    # ── @handler ─────────────────────────────────────────────────────────────

    @testset "@handler creates a callable function" begin
        @handler my_upper_handler function(msg::String, ctx::WorkflowContext)
            send_message(ctx, uppercase(msg))
        end

        @test my_upper_handler isa Function
        # Verify it's callable with correct signature
        test_ctx = WorkflowContext(executor_id = "test")
        my_upper_handler("hello", test_ctx)
        @test length(test_ctx._sent_messages) == 1
        @test test_ctx._sent_messages[1].data == "HELLO"
    end

    @testset "@handler works with ExecutorSpec" begin
        @handler concat_handler function(msg::String, ctx)
            send_message(ctx, msg * msg)
        end

        spec = ExecutorSpec(id = "concat", handler = concat_handler)
        result_ctx = execute_handler(spec, "ab", String[], Dict{String, Any}())
        @test result_ctx._sent_messages[1].data == "abab"
    end

    # ── @middleware ───────────────────────────────────────────────────────────

    @testset "@middleware :agent creates agent middleware" begin
        @middleware :agent test_agent_mw function(ctx, next)
            ctx.metadata["before"] = true
            result = next(ctx)
            ctx.metadata["after"] = true
            return result
        end

        @test test_agent_mw isa Function

        agent_ctx = AgentContext(metadata = Dict{String, Any}())
        test_agent_mw(agent_ctx, ctx -> begin
            ctx.metadata["inner"] = true
            return nothing
        end)

        @test agent_ctx.metadata["before"] == true
        @test agent_ctx.metadata["inner"] == true
        @test agent_ctx.metadata["after"] == true
    end

    @testset "@middleware :chat creates chat middleware" begin
        @middleware :chat test_chat_mw function(ctx, next)
            result = next(ctx)
            return result
        end

        @test test_chat_mw isa Function
    end

    @testset "@middleware :function creates function middleware" begin
        @middleware :function test_func_mw function(ctx, next)
            result = next(ctx)
            return result
        end

        @test test_func_mw isa Function
    end

    # ── @executor in WorkflowBuilder integration ─────────────────────────────

    @testset "@executor integrates with WorkflowBuilder" begin
        start_exec = @executor "start" function(msg::String, ctx)
            send_message(ctx, uppercase(msg))
        end

        end_exec = @executor "finish" function(msg::String, ctx)
            yield_output(ctx, "done: " * msg)
        end

        workflow = WorkflowBuilder(name = "MacroWorkflow", start = start_exec) |>
            b -> add_executor(b, end_exec) |>
            b -> add_edge(b, "start", "finish") |>
            b -> add_output(b, "finish") |>
            build

        @test workflow.name == "MacroWorkflow"
        @test haskey(workflow.executors, "start")
        @test haskey(workflow.executors, "finish")
    end

    @testset "Multiple @executor definitions compose" begin
        e1 = @executor "step1" function(msg::String, ctx)
            send_message(ctx, msg * "_step1")
        end

        e2 = @executor "step2" function(msg::String, ctx)
            send_message(ctx, msg * "_step2")
        end

        e3 = @executor "step3" function(msg::String, ctx)
            yield_output(ctx, msg * "_step3")
        end

        # Build a linear pipeline manually
        workflow = WorkflowBuilder(name = "Multi", start = e1) |>
            b -> add_executor(b, e2) |>
            b -> add_executor(b, e3) |>
            b -> add_edge(b, "step1", "step2") |>
            b -> add_edge(b, "step2", "step3") |>
            b -> add_output(b, "step3") |>
            build

        @test length(workflow.executors) == 3

        # Verify handler chain works
        ctx1 = execute_handler(e1, "hello", String[], Dict{String, Any}())
        @test ctx1._sent_messages[1].data == "hello_step1"

        ctx2 = execute_handler(e2, ctx1._sent_messages[1].data, String[], Dict{String, Any}())
        @test ctx2._sent_messages[1].data == "hello_step1_step2"

        ctx3 = execute_handler(e3, ctx2._sent_messages[1].data, String[], Dict{String, Any}())
        @test ctx3._yielded_outputs[1] == "hello_step1_step2_step3"
    end

    # ── @pipeline ────────────────────────────────────────────────────────────

    @testset "@pipeline creates workflow from chain" begin
        p_upper = @executor "p_upper" function(msg::String, ctx)
            send_message(ctx, uppercase(msg))
        end

        p_reverse = @executor "p_reverse" function(msg::String, ctx)
            yield_output(ctx, reverse(msg))
        end

        wf = @pipeline "PipelineTest" p_upper => p_reverse

        @test wf isa Workflow
        @test wf.name == "PipelineTest"
        @test haskey(wf.executors, "p_upper")
        @test haskey(wf.executors, "p_reverse")
        @test wf.start_executor_id == "p_upper"
        @test "p_reverse" in wf.output_executor_ids
    end

    @testset "@pipeline with three stages" begin
        s1 = @executor "s1" function(msg::String, ctx)
            send_message(ctx, msg * "A")
        end
        s2 = @executor "s2" function(msg::String, ctx)
            send_message(ctx, msg * "B")
        end
        s3 = @executor "s3" function(msg::String, ctx)
            yield_output(ctx, msg * "C")
        end

        wf = @pipeline "ThreeStage" s1 => s2 => s3

        @test wf isa Workflow
        @test length(wf.executors) == 3
        @test wf.start_executor_id == "s1"
        @test "s3" in wf.output_executor_ids
        @test length(wf.edge_groups) == 2
    end

    # ── @executor with yield_output ──────────────────────────────────────────

    @testset "@executor with yield_output" begin
        spec = @executor "yielder" function(msg::String, ctx)
            result = uppercase(msg)
            yield_output(ctx, result)
            send_message(ctx, result)
        end

        result_ctx = execute_handler(spec, "hello", String[], Dict{String, Any}())
        @test length(result_ctx._yielded_outputs) == 1
        @test result_ctx._yielded_outputs[1] == "HELLO"
        @test length(result_ctx._sent_messages) == 1
        @test result_ctx._sent_messages[1].data == "HELLO"
    end

    # ── @executor with state ─────────────────────────────────────────────────

    @testset "@executor with state access" begin
        spec = @executor "stateful" function(msg::String, ctx)
            count = get_state(ctx, "count", 0)
            set_state!(ctx, "count", count + 1)
            send_message(ctx, "count=$(count + 1)")
        end

        state = Dict{String, Any}("count" => 5)
        result_ctx = execute_handler(spec, "inc", String[], state)
        @test result_ctx._sent_messages[1].data == "count=6"
        @test result_ctx._state["count"] == 6
    end

end
