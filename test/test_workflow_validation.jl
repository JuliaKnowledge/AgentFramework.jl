using AgentFramework
using Test
using Logging

@testset "Workflow Validation" begin

    noop_handler = (msg, ctx) -> nothing

    # Helper: create a simple executor spec with specific types
    function make_spec(id; input=DataType[Any], output=DataType[Any], yield_types=DataType[Any])
        ExecutorSpec(
            id = id,
            input_types = input,
            output_types = output,
            yield_types = yield_types,
            handler = noop_handler,
        )
    end

    @testset "Edge duplication detection" begin
        a = make_spec("a")
        b = make_spec("b")

        builder = WorkflowBuilder(name = "EdgeDup", start = a)
        add_executor(builder, b)
        add_edge(builder, "a", "b")
        add_output(builder, "b")

        # Manually duplicate edge group ID
        push!(builder.edge_groups, EdgeGroup(
            kind = DIRECT_EDGE,
            edges = [Edge(source_id = "a", target_id = "b")],
            id = builder.edge_groups[1].id,  # duplicate ID
        ))

        result = validate_workflow(builder; checks = [CHECK_EDGE_DUPLICATION])
        @test !result.valid
        @test length(result.issues) == 1
        @test result.issues[1].check == CHECK_EDGE_DUPLICATION
        @test result.issues[1].severity == :error
        @test occursin("Duplicate", result.issues[1].message)
    end

    @testset "Graph connectivity — all connected" begin
        a = make_spec("a")
        b = make_spec("b")
        c = make_spec("c")

        builder = WorkflowBuilder(name = "Connected", start = a)
        add_executor(builder, b)
        add_executor(builder, c)
        add_edge(builder, "a", "b")
        add_edge(builder, "b", "c")
        add_output(builder, "c")

        result = validate_workflow(builder; checks = [CHECK_GRAPH_CONNECTIVITY])
        @test result.valid
        @test isempty(result.issues)
    end

    @testset "Graph connectivity — unreachable executor" begin
        a = make_spec("a")
        b = make_spec("b")
        c = make_spec("c")

        builder = WorkflowBuilder(name = "Disconnected", start = a)
        add_executor(builder, b)
        add_executor(builder, c)
        add_edge(builder, "a", "b")
        add_output(builder, "b")
        # "c" is not connected via edges from "a"

        result = validate_workflow(builder; checks = [CHECK_GRAPH_CONNECTIVITY])
        @test result.valid  # connectivity issues are warnings, not errors
        @test length(result.issues) == 1
        @test result.issues[1].check == CHECK_GRAPH_CONNECTIVITY
        @test result.issues[1].severity == :warning
        @test occursin("c", result.issues[1].message)
        @test "c" in result.issues[1].executor_ids
    end

    @testset "Self-loop detection" begin
        a = make_spec("a")
        builder = WorkflowBuilder(name = "SelfLoop", start = a)
        add_edge(builder, "a", "a")
        add_output(builder, "a")

        result = validate_workflow(builder; checks = [CHECK_SELF_LOOPS])
        @test result.valid  # self-loops are warnings
        @test length(result.issues) == 1
        @test result.issues[1].check == CHECK_SELF_LOOPS
        @test result.issues[1].severity == :warning
        @test occursin("Self-loop", result.issues[1].message)
        @test "a" in result.issues[1].executor_ids
    end

    @testset "Output executor without yield_types" begin
        # yield_types defaults to [Any] — should be warned
        a = make_spec("a")
        b = make_spec("b")  # default yield_types=[Any]

        builder = WorkflowBuilder(name = "NoYield", start = a)
        add_executor(builder, b)
        add_edge(builder, "a", "b")
        add_output(builder, "b")

        result = validate_workflow(builder; checks = [CHECK_OUTPUT_EXECUTORS])
        @test result.valid  # output executor issues are warnings
        @test length(result.issues) == 1
        @test result.issues[1].check == CHECK_OUTPUT_EXECUTORS
        @test result.issues[1].severity == :warning
        @test occursin("b", result.issues[1].message)

        # Executor with specific yield_types should not produce a warning
        b_typed = make_spec("b_typed"; yield_types = DataType[String])
        builder2 = WorkflowBuilder(name = "WithYield", start = a)
        add_executor(builder2, b_typed)
        add_edge(builder2, "a", "b_typed")
        add_output(builder2, "b_typed")

        result2 = validate_workflow(builder2; checks = [CHECK_OUTPUT_EXECUTORS])
        @test isempty(result2.issues)
    end

    @testset "Dead-end executor detection" begin
        a = make_spec("a")
        b = make_spec("b")
        c = make_spec("c")

        builder = WorkflowBuilder(name = "DeadEnd", start = a)
        add_executor(builder, b)
        add_executor(builder, c)
        add_edge(builder, "a", "b")
        add_edge(builder, "a", "c")
        add_output(builder, "c")
        # "b" has no outgoing edges and is not an output executor → dead end

        result = validate_workflow(builder; checks = [CHECK_DEAD_ENDS])
        @test result.valid  # dead ends are info
        @test length(result.issues) == 1
        @test result.issues[1].check == CHECK_DEAD_ENDS
        @test result.issues[1].severity == :info
        @test "b" in result.issues[1].executor_ids

        # Single executor workflow should not flag dead ends
        solo = make_spec("solo")
        builder_solo = WorkflowBuilder(name = "Solo", start = solo)
        add_output(builder_solo, "solo")
        result_solo = validate_workflow(builder_solo; checks = [CHECK_DEAD_ENDS])
        @test isempty(result_solo.issues)
    end

    @testset "validate_workflow with all checks" begin
        a = make_spec("a"; output = DataType[String])
        b = make_spec("b"; input = DataType[String], output = DataType[String], yield_types = DataType[String])

        builder = WorkflowBuilder(name = "AllChecks", start = a)
        add_executor(builder, b)
        add_edge(builder, "a", "b")
        add_output(builder, "b")

        result = validate_workflow(builder)
        @test result.valid
        @test isempty(filter(i -> i.severity == :error, result.issues))
    end

    @testset "validate_workflow with subset of checks" begin
        a = make_spec("a")
        b = make_spec("b")

        builder = WorkflowBuilder(name = "SubsetChecks", start = a)
        add_executor(builder, b)
        add_edge(builder, "a", "b")
        add_output(builder, "b")

        # Only run self-loop check
        result = validate_workflow(builder; checks = [CHECK_SELF_LOOPS])
        @test result.valid
        @test isempty(result.issues)

        # Only run edge duplication
        result2 = validate_workflow(builder; checks = [CHECK_EDGE_DUPLICATION])
        @test result2.valid
        @test isempty(result2.issues)
    end

    @testset "Validation on WorkflowBuilder" begin
        a = make_spec("a"; output = DataType[String])
        b = make_spec("b"; input = DataType[Int], yield_types = DataType[Int])

        builder = WorkflowBuilder(name = "BuilderVal", start = a)
        add_executor(builder, b)
        add_edge(builder, "a", "b")
        add_output(builder, "b")

        result = validate_workflow(builder; checks = [CHECK_TYPE_COMPATIBILITY])
        @test !result.valid
        @test any(i -> i.check == CHECK_TYPE_COMPATIBILITY && i.severity == :error, result.issues)
    end

    @testset "Valid workflow produces no issues" begin
        a = make_spec("a"; input = DataType[String], output = DataType[String])
        b = make_spec("b"; input = DataType[String], output = DataType[String], yield_types = DataType[String])

        builder = WorkflowBuilder(name = "Clean", start = a)
        add_executor(builder, b)
        add_edge(builder, "a", "b")
        add_output(builder, "b")

        wf = build(builder; validate_types = false)
        result = validate_workflow(wf)
        @test result.valid
        @test isempty(filter(i -> i.severity == :error, result.issues))
    end

    @testset "Integration with build() — warnings are emitted" begin
        a = make_spec("a"; output = DataType[String])
        b = make_spec("b"; input = DataType[Int], yield_types = DataType[Int])

        builder = WorkflowBuilder(name = "BuildIntegration", start = a)
        add_executor(builder, b)
        add_edge(builder, "a", "b")
        add_output(builder, "b")

        logs, wf = Test.collect_test_logs() do
            build(builder; validate_types = true)
        end

        @test wf isa Workflow
        # Should have type compatibility error logged as a warning
        @test any(l -> l.level == Logging.Warn && occursin("Incompatible", l.message), logs)

        # Build with validation disabled — no warnings
        builder2 = WorkflowBuilder(name = "NoVal", start = a)
        add_executor(builder2, b)
        add_edge(builder2, "a", "b")
        add_output(builder2, "b")

        logs2, wf2 = Test.collect_test_logs() do
            build(builder2; validate_types = false)
        end
        @test isempty(logs2)
        @test wf2 isa Workflow
    end

end
