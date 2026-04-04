using AgentFramework
using Test
using Logging

@testset "Protocol Introspection" begin

    # Helper: create a simple executor spec
    noop_handler = (msg, ctx) -> nothing

    @testset "ProtocolDescriptor construction and defaults" begin
        pd = ProtocolDescriptor()
        @test pd.accepts == DataType[Any]
        @test pd.sends == DataType[Any]
        @test pd.yields == DataType[]
        @test pd.accepts_all == false

        pd2 = ProtocolDescriptor(
            accepts = DataType[String, Int],
            sends = DataType[Float64],
            yields = DataType[String],
            accepts_all = false,
        )
        @test pd2.accepts == DataType[String, Int]
        @test pd2.sends == DataType[Float64]
        @test pd2.yields == DataType[String]
        @test pd2.accepts_all == false

        pd3 = ProtocolDescriptor(accepts_all = true)
        @test pd3.accepts_all == true
    end

    @testset "get_protocol extracts from ExecutorSpec" begin
        spec = ExecutorSpec(
            id = "test",
            input_types = DataType[String, Int],
            output_types = DataType[Float64],
            yield_types = DataType[String],
            handler = noop_handler,
        )
        proto = get_protocol(spec)
        @test proto.accepts == DataType[String, Int]
        @test proto.sends == DataType[Float64]
        @test proto.yields == DataType[String]
        @test proto.accepts_all == false

        # Spec with Any in input_types → accepts_all
        spec_any = ExecutorSpec(id = "any_input", handler = noop_handler)
        proto_any = get_protocol(spec_any)
        @test proto_any.accepts_all == true
    end

    @testset "can_handle with concrete types, Any, subtypes" begin
        spec_string = ExecutorSpec(id = "s", input_types = DataType[String], handler = noop_handler)
        @test can_handle(spec_string, String) == true
        @test can_handle(spec_string, Int) == false
        # SubString <: AbstractString but not <: String
        @test can_handle(spec_string, SubString{String}) == false

        # Protocol with Any accepts everything
        proto_any = ProtocolDescriptor(accepts = DataType[Any], accepts_all = true)
        @test can_handle(proto_any, String) == true
        @test can_handle(proto_any, Int) == true
        @test can_handle(proto_any, Vector{Float64}) == true

        # Subtype check with abstract type
        spec_number = ExecutorSpec(id = "n", input_types = DataType[Number], handler = noop_handler)
        @test can_handle(spec_number, Int) == true
        @test can_handle(spec_number, Float64) == true
        @test can_handle(spec_number, String) == false

        # Empty accepts
        proto_empty = ProtocolDescriptor(accepts = DataType[], accepts_all = false)
        @test can_handle(proto_empty, String) == false
    end

    @testset "can_output with concrete types, Any, subtypes" begin
        spec = ExecutorSpec(id = "o", output_types = DataType[String], handler = noop_handler)
        @test can_output(spec, String) == true
        @test can_output(spec, Int) == false

        # Any in output types
        spec_any = ExecutorSpec(id = "oa", output_types = DataType[Any], handler = noop_handler)
        @test can_output(spec_any, String) == true
        @test can_output(spec_any, Int) == true

        # Subtype
        spec_num = ExecutorSpec(id = "on", output_types = DataType[Number], handler = noop_handler)
        @test can_output(spec_num, Int) == true
        @test can_output(spec_num, Float64) == true
        @test can_output(spec_num, String) == false

        # Protocol with empty sends
        proto_empty = ProtocolDescriptor(sends = DataType[])
        @test can_output(proto_empty, String) == false
    end

    @testset "check_type_compatibility - compatible cases" begin
        # Matching types
        src = ExecutorSpec(id = "src", output_types = DataType[String], handler = noop_handler)
        tgt = ExecutorSpec(id = "tgt", input_types = DataType[String], handler = noop_handler)
        result = check_type_compatibility(src, tgt)
        @test result.compatible == true
        @test result.source_id == "src"
        @test result.target_id == "tgt"

        # Target accepts Any
        tgt_any = ExecutorSpec(id = "tgt_any", input_types = DataType[Any], handler = noop_handler)
        result2 = check_type_compatibility(src, tgt_any)
        @test result2.compatible == true
        @test occursin("accepts Any", result2.message)

        # Source sends Any
        src_any = ExecutorSpec(id = "src_any", output_types = DataType[Any], handler = noop_handler)
        tgt_strict = ExecutorSpec(id = "tgt_strict", input_types = DataType[Int], handler = noop_handler)
        result3 = check_type_compatibility(src_any, tgt_strict)
        @test result3.compatible == true
        @test occursin("sends Any", result3.message)

        # Subtype relationship: Int <: Number
        src_int = ExecutorSpec(id = "src_int", output_types = DataType[Int], handler = noop_handler)
        tgt_num = ExecutorSpec(id = "tgt_num", input_types = DataType[Number], handler = noop_handler)
        result4 = check_type_compatibility(src_int, tgt_num)
        @test result4.compatible == true

        # Multiple types with partial overlap
        src_multi = ExecutorSpec(id = "src_multi", output_types = DataType[String, Int], handler = noop_handler)
        tgt_num2 = ExecutorSpec(id = "tgt_num2", input_types = DataType[Number], handler = noop_handler)
        result5 = check_type_compatibility(src_multi, tgt_num2)
        @test result5.compatible == true  # Int <: Number
    end

    @testset "check_type_compatibility - incompatible cases" begin
        src = ExecutorSpec(id = "src", output_types = DataType[String], handler = noop_handler)
        tgt = ExecutorSpec(id = "tgt", input_types = DataType[Int], handler = noop_handler)
        result = check_type_compatibility(src, tgt)
        @test result.compatible == false
        @test result.source_id == "src"
        @test result.target_id == "tgt"
        @test occursin("Incompatible", result.message)

        # No overlap between completely different types
        src2 = ExecutorSpec(id = "src2", output_types = DataType[Float64, Bool], handler = noop_handler)
        tgt2 = ExecutorSpec(id = "tgt2", input_types = DataType[String, Vector{Int}], handler = noop_handler)
        result2 = check_type_compatibility(src2, tgt2)
        @test result2.compatible == false
    end

    @testset "validate_workflow_types - valid workflow" begin
        upper = ExecutorSpec(
            id = "upper",
            input_types = DataType[String],
            output_types = DataType[String],
            handler = (msg, ctx) -> send_message(ctx, uppercase(msg)),
        )
        reverser = ExecutorSpec(
            id = "reverse",
            input_types = DataType[String],
            output_types = DataType[String],
            handler = (msg, ctx) -> yield_output(ctx, reverse(msg)),
        )

        wf = WorkflowBuilder(name = "Valid", start = upper) |>
            b -> add_executor(b, reverser) |>
            b -> add_edge(b, "upper", "reverse") |>
            b -> add_output(b, "reverse") |>
            b -> build(b; validate_types = false)

        vresult = validate_workflow_types(wf)
        @test vresult.valid == true
        @test isempty(vresult.errors)
        @test isempty(vresult.warnings)
    end

    @testset "validate_workflow_types - invalid workflow" begin
        string_out = ExecutorSpec(
            id = "string_out",
            input_types = DataType[Any],
            output_types = DataType[String],
            handler = (msg, ctx) -> send_message(ctx, string(msg)),
        )
        int_in = ExecutorSpec(
            id = "int_in",
            input_types = DataType[Int],
            output_types = DataType[Int],
            handler = (msg, ctx) -> yield_output(ctx, msg + 1),
        )

        wf = WorkflowBuilder(name = "Invalid", start = string_out) |>
            b -> add_executor(b, int_in) |>
            b -> add_edge(b, "string_out", "int_in") |>
            b -> add_output(b, "int_in") |>
            b -> build(b; validate_types = false)

        vresult = validate_workflow_types(wf)
        @test vresult.valid == false
        @test length(vresult.errors) == 1
        @test vresult.errors[1].source_id == "string_out"
        @test vresult.errors[1].target_id == "int_in"
    end

    @testset "validate_workflow_types with WorkflowBuilder" begin
        string_out = ExecutorSpec(
            id = "string_out",
            input_types = DataType[Any],
            output_types = DataType[String],
            handler = (msg, ctx) -> send_message(ctx, string(msg)),
        )
        int_in = ExecutorSpec(
            id = "int_in",
            input_types = DataType[Int],
            output_types = DataType[Int],
            handler = (msg, ctx) -> yield_output(ctx, msg + 1),
        )

        builder = WorkflowBuilder(name = "BuilderTest", start = string_out)
        add_executor(builder, int_in)
        add_edge(builder, "string_out", "int_in")
        add_output(builder, "int_in")

        vresult = validate_workflow_types(builder)
        @test vresult.valid == false
        @test length(vresult.errors) == 1
    end

    @testset "describe_protocol output formatting" begin
        spec = ExecutorSpec(
            id = "formatter",
            input_types = DataType[String, Int],
            output_types = DataType[Float64],
            yield_types = DataType[String],
            handler = noop_handler,
        )
        desc = describe_protocol(spec)
        @test occursin("Protocol for 'formatter'", desc)
        @test occursin("Accepts:", desc)
        @test occursin("String", desc)
        @test occursin("Int", desc) || occursin("Int64", desc)
        @test occursin("Sends:", desc)
        @test occursin("Float64", desc)
        @test occursin("Yields:", desc)

        # Protocol descriptor directly
        proto = get_protocol(spec)
        desc2 = describe_protocol(proto)
        @test occursin("Protocol for", desc2)
        @test occursin("Accepts:", desc2)

        # Executor with Any input types shows "accepts all"
        spec_any = ExecutorSpec(id = "any_exec", handler = noop_handler)
        desc_any = describe_protocol(spec_any)
        @test occursin("accepts all", desc_any)

        # Empty yields
        spec_no_yield = ExecutorSpec(
            id = "no_yield",
            yield_types = DataType[],
            handler = noop_handler,
        )
        desc3 = describe_protocol(spec_no_yield)
        @test occursin("nothing", desc3)
    end

    @testset "Build-time validation warning integration" begin
        string_out = ExecutorSpec(
            id = "string_out",
            input_types = DataType[Any],
            output_types = DataType[String],
            handler = (msg, ctx) -> send_message(ctx, string(msg)),
        )
        int_in = ExecutorSpec(
            id = "int_in",
            input_types = DataType[Int],
            output_types = DataType[Int],
            handler = (msg, ctx) -> yield_output(ctx, msg + 1),
        )

        # Build with validation enabled — should emit warnings but not throw
        builder = WorkflowBuilder(name = "WarnTest", start = string_out)
        add_executor(builder, int_in)
        add_edge(builder, "string_out", "int_in")
        add_output(builder, "int_in")

        logs, wf = Test.collect_test_logs() do
            build(builder; validate_types = true)
        end

        # Should have produced a warning about type incompatibility
        @test any(l -> l.level == Logging.Warn && occursin("Incompatible", l.message), logs)
        # Workflow should still be built successfully
        @test wf isa Workflow

        # Build with validation disabled — no warnings
        builder2 = WorkflowBuilder(name = "NoWarnTest", start = string_out)
        add_executor(builder2, int_in)
        add_edge(builder2, "string_out", "int_in")
        add_output(builder2, "int_in")

        logs2, wf2 = Test.collect_test_logs() do
            build(builder2; validate_types = false)
        end
        @test isempty(logs2)
        @test wf2 isa Workflow
    end

    @testset "Edge cases" begin
        # Empty input/output types
        spec_empty = ExecutorSpec(
            id = "empty",
            input_types = DataType[],
            output_types = DataType[],
            yield_types = DataType[],
            handler = noop_handler,
        )
        proto = get_protocol(spec_empty)
        @test proto.accepts_all == false
        @test can_handle(spec_empty, String) == false
        @test can_output(spec_empty, String) == false

        # Single type
        spec_single = ExecutorSpec(
            id = "single",
            input_types = DataType[Int],
            output_types = DataType[Int],
            handler = noop_handler,
        )
        @test can_handle(spec_single, Int) == true
        @test can_handle(spec_single, String) == false

        # accepts_all flag directly
        proto_all = ProtocolDescriptor(
            accepts = DataType[String],
            accepts_all = true,
        )
        @test can_handle(proto_all, Int) == true  # accepts_all overrides
        @test can_handle(proto_all, Vector{Float64}) == true

        # Compatibility between two empty-output and empty-input executors
        src_empty = ExecutorSpec(id = "se", output_types = DataType[], handler = noop_handler)
        tgt_empty = ExecutorSpec(id = "te", input_types = DataType[], handler = noop_handler)
        result = check_type_compatibility(src_empty, tgt_empty)
        # No sends means no overlap possible, but also no requirement — not compatible by overlap rule
        @test result.compatible == false

        # Abstract type hierarchy: AbstractString
        spec_abs = ExecutorSpec(id = "abs", input_types = DataType[AbstractString], handler = noop_handler)
        @test can_handle(spec_abs, String) == true
        @test can_handle(spec_abs, SubString{String}) == true
        @test can_handle(spec_abs, Int) == false
    end

end
