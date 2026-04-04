# Fluent workflow builder API for AgentFramework.jl
# Inspired by C#'s WorkflowBuilder and Python's WorkflowBuilder.

"""
    WorkflowBuilder

Fluent API for constructing workflow DAGs. Validates structure at build time.

# Example
```julia
upper = ExecutorSpec(id="upper", handler=(msg, ctx) -> send_message(ctx, uppercase(msg)))
reverse = ExecutorSpec(id="reverse", handler=(msg, ctx) -> yield_output(ctx, reverse(msg)))

workflow = WorkflowBuilder(name="TextPipeline", start=upper) |>
    b -> add_executor(b, reverse) |>
    b -> add_edge(b, "upper", "reverse") |>
    b -> add_output(b, "reverse") |>
    build
```
"""
mutable struct WorkflowBuilder
    name::String
    executors::Dict{String, ExecutorSpec}
    edge_groups::Vector{EdgeGroup}
    start_executor_id::String
    output_executor_ids::Vector{String}
    max_iterations::Int
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage}
end

"""
    WorkflowBuilder(; name, start, max_iterations=100) -> WorkflowBuilder

Create a new workflow builder.

# Arguments
- `name::String`: Workflow name (default: "Workflow").
- `start::ExecutorSpec`: The start executor (required).
- `max_iterations::Int`: Maximum supersteps (default: 100).
"""
function WorkflowBuilder(;
    name::String = "Workflow",
    start::ExecutorSpec,
    max_iterations::Int = 100,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
)
    executors = Dict{String, ExecutorSpec}(start.id => start)
    WorkflowBuilder(name, executors, EdgeGroup[], start.id, String[], max_iterations, checkpoint_storage)
end

"""Add an executor to the workflow."""
function add_executor(builder::WorkflowBuilder, spec::ExecutorSpec)::WorkflowBuilder
    if haskey(builder.executors, spec.id)
        throw(WorkflowError("Duplicate executor id: $(spec.id)"))
    end
    builder.executors[spec.id] = spec
    return builder
end

"""
    add_edge(builder, source_id, target_id; condition=nothing) -> WorkflowBuilder

Add a direct edge between two executors. Automatically registers
unknown executor IDs if they appear in already-added executors.
"""
function add_edge(
    builder::WorkflowBuilder,
    source_id::String,
    target_id::String;
    condition::Union{Nothing, Function} = nothing,
    condition_name::Union{Nothing, String} = nothing,
)::WorkflowBuilder
    push!(builder.edge_groups, direct_edge(source_id, target_id; condition=condition, condition_name=condition_name))
    return builder
end

"""
    add_fan_out(builder, source_id, target_ids; selection_func=nothing) -> WorkflowBuilder

Add a fan-out edge: one source broadcasts to multiple targets.
"""
function add_fan_out(
    builder::WorkflowBuilder,
    source_id::String,
    target_ids::Vector{String};
    selection_func::Union{Nothing, Function} = nothing,
)::WorkflowBuilder
    push!(builder.edge_groups, fan_out_edge(source_id, target_ids; selection_func=selection_func))
    return builder
end

"""
    add_fan_in(builder, source_ids, target_id) -> WorkflowBuilder

Add a fan-in edge: multiple sources converge to one target.
"""
function add_fan_in(
    builder::WorkflowBuilder,
    source_ids::Vector{String},
    target_id::String,
)::WorkflowBuilder
    push!(builder.edge_groups, fan_in_edge(source_ids, target_id))
    return builder
end

"""
    add_switch(builder, source_id, cases; default=nothing) -> WorkflowBuilder

Add conditional routing from one source to multiple targets.
`cases` is a vector of `condition => target_id` pairs.
"""
function add_switch(
    builder::WorkflowBuilder,
    source_id::String,
    cases::Vector{Pair{Function, String}};
    default::Union{Nothing, String} = nothing,
)::WorkflowBuilder
    groups = switch_edge(source_id, cases; default=default)
    append!(builder.edge_groups, groups)
    return builder
end

"""
    add_output(builder, executor_id) -> WorkflowBuilder

Mark an executor as an output executor. Its `yield_output` calls
become workflow-level outputs.
"""
function add_output(builder::WorkflowBuilder, executor_id::String)::WorkflowBuilder
    if executor_id ∉ builder.output_executor_ids
        push!(builder.output_executor_ids, executor_id)
    end
    return builder
end

"""
    build(builder::WorkflowBuilder; validate_types=true) -> Workflow

Build and validate the workflow.

# Validation
- Start executor must exist.
- All edge source/target IDs must reference known executors.
- Output executor IDs must reference known executors.
- When `validate_types=true`, checks type compatibility across edges and warns about mismatches.
"""
function build(builder::WorkflowBuilder; validate_types::Bool = true)::Workflow
    # Validate start executor
    if !haskey(builder.executors, builder.start_executor_id)
        throw(WorkflowError("Start executor '$(builder.start_executor_id)' not found"))
    end

    # Validate edges reference known executors
    for group in builder.edge_groups
        for edge in group.edges
            if !haskey(builder.executors, edge.source_id)
                throw(WorkflowError("Edge source '$(edge.source_id)' not found in executors"))
            end
            if !haskey(builder.executors, edge.target_id)
                throw(WorkflowError("Edge target '$(edge.target_id)' not found in executors"))
            end
        end
    end

    # Validate output executors
    for oid in builder.output_executor_ids
        if !haskey(builder.executors, oid)
            throw(WorkflowError("Output executor '$oid' not found in executors"))
        end
    end

    # Optional comprehensive validation
    if validate_types
        vresult = validate_workflow(builder)
        for issue in vresult.issues
            if issue.severity == :error
                @warn "Validation error: $(issue.message)"
            elseif issue.severity == :warning
                @warn "Validation warning: $(issue.message)"
            else
                @info "Validation info: $(issue.message)"
            end
        end
    end

    workflow = Workflow(
        name = builder.name,
        executors = copy(builder.executors),
        edge_groups = copy(builder.edge_groups),
        start_executor_id = builder.start_executor_id,
        output_executor_ids = copy(builder.output_executor_ids),
        max_iterations = builder.max_iterations,
        checkpoint_storage = builder.checkpoint_storage,
    )
    workflow.graph_signature_hash = _workflow_graph_signature_hash(workflow)
    return workflow
end
