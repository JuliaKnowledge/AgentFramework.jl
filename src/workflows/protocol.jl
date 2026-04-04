# Protocol introspection for AgentFramework.jl workflows.
# Runtime inspection of executor input/output types and build-time edge type validation.
# Inspired by C#/Python protocol descriptor pattern.

"""
    ProtocolDescriptor

Protocol descriptor capturing what an executor accepts, sends, and yields.

# Fields
- `accepts::Vector{DataType}`: Input types this executor handles.
- `sends::Vector{DataType}`: Output message types.
- `yields::Vector{DataType}`: Workflow output types.
- `accepts_all::Bool`: If true, accepts any input type.
"""
Base.@kwdef struct ProtocolDescriptor
    accepts::Vector{DataType} = DataType[Any]
    sends::Vector{DataType} = DataType[Any]
    yields::Vector{DataType} = DataType[]
    accepts_all::Bool = false
end

function Base.show(io::IO, p::ProtocolDescriptor)
    print(io, "ProtocolDescriptor(accepts=", p.accepts, ", sends=", p.sends, ")")
end

"""
    TypeCompatibilityResult

Result of a type compatibility check between two connected executors.

# Fields
- `compatible::Bool`: Whether the types are compatible.
- `source_id::String`: Source executor ID.
- `target_id::String`: Target executor ID.
- `source_types::Vector{DataType}`: Output types from the source.
- `target_types::Vector{DataType}`: Input types for the target.
- `message::String`: Human-readable description of the result.
"""
Base.@kwdef struct TypeCompatibilityResult
    compatible::Bool = true
    source_id::String = ""
    target_id::String = ""
    source_types::Vector{DataType} = DataType[]
    target_types::Vector{DataType} = DataType[]
    message::String = ""
end

"""
    WorkflowValidationResult

Full validation result for a workflow's type compatibility.

# Fields
- `valid::Bool`: Whether the workflow is valid (no type errors).
- `errors::Vector{TypeCompatibilityResult}`: Type incompatibility errors.
- `warnings::Vector{String}`: Non-fatal warnings.
"""
Base.@kwdef struct WorkflowValidationResult
    valid::Bool = true
    errors::Vector{TypeCompatibilityResult} = TypeCompatibilityResult[]
    warnings::Vector{String} = String[]
end

# ── Protocol Extraction ──────────────────────────────────────────────────────

"""
    get_protocol(spec::ExecutorSpec) -> ProtocolDescriptor

Extract a protocol descriptor from an ExecutorSpec, mapping its declared
input/output/yield types to the protocol representation.
"""
function get_protocol(spec::ExecutorSpec)::ProtocolDescriptor
    ProtocolDescriptor(
        accepts = copy(spec.input_types),
        sends = copy(spec.output_types),
        yields = copy(spec.yield_types),
        accepts_all = Any in spec.input_types,
    )
end

# ── Type Checking ────────────────────────────────────────────────────────────

"""
    can_handle(spec::ExecutorSpec, ::Type{T}) -> Bool

Check if an executor can handle input of type `T`.
"""
function can_handle(spec::ExecutorSpec, ::Type{T})::Bool where T
    can_handle(get_protocol(spec), T)
end

"""
    can_handle(proto::ProtocolDescriptor, ::Type{T}) -> Bool

Check if a protocol descriptor accepts input of type `T`.
Returns true if `T` is a subtype of any accepted type, or if `accepts_all` is set.
"""
function can_handle(proto::ProtocolDescriptor, ::Type{T})::Bool where T
    proto.accepts_all && return true
    isempty(proto.accepts) && return false
    return any(a -> T <: a, proto.accepts)
end

"""
    can_output(spec::ExecutorSpec, ::Type{T}) -> Bool

Check if an executor can output type `T`.
"""
function can_output(spec::ExecutorSpec, ::Type{T})::Bool where T
    can_output(get_protocol(spec), T)
end

"""
    can_output(proto::ProtocolDescriptor, ::Type{T}) -> Bool

Check if a protocol descriptor can send type `T`.
Returns true if `T` is a subtype of any send type, or if `Any` is in sends.
"""
function can_output(proto::ProtocolDescriptor, ::Type{T})::Bool where T
    isempty(proto.sends) && return false
    return any(s -> T <: s, proto.sends)
end

# ── Type Compatibility ───────────────────────────────────────────────────────

"""
    check_type_compatibility(source::ExecutorSpec, target::ExecutorSpec) -> TypeCompatibilityResult

Check whether the output types of `source` are compatible with the input types
of `target`. Uses Julia's type system for subtype checks.

# Compatibility rules
- If target accepts `Any`, always compatible.
- If source sends `Any`, always compatible (can't validate further).
- Otherwise, at least one source send type must be a subtype of at least one target accept type.
"""
function check_type_compatibility(source::ExecutorSpec, target::ExecutorSpec)::TypeCompatibilityResult
    source_sends = source.output_types
    target_accepts = target.input_types

    # Target accepts Any → always compatible
    if Any in target_accepts
        return TypeCompatibilityResult(
            compatible = true,
            source_id = source.id,
            target_id = target.id,
            source_types = copy(source_sends),
            target_types = copy(target_accepts),
            message = "Compatible: target '$(target.id)' accepts Any",
        )
    end

    # Source sends Any → compatible (can't validate)
    if Any in source_sends
        return TypeCompatibilityResult(
            compatible = true,
            source_id = source.id,
            target_id = target.id,
            source_types = copy(source_sends),
            target_types = copy(target_accepts),
            message = "Compatible: source '$(source.id)' sends Any (unvalidatable)",
        )
    end

    # Check for at least one overlapping type (subtype relationship)
    has_overlap = any(s -> any(a -> s <: a, target_accepts), source_sends)

    if has_overlap
        return TypeCompatibilityResult(
            compatible = true,
            source_id = source.id,
            target_id = target.id,
            source_types = copy(source_sends),
            target_types = copy(target_accepts),
            message = "Compatible: '$(source.id)' → '$(target.id)'",
        )
    else
        return TypeCompatibilityResult(
            compatible = false,
            source_id = source.id,
            target_id = target.id,
            source_types = copy(source_sends),
            target_types = copy(target_accepts),
            message = "Incompatible: '$(source.id)' sends $(source_sends) but '$(target.id)' accepts $(target_accepts)",
        )
    end
end

# ── Workflow Validation ──────────────────────────────────────────────────────

"""
    validate_workflow_types(workflow::Workflow) -> WorkflowValidationResult

Validate all edges in a workflow for type compatibility between connected executors.
"""
function validate_workflow_types(workflow::Workflow)::WorkflowValidationResult
    errors = TypeCompatibilityResult[]
    warnings = String[]

    for group in workflow.edge_groups
        for edge in group.edges
            source = get(workflow.executors, edge.source_id, nothing)
            target = get(workflow.executors, edge.target_id, nothing)

            if source === nothing
                push!(warnings, "Edge references unknown source '$(edge.source_id)'")
                continue
            end
            if target === nothing
                push!(warnings, "Edge references unknown target '$(edge.target_id)'")
                continue
            end

            result = check_type_compatibility(source, target)
            if !result.compatible
                push!(errors, result)
            end
        end
    end

    WorkflowValidationResult(
        valid = isempty(errors),
        errors = errors,
        warnings = warnings,
    )
end

"""
    validate_workflow_types(builder::WorkflowBuilder) -> WorkflowValidationResult

Validate all edges in a WorkflowBuilder for type compatibility.
"""
function validate_workflow_types(builder::WorkflowBuilder)::WorkflowValidationResult
    errors = TypeCompatibilityResult[]
    warnings = String[]

    for group in builder.edge_groups
        for edge in group.edges
            source = get(builder.executors, edge.source_id, nothing)
            target = get(builder.executors, edge.target_id, nothing)

            if source === nothing
                push!(warnings, "Edge references unknown source '$(edge.source_id)'")
                continue
            end
            if target === nothing
                push!(warnings, "Edge references unknown target '$(edge.target_id)'")
                continue
            end

            result = check_type_compatibility(source, target)
            if !result.compatible
                push!(errors, result)
            end
        end
    end

    WorkflowValidationResult(
        valid = isempty(errors),
        errors = errors,
        warnings = warnings,
    )
end

# ── Pretty Printing ──────────────────────────────────────────────────────────

"""
    describe_protocol(spec::ExecutorSpec) -> String

Pretty-print protocol information for an executor.
"""
function describe_protocol(spec::ExecutorSpec)::String
    describe_protocol(get_protocol(spec), spec.id)
end

"""
    describe_protocol(proto::ProtocolDescriptor) -> String

Pretty-print protocol information from a descriptor.
"""
function describe_protocol(proto::ProtocolDescriptor)::String
    describe_protocol(proto, "unknown")
end

function describe_protocol(proto::ProtocolDescriptor, id::String)::String
    lines = String[]
    push!(lines, "Protocol for '$id':")
    accepts_str = proto.accepts_all ? "Any (accepts all)" : join(string.(proto.accepts), ", ")
    push!(lines, "  Accepts: $accepts_str")
    sends_str = isempty(proto.sends) ? "nothing" : join(string.(proto.sends), ", ")
    push!(lines, "  Sends:   $sends_str")
    yields_str = isempty(proto.yields) ? "nothing" : join(string.(proto.yields), ", ")
    push!(lines, "  Yields:  $yields_str")
    return join(lines, "\n")
end

# ── Comprehensive Workflow Validation ────────────────────────────────────────

"""
    ValidationCheck

Enum of validation checks that can be performed on a workflow.
"""
@enum ValidationCheck begin
    CHECK_TYPE_COMPATIBILITY
    CHECK_EDGE_DUPLICATION
    CHECK_EXECUTOR_DUPLICATION
    CHECK_GRAPH_CONNECTIVITY
    CHECK_SELF_LOOPS
    CHECK_OUTPUT_EXECUTORS
    CHECK_DEAD_ENDS
end

"""All available validation checks."""
const ALL_CHECKS = ValidationCheck[
    CHECK_TYPE_COMPATIBILITY,
    CHECK_EDGE_DUPLICATION,
    CHECK_EXECUTOR_DUPLICATION,
    CHECK_GRAPH_CONNECTIVITY,
    CHECK_SELF_LOOPS,
    CHECK_OUTPUT_EXECUTORS,
    CHECK_DEAD_ENDS,
]

"""
    ValidationIssue

A single issue found during workflow validation.

# Fields
- `check::ValidationCheck`: Which check produced this issue.
- `severity::Symbol`: `:error`, `:warning`, or `:info`.
- `message::String`: Human-readable description.
- `executor_ids::Vector{String}`: Related executor IDs.
- `edge_ids::Vector{String}`: Related edge group IDs.
"""
Base.@kwdef struct ValidationIssue
    check::ValidationCheck
    severity::Symbol = :error
    message::String
    executor_ids::Vector{String} = String[]
    edge_ids::Vector{String} = String[]
end

"""
    FullValidationResult

Comprehensive validation result containing all issues found.

# Fields
- `valid::Bool`: Whether the workflow has no `:error` severity issues.
- `issues::Vector{ValidationIssue}`: All issues found.
"""
Base.@kwdef struct FullValidationResult
    valid::Bool = true
    issues::Vector{ValidationIssue} = ValidationIssue[]
end

# ── Individual Validation Checks ─────────────────────────────────────────────

function _check_edge_duplication(edge_groups::Vector{EdgeGroup})::Vector{ValidationIssue}
    issues = ValidationIssue[]
    seen_ids = Dict{String, Int}()
    for group in edge_groups
        seen_ids[group.id] = get(seen_ids, group.id, 0) + 1
    end
    for (id, count) in seen_ids
        if count > 1
            push!(issues, ValidationIssue(
                check = CHECK_EDGE_DUPLICATION,
                severity = :error,
                message = "Duplicate edge group ID '$id' appears $count times",
                edge_ids = [id],
            ))
        end
    end
    return issues
end

function _check_executor_duplication(executors::Dict{String, ExecutorSpec})::Vector{ValidationIssue}
    # Dict keys are unique by definition, so no duplicates possible in the dict itself.
    # This check exists for completeness — the builder already prevents duplicates.
    return ValidationIssue[]
end

function _check_graph_connectivity(
    executors::Dict{String, ExecutorSpec},
    edge_groups::Vector{EdgeGroup},
    start_executor_id::String,
)::Vector{ValidationIssue}
    issues = ValidationIssue[]
    all_ids = Set(keys(executors))
    isempty(all_ids) && return issues

    # BFS from start executor
    visited = Set{String}()
    queue = String[start_executor_id]
    # Build adjacency: source → set of targets
    adj = Dict{String, Set{String}}()
    for group in edge_groups
        for edge in group.edges
            targets = get!(adj, edge.source_id, Set{String}())
            push!(targets, edge.target_id)
        end
    end

    while !isempty(queue)
        current = popfirst!(queue)
        current in visited && continue
        push!(visited, current)
        for neighbor in get(adj, current, Set{String}())
            if neighbor ∉ visited
                push!(queue, neighbor)
            end
        end
    end

    unreachable = setdiff(all_ids, visited)
    for uid in sort(collect(unreachable))
        push!(issues, ValidationIssue(
            check = CHECK_GRAPH_CONNECTIVITY,
            severity = :warning,
            message = "Executor '$uid' is not reachable from start executor '$start_executor_id'",
            executor_ids = [uid],
        ))
    end
    return issues
end

function _check_self_loops(edge_groups::Vector{EdgeGroup})::Vector{ValidationIssue}
    issues = ValidationIssue[]
    for group in edge_groups
        for edge in group.edges
            if edge.source_id == edge.target_id
                push!(issues, ValidationIssue(
                    check = CHECK_SELF_LOOPS,
                    severity = :warning,
                    message = "Self-loop detected: executor '$(edge.source_id)' has an edge to itself in edge group '$(group.id)'",
                    executor_ids = [edge.source_id],
                    edge_ids = [group.id],
                ))
            end
        end
    end
    return issues
end

function _check_output_executors(
    executors::Dict{String, ExecutorSpec},
    output_executor_ids::Vector{String},
)::Vector{ValidationIssue}
    issues = ValidationIssue[]
    for oid in output_executor_ids
        spec = get(executors, oid, nothing)
        spec === nothing && continue
        if isempty(spec.yield_types) || spec.yield_types == DataType[Any]
            push!(issues, ValidationIssue(
                check = CHECK_OUTPUT_EXECUTORS,
                severity = :warning,
                message = "Output executor '$oid' has unspecified yield_types ($(isempty(spec.yield_types) ? "empty" : "[Any]"))",
                executor_ids = [oid],
            ))
        end
    end
    return issues
end

function _check_dead_ends(
    executors::Dict{String, ExecutorSpec},
    edge_groups::Vector{EdgeGroup},
    output_executor_ids::Vector{String},
)::Vector{ValidationIssue}
    issues = ValidationIssue[]
    length(executors) <= 1 && return issues

    # Collect all executor IDs that appear as edge sources
    sources_with_outgoing = Set{String}()
    for group in edge_groups
        for edge in group.edges
            push!(sources_with_outgoing, edge.source_id)
        end
    end

    output_set = Set(output_executor_ids)
    for eid in sort(collect(keys(executors)))
        if eid ∉ sources_with_outgoing && eid ∉ output_set
            push!(issues, ValidationIssue(
                check = CHECK_DEAD_ENDS,
                severity = :info,
                message = "Executor '$eid' has no outgoing edges and is not an output executor",
                executor_ids = [eid],
            ))
        end
    end
    return issues
end

function _check_type_compat_issues(
    executors::Dict{String, ExecutorSpec},
    edge_groups::Vector{EdgeGroup},
)::Vector{ValidationIssue}
    issues = ValidationIssue[]
    for group in edge_groups
        for edge in group.edges
            source = get(executors, edge.source_id, nothing)
            target = get(executors, edge.target_id, nothing)

            if source === nothing
                push!(issues, ValidationIssue(
                    check = CHECK_TYPE_COMPATIBILITY,
                    severity = :warning,
                    message = "Edge references unknown source '$(edge.source_id)'",
                    edge_ids = [group.id],
                ))
                continue
            end
            if target === nothing
                push!(issues, ValidationIssue(
                    check = CHECK_TYPE_COMPATIBILITY,
                    severity = :warning,
                    message = "Edge references unknown target '$(edge.target_id)'",
                    edge_ids = [group.id],
                ))
                continue
            end

            result = check_type_compatibility(source, target)
            if !result.compatible
                push!(issues, ValidationIssue(
                    check = CHECK_TYPE_COMPATIBILITY,
                    severity = :error,
                    message = result.message,
                    executor_ids = [result.source_id, result.target_id],
                    edge_ids = [group.id],
                ))
            end
        end
    end
    return issues
end

# ── Main Validation Entry Points ─────────────────────────────────────────────

"""
    validate_workflow(workflow::Workflow; checks=ALL_CHECKS) -> FullValidationResult

Run comprehensive validation on a built workflow.
"""
function validate_workflow(
    workflow::Workflow;
    checks::Vector{ValidationCheck} = ALL_CHECKS,
)::FullValidationResult
    _run_validation(
        workflow.executors,
        workflow.edge_groups,
        workflow.start_executor_id,
        workflow.output_executor_ids,
        checks,
    )
end

"""
    validate_workflow(builder::WorkflowBuilder; checks=ALL_CHECKS) -> FullValidationResult

Run comprehensive validation on a workflow builder before building.
"""
function validate_workflow(
    builder::WorkflowBuilder;
    checks::Vector{ValidationCheck} = ALL_CHECKS,
)::FullValidationResult
    _run_validation(
        builder.executors,
        builder.edge_groups,
        builder.start_executor_id,
        builder.output_executor_ids,
        checks,
    )
end

function _run_validation(
    executors::Dict{String, ExecutorSpec},
    edge_groups::Vector{EdgeGroup},
    start_executor_id::String,
    output_executor_ids::Vector{String},
    checks::Vector{ValidationCheck},
)::FullValidationResult
    issues = ValidationIssue[]

    if CHECK_EDGE_DUPLICATION in checks
        append!(issues, _check_edge_duplication(edge_groups))
    end
    if CHECK_EXECUTOR_DUPLICATION in checks
        append!(issues, _check_executor_duplication(executors))
    end
    if CHECK_GRAPH_CONNECTIVITY in checks
        append!(issues, _check_graph_connectivity(executors, edge_groups, start_executor_id))
    end
    if CHECK_SELF_LOOPS in checks
        append!(issues, _check_self_loops(edge_groups))
    end
    if CHECK_OUTPUT_EXECUTORS in checks
        append!(issues, _check_output_executors(executors, output_executor_ids))
    end
    if CHECK_DEAD_ENDS in checks
        append!(issues, _check_dead_ends(executors, edge_groups, output_executor_ids))
    end
    if CHECK_TYPE_COMPATIBILITY in checks
        append!(issues, _check_type_compat_issues(executors, edge_groups))
    end

    has_errors = any(i -> i.severity == :error, issues)
    FullValidationResult(valid = !has_errors, issues = issues)
end
