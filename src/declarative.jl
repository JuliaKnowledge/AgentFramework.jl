# Declarative workflow loading/saving for AgentFramework.jl
# Enables workflow persistence and sharing via Dict/JSON definitions.

# ── Handler Registry ─────────────────────────────────────────────────────────

"""Global registry mapping handler names to functions."""
const _HANDLER_REGISTRY = Dict{String, Function}()

"""
    register_handler!(name::String, handler::Function)

Register a handler function by name in the global registry.
"""
function register_handler!(name::String, handler::Function)
    _HANDLER_REGISTRY[name] = handler
end

"""
    get_handler(name::String) -> Union{Nothing, Function}

Get a handler by name from the global registry. Returns `nothing` if not found.
"""
function get_handler(name::String)::Union{Nothing, Function}
    get(_HANDLER_REGISTRY, name, nothing)
end

"""
    @register_handler name func

Macro to register a handler function by name.
"""
macro register_handler(name, func)
    quote
        register_handler!($(esc(name)), $(esc(func)))
    end
end

# ── Type Parsing ─────────────────────────────────────────────────────────────

function _parse_type_list(types::Vector)::Vector{DataType}
    [_parse_type_name(string(t)) for t in types]
end

function _parse_type_name(name::String)::DataType
    type_map = Dict(
        "Any" => Any,
        "String" => String,
        "Int" => Int,
        "Int64" => Int64,
        "Float64" => Float64,
        "Bool" => Bool,
        "Dict" => Dict{String, Any},
        "Vector" => Vector{Any},
        "Message" => Message,
    )
    get(type_map, name, Any)
end

# ── Loading from Dict ────────────────────────────────────────────────────────

"""
    workflow_from_dict(definition::Dict{String, Any}; handlers=Dict{String, Function}()) -> Workflow

Load a workflow from a dictionary definition.

Handlers can be provided in the `handlers` argument or looked up from the global registry.
If an executor references a handler name not found in either, loading throws by default.
Set `allow_missing_handlers=true` to explicitly use passthrough handlers instead.

# Example
```julia
definition = Dict{String, Any}(
    "name" => "TextPipeline",
    "max_iterations" => 100,
    "start" => "upper",
    "outputs" => ["reverse"],
    "executors" => [
        Dict("id" => "upper", "description" => "Uppercase text",
             "input_types" => ["String"], "output_types" => ["String"]),
        Dict("id" => "reverse", "description" => "Reverse text",
             "input_types" => ["String"], "output_types" => ["String"]),
    ],
    "edges" => [
        Dict("kind" => "direct", "source" => "upper", "target" => "reverse"),
    ]
)
workflow = workflow_from_dict(definition; handlers=Dict("upper" => my_handler))
```
"""
function workflow_from_dict(
    definition::Dict{String, Any};
    handlers::Dict{String, Function} = Dict{String, Function}(),
    allow_missing_handlers::Bool = false,
)::Workflow
    name = get(definition, "name", "Workflow")
    max_iter = get(definition, "max_iterations", 100)
    start_id = definition["start"]
    output_ids = String[string(o) for o in get(definition, "outputs", String[])]

    # Build executors
    executors = Dict{String, ExecutorSpec}()
    for exec_def in get(definition, "executors", [])
        id = exec_def["id"]
        desc = get(exec_def, "description", "")
        handler_name = get(exec_def, "handler", id)

        # Look up handler: local → global registry → explicit passthrough opt-in
        handler = get(handlers, handler_name, nothing)
        if handler === nothing
            handler = get_handler(handler_name)
        end
        if handler === nothing
            if allow_missing_handlers
                handler = (msg, ctx) -> send_message(ctx, msg)
            else
                throw(WorkflowError("Handler '$handler_name' not found for executor '$id'"))
            end
        end

        input_types = _parse_type_list(get(exec_def, "input_types", ["Any"]))
        output_types = _parse_type_list(get(exec_def, "output_types", ["Any"]))
        yield_types = _parse_type_list(get(exec_def, "yield_types", []))

        executors[id] = ExecutorSpec(
            id=id, description=desc, handler=handler,
            input_types=input_types, output_types=output_types, yield_types=yield_types,
        )
    end

    # Build edge groups
    edge_groups = EdgeGroup[]
    for edge_def in get(definition, "edges", [])
        kind = get(edge_def, "kind", "direct")
        if kind == "direct"
            push!(edge_groups, direct_edge(edge_def["source"], edge_def["target"]))
        elseif kind == "fan_out"
            targets = String[string(t) for t in edge_def["targets"]]
            push!(edge_groups, fan_out_edge(edge_def["source"], targets))
        elseif kind == "fan_in"
            sources = String[string(s) for s in edge_def["sources"]]
            push!(edge_groups, fan_in_edge(sources, edge_def["target"]))
        end
    end

    Workflow(
        name=name, executors=executors, edge_groups=edge_groups,
        start_executor_id=start_id, output_executor_ids=output_ids,
        max_iterations=max_iter,
    )
end

# ── Saving to Dict ───────────────────────────────────────────────────────────

"""
    workflow_to_dict(workflow::Workflow) -> Dict{String, Any}

Serialize a workflow structure to a dictionary. Handlers are stored by name/ID only
(functions cannot be serialized).
"""
function workflow_to_dict(workflow::Workflow)::Dict{String, Any}
    executors = [Dict{String, Any}(
        "id" => spec.id,
        "description" => spec.description,
        "handler" => spec.id,
        "input_types" => [string(t) for t in spec.input_types],
        "output_types" => [string(t) for t in spec.output_types],
        "yield_types" => [string(t) for t in spec.yield_types],
    ) for (_, spec) in sort(collect(workflow.executors), by=first)]

    edges = Dict{String, Any}[]
    for group in workflow.edge_groups
        if group.kind == DIRECT_EDGE
            for edge in group.edges
                push!(edges, Dict{String, Any}(
                    "kind" => "direct",
                    "source" => edge.source_id,
                    "target" => edge.target_id,
                ))
            end
        elseif group.kind == FAN_OUT_EDGE
            src = source_executor_ids(group)[1]
            targets = target_executor_ids(group)
            push!(edges, Dict{String, Any}(
                "kind" => "fan_out",
                "source" => src,
                "targets" => targets,
            ))
        elseif group.kind == FAN_IN_EDGE
            sources = source_executor_ids(group)
            tgt = target_executor_ids(group)[1]
            push!(edges, Dict{String, Any}(
                "kind" => "fan_in",
                "sources" => sources,
                "target" => tgt,
            ))
        end
    end

    Dict{String, Any}(
        "name" => workflow.name,
        "max_iterations" => workflow.max_iterations,
        "start" => workflow.start_executor_id,
        "outputs" => workflow.output_executor_ids,
        "executors" => executors,
        "edges" => edges,
    )
end

# ── JSON Convenience ─────────────────────────────────────────────────────────

"""
    workflow_from_json(json::String; handlers=Dict{String, Function}()) -> Workflow

Load a workflow from a JSON string.
"""
function workflow_from_json(
    json::String;
    handlers::Dict{String, Function} = Dict{String, Function}(),
    allow_missing_handlers::Bool = false,
)::Workflow
    dict = JSON3.read(json, Dict{String, Any})
    workflow_from_dict(dict; handlers=handlers, allow_missing_handlers=allow_missing_handlers)
end

"""
    workflow_to_json(workflow::Workflow) -> String

Save a workflow to a JSON string.
"""
function workflow_to_json(workflow::Workflow)::String
    JSON3.write(workflow_to_dict(workflow))
end

"""
    workflow_from_file(path::String; handlers=Dict{String, Function}()) -> Workflow

Load a workflow from a JSON file.
"""
function workflow_from_file(
    path::String;
    handlers::Dict{String, Function} = Dict{String, Function}(),
    allow_missing_handlers::Bool = false,
)::Workflow
    json = read(path, String)
    workflow_from_json(json; handlers=handlers, allow_missing_handlers=allow_missing_handlers)
end

"""
    workflow_to_file(workflow::Workflow, path::String)

Save a workflow to a JSON file.
"""
function workflow_to_file(workflow::Workflow, path::String)
    write(path, workflow_to_json(workflow))
end
