# Declarative agent/workflow loading for AgentFramework.jl.
# Provides JSON/YAML support for workflow definitions and registry-based agent loading.

# ── Registries ─────────────────────────────────────────────────────────────────

"""Global registry mapping workflow handler names to functions."""
const _HANDLER_REGISTRY = Dict{String, Function}()

"""Global registry mapping declarative tool names to concrete FunctionTool instances."""
const _TOOL_REGISTRY = Dict{String, FunctionTool}()

"""Global registry mapping declarative client names to concrete chat clients."""
const _CLIENT_REGISTRY = Dict{String, AbstractChatClient}()

"""Global registry mapping declarative context-provider names to concrete providers."""
const _CONTEXT_PROVIDER_REGISTRY = Dict{String, Any}()

const _YAML_EXTENSIONS = Set([".yaml", ".yml"])
const _JSON_EXTENSIONS = Set([".json"])

const _DECLARATIVE_CLIENT_TYPE_ALIASES = Dict{String, String}(
    "ollama" => "OllamaChatClient",
    "ollamachatclient" => "OllamaChatClient",
    "openai" => "OpenAIChatClient",
    "openaichat" => "OpenAIChatClient",
    "openaichatclient" => "OpenAIChatClient",
    "openairesponses" => "OpenAIChatClient",
    "azureopenai" => "AzureOpenAIChatClient",
    "azureopenaichat" => "AzureOpenAIChatClient",
    "azureopenaichatclient" => "AzureOpenAIChatClient",
    "azureopenairesponses" => "AzureOpenAIChatClient",
    "anthropic" => "AnthropicChatClient",
    "anthropicchat" => "AnthropicChatClient",
    "anthropicchatclient" => "AnthropicChatClient",
    "foundry" => "FoundryChatClient",
    "foundrychat" => "FoundryChatClient",
    "foundrychatclient" => "FoundryChatClient",
)

# ── Registry Helpers ───────────────────────────────────────────────────────────

"""
    register_handler!(name::String, handler::Function)

Register a workflow handler function by name in the global registry.
"""
function register_handler!(name::String, handler::Function)
    _HANDLER_REGISTRY[name] = handler
end

"""
    get_handler(name::String) -> Union{Nothing, Function}

Get a workflow handler by name from the global registry.
"""
function get_handler(name::String)::Union{Nothing, Function}
    get(_HANDLER_REGISTRY, name, nothing)
end

"""
    @register_handler name func

Macro to register a workflow handler function by name.
"""
macro register_handler(name, func)
    quote
        register_handler!($(esc(name)), $(esc(func)))
    end
end

"""
    register_tool!(name::String, tool::FunctionTool)

Register a declarative tool reference by name.
"""
function register_tool!(name::String, tool::FunctionTool)
    _TOOL_REGISTRY[name] = tool
end

"""
    get_tool(name::String) -> Union{Nothing, FunctionTool}

Get a declarative tool by name from the global registry.
"""
function get_tool(name::String)::Union{Nothing, FunctionTool}
    get(_TOOL_REGISTRY, name, nothing)
end

"""
    register_client!(name::String, client::AbstractChatClient)

Register a declarative chat client reference by name.
"""
function register_client!(name::String, client::AbstractChatClient)
    _CLIENT_REGISTRY[name] = client
end

"""
    get_client(name::String) -> Union{Nothing, AbstractChatClient}

Get a declarative chat client by name from the global registry.
"""
function get_client(name::String)::Union{Nothing, AbstractChatClient}
    get(_CLIENT_REGISTRY, name, nothing)
end

"""
    register_context_provider!(name::String, provider)

Register a declarative context provider reference by name.
"""
function register_context_provider!(name::String, provider)
    _CONTEXT_PROVIDER_REGISTRY[name] = provider
end

"""
    get_context_provider(name::String)

Get a declarative context provider by name from the global registry.
"""
function get_context_provider(name::String)
    get(_CONTEXT_PROVIDER_REGISTRY, name, nothing)
end

function _lookup_registry(registry::AbstractDict, name::String)
    for (key, value) in pairs(registry)
        if string(key) == name
            return true, value
        end
    end
    return false, nothing
end

function _registry_name(registry::AbstractDict, value)::Union{Nothing, String}
    for (key, entry) in pairs(registry)
        if entry === value
            return string(key)
        end
    end
    return nothing
end

# ── Materialization / Parsing Helpers ──────────────────────────────────────────

function _materialize_declarative(value)
    if value isa AbstractDict
        return Dict{String, Any}(string(k) => _materialize_declarative(v) for (k, v) in pairs(value))
    elseif value isa AbstractVector
        return Any[_materialize_declarative(v) for v in value]
    end
    return value
end

function _require_declarative_dict(value, what::String)::Dict{String, Any}
    value isa AbstractDict || throw(DeclarativeError("$what must be a mapping."))
    return _materialize_declarative(value)
end

function _ensure_declarative_vector(value)
    value === nothing && return Any[]
    if value isa AbstractVector
        return Any[_materialize_declarative(v) for v in value]
    end
    return Any[_materialize_declarative(value)]
end

function _string_vector(value)::Vector{String}
    value === nothing && return String[]
    if value isa AbstractVector
        return [string(v) for v in value]
    end
    return [string(value)]
end

function _maybe_put!(dict::Dict{String, Any}, key::String, value)
    if value === nothing
        return dict
    elseif value isa AbstractString && isempty(value)
        return dict
    elseif value isa AbstractVector && isempty(value)
        return dict
    elseif value isa AbstractDict && isempty(value)
        return dict
    end

    dict[key] = _materialize_declarative(value)
    return dict
end

function _apply_aliases(mapping::Dict{String, Any}, aliases::Dict{String, String})::Dict{String, Any}
    normalized = Dict{String, Any}()
    for (key, value) in mapping
        normalized[get(aliases, key, key)] = value
    end
    return normalized
end

function _first_non_whitespace(text::AbstractString)::Union{Nothing, Char}
    for ch in text
        if !isspace(ch)
            return ch
        end
    end
    return nothing
end

function _detect_declarative_input_format(path::String, text::AbstractString)::Symbol
    ext = lowercase(splitext(path)[2])
    ext in _YAML_EXTENSIONS && return :yaml
    ext in _JSON_EXTENSIONS && return :json

    first_char = _first_non_whitespace(text)
    return first_char in ('{', '[') ? :json : :yaml
end

function _output_format_for_path(path::String)::Symbol
    ext = lowercase(splitext(path)[2])
    return ext in _YAML_EXTENSIONS ? :yaml : :json
end

function _load_json_definition(json::AbstractString)::Dict{String, Any}
    try
        value = JSON3.read(String(json), Any)
        return _require_declarative_dict(_materialize_declarative(value), "Declarative JSON")
    catch err
        throw(DeclarativeError("Failed to parse declarative JSON.", err))
    end
end

function _load_yaml_definition(yaml::AbstractString)::Dict{String, Any}
    try
        value = YAML.load(String(yaml))
        value === nothing && return Dict{String, Any}()
        return _require_declarative_dict(value, "Declarative YAML")
    catch err
        throw(DeclarativeError("Failed to parse declarative YAML.", err))
    end
end

function _read_declarative_definition(path::String)::Tuple{Dict{String, Any}, Symbol}
    text = read(path, String)
    format = _detect_declarative_input_format(path, text)
    if format == :yaml
        return _load_yaml_definition(text), :yaml
    end
    return _load_json_definition(text), :json
end

function _write_declarative_definition(definition::Dict{String, Any}, format::Symbol)::String
    if format == :yaml
        return YAML.write(definition)
    elseif format == :json
        return JSON3.write(definition)
    end
    throw(DeclarativeError("Unsupported declarative format '$format'."))
end

function _write_declarative_file(definition::Dict{String, Any}, path::String)
    write(path, _write_declarative_definition(definition, _output_format_for_path(path)))
end

# ── Type Parsing ───────────────────────────────────────────────────────────────

function _split_type_parameters(text::String)::Vector{String}
    isempty(text) && return String[]

    params = String[]
    depth = 0
    start_idx = firstindex(text)
    idx = firstindex(text)

    while idx <= lastindex(text)
        ch = text[idx]
        if ch == '{'
            depth += 1
        elseif ch == '}'
            depth -= 1
        elseif ch == ',' && depth == 0
            push!(params, strip(text[start_idx:prevind(text, idx)]))
            start_idx = nextind(text, idx)
        end
        idx = nextind(text, idx)
    end

    push!(params, strip(text[start_idx:end]))
    return filter(!isempty, params)
end

function _parse_type_list(types)::Vector{DataType}
    [_parse_type_name(string(t)) for t in _ensure_declarative_vector(types)]
end

function _parse_type_name(name::String)::DataType
    stripped = replace(strip(name), " " => "")
    isempty(stripped) && return Any

    type_map = Dict(
        "Any" => Any,
        "Nothing" => Nothing,
        "String" => String,
        "Int" => Int,
        "Int64" => Int64,
        "Int32" => Int32,
        "Float64" => Float64,
        "Float32" => Float32,
        "Bool" => Bool,
        "Dict" => Dict{String, Any},
        "Dict{String,Any}" => Dict{String, Any},
        "Dict{String,String}" => Dict{String, String},
        "Vector" => Vector{Any},
        "Vector{Any}" => Vector{Any},
        "Vector{String}" => Vector{String},
        "Vector{Message}" => Vector{Message},
        "Message" => Message,
    )
    haskey(type_map, stripped) && return type_map[stripped]

    if startswith(stripped, "Vector{") && endswith(stripped, "}")
        inner = _parse_type_name(stripped[8:end-1])
        return Vector{inner}
    end

    if startswith(stripped, "Dict{") && endswith(stripped, "}")
        params = _split_type_parameters(stripped[6:end-1])
        if length(params) == 2
            key_type = _parse_type_name(params[1])
            value_type = _parse_type_name(params[2])
            return Dict{key_type, value_type}
        end
    end

    return Any
end

# ── Workflow Loading / Saving ──────────────────────────────────────────────────

function _resolve_named_callable(spec, handlers::AbstractDict, what::String)::Function
    if spec isa Function
        return spec
    end

    name = string(spec)
    found, value = _lookup_registry(handlers, name)
    if !found
        found, value = _lookup_registry(_HANDLER_REGISTRY, name)
    end
    found || throw(WorkflowError("$what '$name' not found."))
    value isa Function || throw(WorkflowError("$what '$name' must resolve to a function."))
    return value
end

function _normalize_executor_definitions(raw_definitions)
    raw_definitions === nothing && return Dict{String, Any}[]

    if raw_definitions isa AbstractVector
        return [_require_declarative_dict(def, "Executor definition") for def in raw_definitions]
    elseif raw_definitions isa AbstractDict
        definitions = Dict{String, Any}[]
        for (key, value) in pairs(raw_definitions)
            definition = _require_declarative_dict(value, "Executor definition")
            definition["id"] = get(definition, "id", string(key))
            push!(definitions, definition)
        end
        return definitions
    end

    throw(WorkflowError("'executors' must be a list or mapping."))
end

"""
    workflow_from_dict(definition; handlers=Dict(), allow_missing_handlers=false) -> Workflow

Load a workflow from a declarative dictionary definition.
"""
function workflow_from_dict(
    definition::AbstractDict;
    handlers::AbstractDict = Dict{String, Function}(),
    allow_missing_handlers::Bool = false,
)::Workflow
    materialized = _materialize_declarative(definition)
    name = get(materialized, "name", "Workflow")
    max_iter = Int(get(materialized, "max_iterations", 100))
    start_id = string(materialized["start"])
    output_ids = _string_vector(get(materialized, "outputs", String[]))

    executors = Dict{String, ExecutorSpec}()
    for exec_def in _normalize_executor_definitions(get(materialized, "executors", nothing))
        id = string(exec_def["id"])
        description = string(get(exec_def, "description", ""))
        handler_spec = get(exec_def, "handler", id)

        handler = if handler_spec isa Function
            handler_spec
        else
            found, value = _lookup_registry(handlers, string(handler_spec))
            if !found
                found, value = _lookup_registry(_HANDLER_REGISTRY, string(handler_spec))
            end

            if !found
                if allow_missing_handlers
                    (msg, ctx) -> send_message(ctx, msg)
                else
                    throw(WorkflowError("Handler '$(handler_spec)' not found for executor '$id'."))
                end
            else
                value isa Function || throw(WorkflowError("Handler '$(handler_spec)' must resolve to a function."))
                value
            end
        end

        executors[id] = ExecutorSpec(
            id = id,
            description = description,
            handler = handler,
            input_types = _parse_type_list(get(exec_def, "input_types", ["Any"])),
            output_types = _parse_type_list(get(exec_def, "output_types", ["Any"])),
            yield_types = _parse_type_list(get(exec_def, "yield_types", Any[])),
        )
    end

    edge_groups = EdgeGroup[]
    for raw_edge in _ensure_declarative_vector(get(materialized, "edges", Any[]))
        edge_def = _require_declarative_dict(raw_edge, "Edge definition")
        kind = lowercase(string(get(edge_def, "kind", "direct")))

        if kind == "direct"
            condition = haskey(edge_def, "condition") ?
                _resolve_named_callable(edge_def["condition"], handlers, "Condition") :
                nothing
            condition_name = if haskey(edge_def, "condition_name")
                string(edge_def["condition_name"])
            elseif get(edge_def, "condition", nothing) isa AbstractString
                string(edge_def["condition"])
            else
                nothing
            end
            push!(
                edge_groups,
                direct_edge(
                    string(edge_def["source"]),
                    string(edge_def["target"]);
                    condition = condition,
                    condition_name = condition_name,
                ),
            )
        elseif kind == "fan_out"
            selection_func = haskey(edge_def, "selection_func") ?
                _resolve_named_callable(edge_def["selection_func"], handlers, "Selection function") :
                nothing
            push!(
                edge_groups,
                fan_out_edge(
                    string(edge_def["source"]),
                    _string_vector(edge_def["targets"]);
                    selection_func = selection_func,
                ),
            )
        elseif kind == "fan_in"
            push!(
                edge_groups,
                fan_in_edge(_string_vector(edge_def["sources"]), string(edge_def["target"])),
            )
        elseif kind == "switch"
            cases = Pair{Function, String}[]
            for raw_case in _ensure_declarative_vector(edge_def["cases"])
                case_def = _require_declarative_dict(raw_case, "Switch case")
                condition = _resolve_named_callable(case_def["condition"], handlers, "Switch condition")
                push!(cases, condition => string(case_def["target"]))
            end
            default_target = haskey(edge_def, "default") ? string(edge_def["default"]) : nothing
            append!(edge_groups, switch_edge(string(edge_def["source"]), cases; default = default_target))
        else
            throw(WorkflowError("Unsupported declarative edge kind '$kind'."))
        end
    end

    return Workflow(
        name = string(name),
        executors = executors,
        edge_groups = edge_groups,
        start_executor_id = start_id,
        output_executor_ids = output_ids,
        max_iterations = max_iter,
    )
end

"""
    workflow_to_dict(workflow::Workflow) -> Dict{String, Any}

Serialize a workflow structure to a declarative dictionary.
"""
function workflow_to_dict(workflow::Workflow)::Dict{String, Any}
    executors = Dict{String, Any}[]
    for (_, spec) in sort(collect(workflow.executors), by = first)
        handler_name = something(_registry_name(_HANDLER_REGISTRY, spec.handler), spec.id)
        push!(
            executors,
            Dict{String, Any}(
                "id" => spec.id,
                "description" => spec.description,
                "handler" => handler_name,
                "input_types" => [string(t) for t in spec.input_types],
                "output_types" => [string(t) for t in spec.output_types],
                "yield_types" => [string(t) for t in spec.yield_types],
            ),
        )
    end

    edges = Dict{String, Any}[]
    for group in workflow.edge_groups
        if group.kind == DIRECT_EDGE
            for edge in group.edges
                edge_dict = Dict{String, Any}(
                    "kind" => "direct",
                    "source" => edge.source_id,
                    "target" => edge.target_id,
                )
                if edge.condition !== nothing
                    condition_name = something(
                        _registry_name(_HANDLER_REGISTRY, edge.condition),
                        edge.condition_name,
                    )
                    condition_name !== nothing && (edge_dict["condition"] = condition_name)
                end
                edge.condition_name !== nothing && (edge_dict["condition_name"] = edge.condition_name)
                push!(edges, edge_dict)
            end
        elseif group.kind == FAN_OUT_EDGE
            edge_dict = Dict{String, Any}(
                "kind" => "fan_out",
                "source" => only(source_executor_ids(group)),
                "targets" => target_executor_ids(group),
            )
            if group.selection_func !== nothing
                selection_name = _registry_name(_HANDLER_REGISTRY, group.selection_func)
                selection_name !== nothing && (edge_dict["selection_func"] = selection_name)
            end
            push!(edges, edge_dict)
        elseif group.kind == FAN_IN_EDGE
            push!(
                edges,
                Dict{String, Any}(
                    "kind" => "fan_in",
                    "sources" => source_executor_ids(group),
                    "target" => only(target_executor_ids(group)),
                ),
            )
        end
    end

    return Dict{String, Any}(
        "kind" => "Workflow",
        "name" => workflow.name,
        "max_iterations" => workflow.max_iterations,
        "start" => workflow.start_executor_id,
        "outputs" => workflow.output_executor_ids,
        "executors" => executors,
        "edges" => edges,
    )
end

"""
    workflow_from_json(json; handlers=Dict(), allow_missing_handlers=false) -> Workflow

Load a workflow from a JSON string.
"""
function workflow_from_json(
    json::AbstractString;
    handlers::AbstractDict = Dict{String, Function}(),
    allow_missing_handlers::Bool = false,
)::Workflow
    definition = _load_json_definition(json)
    return workflow_from_dict(
        definition;
        handlers = handlers,
        allow_missing_handlers = allow_missing_handlers,
    )
end

"""
    workflow_to_json(workflow::Workflow) -> String

Save a workflow to a JSON string.
"""
function workflow_to_json(workflow::Workflow)::String
    JSON3.write(workflow_to_dict(workflow))
end

"""
    workflow_from_yaml(yaml; handlers=Dict(), allow_missing_handlers=false) -> Workflow

Load a workflow from a YAML string.
"""
function workflow_from_yaml(
    yaml::AbstractString;
    handlers::AbstractDict = Dict{String, Function}(),
    allow_missing_handlers::Bool = false,
)::Workflow
    definition = _load_yaml_definition(yaml)
    return workflow_from_dict(
        definition;
        handlers = handlers,
        allow_missing_handlers = allow_missing_handlers,
    )
end

"""
    workflow_to_yaml(workflow::Workflow) -> String

Save a workflow to a YAML string.
"""
function workflow_to_yaml(workflow::Workflow)::String
    YAML.write(workflow_to_dict(workflow))
end

"""
    workflow_from_file(path; handlers=Dict(), allow_missing_handlers=false) -> Workflow

Load a workflow from a JSON or YAML file.
"""
function workflow_from_file(
    path::String;
    handlers::AbstractDict = Dict{String, Function}(),
    allow_missing_handlers::Bool = false,
)::Workflow
    definition, _ = _read_declarative_definition(path)
    return workflow_from_dict(
        definition;
        handlers = handlers,
        allow_missing_handlers = allow_missing_handlers,
    )
end

"""
    workflow_to_file(workflow::Workflow, path::String)

Save a workflow to a JSON or YAML file based on its extension.
"""
function workflow_to_file(workflow::Workflow, path::String)
    _write_declarative_file(workflow_to_dict(workflow), path)
end

# ── Declarative Agent Loading / Saving ─────────────────────────────────────────

function _resolve_named_component(name::String, local_registry::AbstractDict, global_registry::AbstractDict, what::String)
    found, value = _lookup_registry(local_registry, name)
    if !found
        found, value = _lookup_registry(global_registry, name)
    end
    found || throw(DeclarativeError("$what '$name' not found in local or global registry."))
    return value
end

function _normalize_client_type_token(name::String)::String
    return lowercase(replace(strip(name), "." => "", "_" => "", "-" => ""))
end

function _resolve_client_type_name(name::String)::String
    token = _normalize_client_type_token(name)
    if haskey(_DECLARATIVE_CLIENT_TYPE_ALIASES, token)
        return _DECLARATIVE_CLIENT_TYPE_ALIASES[token]
    end

    if isdefined(@__MODULE__, Symbol(name))
        return name
    end

    throw(
        DeclarativeError(
            "Unsupported declarative client type '$name'. Register a client instance and reference it by name, or use a built-in provider alias.",
        ),
    )
end

function _resolve_client_type(name::String)::DataType
    resolved_name = _resolve_client_type_name(name)
    client_type = getfield(@__MODULE__, Symbol(resolved_name))
    client_type isa DataType || throw(DeclarativeError("Declarative client type '$resolved_name' is not a concrete type."))
    client_type <: AbstractChatClient || throw(DeclarativeError("Declarative client type '$resolved_name' is not a chat client."))
    return client_type
end

function _coerce_declarative_value(value, T)
    value === nothing && return nothing
    T === Any && return _materialize_declarative(value)

    if T isa Union
        for subtype in Base.uniontypes(T)
            subtype === Nothing && continue
            try
                return _coerce_declarative_value(value, subtype)
            catch
            end
        end
        return value
    end

    if value isa T
        return value
    elseif T == String
        return value isa AbstractString ? String(value) : string(value)
    elseif T <: Integer
        return Int(value)
    elseif T <: AbstractFloat
        return Float64(value)
    elseif T == Bool
        return Bool(value)
    elseif T <: AbstractDict
        materialized = _require_declarative_dict(value, "Declarative mapping")
        key_t = keytype(T)
        value_t = valtype(T)
        return Dict{key_t, value_t}(
            _coerce_declarative_value(k, key_t) => _coerce_declarative_value(v, value_t)
            for (k, v) in pairs(materialized)
        )
    elseif T <: AbstractVector
        items = _ensure_declarative_vector(value)
        element_t = eltype(T)
        return [_coerce_declarative_value(item, element_t) for item in items]
    end

    return value
end

function _construct_kw_instance(T::DataType, config::Dict{String, Any})
    fields = fieldnames(T)
    kwargs = Pair{Symbol, Any}[]

    for field in fields
        key = string(field)
        haskey(config, key) || continue
        index = findfirst(==(field), fields)
        field_t = index === nothing ? Any : fieldtype(T, index)
        push!(kwargs, field => _coerce_declarative_value(config[key], field_t))
    end

    try
        return T(; kwargs...)
    catch err
        throw(DeclarativeError("Failed to construct $(nameof(T)) from declarative config.", err))
    end
end

function _construct_chat_client(definition::AbstractDict)::AbstractChatClient
    materialized = _require_declarative_dict(definition, "Client definition")
    normalized = _apply_aliases(
        materialized,
        Dict(
            "apiKey" => "api_key",
            "apiVersion" => "api_version",
            "baseUrl" => "base_url",
            "defaultHeaders" => "default_headers",
            "projectEndpoint" => "project_endpoint",
            "readTimeout" => "read_timeout",
            "tokenScope" => "token_scope",
        ),
    )

    if haskey(normalized, "parameters") && !haskey(normalized, "options")
        normalized["options"] = _require_declarative_dict(normalized["parameters"], "Client parameters")
    elseif haskey(normalized, "parameters") && haskey(normalized, "options")
        normalized["options"] = merge(
            _require_declarative_dict(normalized["options"], "Client options"),
            _require_declarative_dict(normalized["parameters"], "Client parameters"),
        )
    end

    if haskey(normalized, "id") && !haskey(normalized, "model")
        normalized["model"] = normalized["id"]
    end

    type_name = if haskey(normalized, "type")
        string(normalized["type"])
    elseif haskey(normalized, "provider")
        string(normalized["provider"])
    elseif haskey(normalized, "kind")
        string(normalized["kind"])
    else
        throw(DeclarativeError("Client definition must include 'type' or 'provider'."))
    end

    resolved_type = _resolve_client_type_name(type_name)
    if resolved_type in ("OpenAIChatClient", "OllamaChatClient") &&
       haskey(normalized, "endpoint") &&
       !haskey(normalized, "base_url")
        normalized["base_url"] = normalized["endpoint"]
    elseif resolved_type == "FoundryChatClient" &&
           haskey(normalized, "endpoint") &&
           !haskey(normalized, "project_endpoint")
        normalized["project_endpoint"] = normalized["endpoint"]
    end

    return _construct_kw_instance(_resolve_client_type(resolved_type), normalized)
end

function _resolve_tool_spec(spec, tools::AbstractDict)::FunctionTool
    if spec isa FunctionTool
        return spec
    elseif spec isa AbstractString
        value = _resolve_named_component(string(spec), tools, _TOOL_REGISTRY, "Tool")
        value isa FunctionTool || throw(DeclarativeError("Tool '$(spec)' must resolve to a FunctionTool."))
        return value
    elseif spec isa AbstractDict
        definition = _require_declarative_dict(spec, "Tool definition")
        if haskey(definition, "ref")
            return _resolve_tool_spec(string(definition["ref"]), tools)
        elseif haskey(definition, "name") && all(k in ("name", "kind") for k in keys(definition))
            return _resolve_tool_spec(string(definition["name"]), tools)
        elseif haskey(definition, "kind")
            throw(
                DeclarativeError(
                    "Inline declarative tool kind '$(definition["kind"])' is not supported in core. Register the tool and reference it by name instead.",
                ),
            )
        end
    end

    throw(
        DeclarativeError(
            "Tool definitions must be FunctionTool instances, string refs, or mappings with 'ref'.",
        ),
    )
end

function _resolve_context_provider_spec(spec, context_providers::AbstractDict)
    if spec isa AbstractString
        return _resolve_named_component(
            string(spec),
            context_providers,
            _CONTEXT_PROVIDER_REGISTRY,
            "Context provider",
        )
    elseif spec isa AbstractDict
        definition = _require_declarative_dict(spec, "Context provider definition")
        if haskey(definition, "ref")
            return _resolve_context_provider_spec(string(definition["ref"]), context_providers)
        elseif haskey(definition, "name") && length(definition) == 1
            return _resolve_context_provider_spec(string(definition["name"]), context_providers)
        end
    elseif spec !== nothing
        return spec
    end

    throw(
        DeclarativeError(
            "Context provider definitions must be string refs, mappings with 'ref', or concrete provider instances.",
        ),
    )
end

function _resolve_client_spec(spec, clients::AbstractDict, default_client::Union{Nothing, AbstractChatClient})::AbstractChatClient
    if spec === nothing
        default_client !== nothing && return default_client
        throw(DeclarativeError("Declarative agent definition requires a client or model section."))
    elseif spec isa AbstractChatClient
        return spec
    elseif spec isa AbstractString
        value = _resolve_named_component(string(spec), clients, _CLIENT_REGISTRY, "Client")
        value isa AbstractChatClient || throw(DeclarativeError("Client '$spec' must resolve to an AbstractChatClient."))
        return value
    elseif spec isa AbstractDict
        definition = _require_declarative_dict(spec, "Client definition")
        if haskey(definition, "ref")
            return _resolve_client_spec(string(definition["ref"]), clients, default_client)
        end
        return _construct_chat_client(definition)
    end

    throw(
        DeclarativeError(
            "Client definitions must be chat clients, string refs, or mappings with 'ref'/'type'.",
        ),
    )
end

function _chat_options_from_dict(definition::AbstractDict)::ChatOptions
    materialized = _require_declarative_dict(definition, "Agent options")
    normalized = _apply_aliases(
        materialized,
        Dict(
            "maxTokens" => "max_tokens",
            "responseFormat" => "response_format",
            "toolChoice" => "tool_choice",
            "topP" => "top_p",
        ),
    )

    additional = Dict{String, Any}()
    if haskey(normalized, "additional")
        merge!(additional, _require_declarative_dict(normalized["additional"], "Agent options.additional"))
    end

    known_keys = Set(["model", "temperature", "top_p", "max_tokens", "stop", "tool_choice", "response_format", "additional"])
    for (key, value) in normalized
        key in known_keys && continue
        additional[key] = _materialize_declarative(value)
    end

    response_format = if haskey(normalized, "response_format") && normalized["response_format"] !== nothing
        _require_declarative_dict(normalized["response_format"], "Agent response_format")
    else
        nothing
    end

    return ChatOptions(
        model = haskey(normalized, "model") ? string(normalized["model"]) : nothing,
        temperature = haskey(normalized, "temperature") ? Float64(normalized["temperature"]) : nothing,
        top_p = haskey(normalized, "top_p") ? Float64(normalized["top_p"]) : nothing,
        max_tokens = haskey(normalized, "max_tokens") ? Int(normalized["max_tokens"]) : nothing,
        stop = haskey(normalized, "stop") ? _string_vector(normalized["stop"]) : nothing,
        tool_choice = haskey(normalized, "tool_choice") ? string(normalized["tool_choice"]) : nothing,
        response_format = response_format,
        additional = additional,
    )
end

function _chat_options_to_dict(options::ChatOptions)::Dict{String, Any}
    definition = Dict{String, Any}()
    _maybe_put!(definition, "model", options.model)
    _maybe_put!(definition, "temperature", options.temperature)
    _maybe_put!(definition, "top_p", options.top_p)
    _maybe_put!(definition, "max_tokens", options.max_tokens)
    _maybe_put!(definition, "stop", options.stop)
    _maybe_put!(definition, "tool_choice", options.tool_choice)
    _maybe_put!(definition, "response_format", options.response_format)

    for (key, value) in sort(collect(options.additional), by = first)
        definition[string(key)] = _materialize_declarative(value)
    end
    return definition
end

function _serialize_client_inline(client::AbstractChatClient)
    try
        _resolve_client_type(string(nameof(typeof(client))))
    catch err
        throw(
            DeclarativeError(
                "Client '$(typeof(client))' cannot be serialized inline. Register it with register_client! and reference it by name instead.",
                err,
            ),
        )
    end

    definition = Dict{String, Any}("type" => string(nameof(typeof(client))))
    for field in fieldnames(typeof(client))
        value = getfield(client, field)
        if value === nothing ||
           (value isa AbstractString && isempty(value)) ||
           (value isa AbstractVector && isempty(value)) ||
           (value isa AbstractDict && isempty(value))
            continue
        elseif value isa Function
            throw(
                DeclarativeError(
                    "Client field '$(field)' is a function and cannot be serialized inline. Register the client and reference it by name instead.",
                ),
            )
        elseif value isa AbstractDict || value isa AbstractVector || value isa Number || value isa Bool || value isa AbstractString
            definition[string(field)] = _materialize_declarative(value)
        else
            throw(
                DeclarativeError(
                    "Client field '$(field)' of type $(typeof(value)) cannot be serialized inline. Register the client and reference it by name instead.",
                ),
            )
        end
    end
    return definition
end

function _serialize_registered_tool(tool::FunctionTool)::String
    name = _registry_name(_TOOL_REGISTRY, tool)
    name !== nothing && return name
    throw(
        DeclarativeError(
            "Tool '$(tool.name)' cannot be serialized declaratively because it is not registered. Register it with register_tool! first.",
        ),
    )
end

function _serialize_registered_context_provider(provider)::String
    name = _registry_name(_CONTEXT_PROVIDER_REGISTRY, provider)
    name !== nothing && return name
    throw(
        DeclarativeError(
            "Context provider '$(typeof(provider))' cannot be serialized declaratively because it is not registered. Register it with register_context_provider! first.",
        ),
    )
end

"""
    agent_from_dict(definition; clients=Dict(), tools=Dict(), context_providers=Dict(), default_client=nothing) -> Agent

Create an Agent from a declarative dictionary definition.
"""
function agent_from_dict(
    definition::AbstractDict;
    clients::AbstractDict = Dict{String, Any}(),
    tools::AbstractDict = Dict{String, Any}(),
    context_providers::AbstractDict = Dict{String, Any}(),
    default_client::Union{Nothing, AbstractChatClient} = nothing,
)::Agent
    materialized = _materialize_declarative(definition)
    kind = lowercase(string(get(materialized, "kind", "prompt")))
    kind in ("prompt", "promptagent", "agent") ||
        throw(DeclarativeError("Unsupported declarative agent kind '$kind'."))

    client_spec = haskey(materialized, "client") ? materialized["client"] : get(materialized, "model", nothing)
    client = _resolve_client_spec(client_spec, clients, default_client)

    resolved_tools = FunctionTool[]
    for spec in _ensure_declarative_vector(get(materialized, "tools", Any[]))
        push!(resolved_tools, _resolve_tool_spec(spec, tools))
    end

    resolved_context_providers = Any[]
    context_specs = haskey(materialized, "context_providers") ?
        materialized["context_providers"] :
        get(materialized, "contextProviders", Any[])
    for spec in _ensure_declarative_vector(context_specs)
        push!(resolved_context_providers, _resolve_context_provider_spec(spec, context_providers))
    end

    options = haskey(materialized, "options") ? _chat_options_from_dict(materialized["options"]) : ChatOptions()
    max_tool_iterations = Int(
        get(
            materialized,
            "max_tool_iterations",
            get(materialized, "maxToolIterations", DEFAULT_MAX_TOOL_ITERATIONS),
        ),
    )

    return Agent(
        name = string(get(materialized, "name", "Agent")),
        description = string(get(materialized, "description", "")),
        instructions = string(get(materialized, "instructions", get(materialized, "system", ""))),
        client = client,
        tools = resolved_tools,
        context_providers = resolved_context_providers,
        options = options,
        max_tool_iterations = max_tool_iterations,
    )
end

"""
    agent_to_dict(agent::Agent) -> Dict{String, Any}

Serialize an Agent to a declarative dictionary.
"""
function agent_to_dict(agent::Agent)::Dict{String, Any}
    definition = Dict{String, Any}(
        "kind" => "Prompt",
        "name" => agent.name,
    )
    _maybe_put!(definition, "description", agent.description)
    _maybe_put!(definition, "instructions", agent.instructions)

    client_name = _registry_name(_CLIENT_REGISTRY, agent.client)
    definition["client"] = client_name === nothing ? _serialize_client_inline(agent.client) : client_name

    if !isempty(agent.tools)
        definition["tools"] = [_serialize_registered_tool(tool) for tool in agent.tools]
    end

    if !isempty(agent.context_providers)
        definition["context_providers"] = [
            _serialize_registered_context_provider(provider) for provider in agent.context_providers
        ]
    end

    options = _chat_options_to_dict(agent.options)
    !isempty(options) && (definition["options"] = options)
    agent.max_tool_iterations != DEFAULT_MAX_TOOL_ITERATIONS &&
        (definition["max_tool_iterations"] = agent.max_tool_iterations)

    return definition
end

"""
    agent_from_json(json; kwargs...) -> Agent

Create an Agent from a declarative JSON string.
"""
function agent_from_json(json::AbstractString; kwargs...)::Agent
    agent_from_dict(_load_json_definition(json); kwargs...)
end

"""
    agent_to_json(agent::Agent) -> String

Serialize an Agent to a declarative JSON string.
"""
agent_to_json(agent::Agent)::String = JSON3.write(agent_to_dict(agent))

"""
    agent_from_yaml(yaml; kwargs...) -> Agent

Create an Agent from a declarative YAML string.
"""
function agent_from_yaml(yaml::AbstractString; kwargs...)::Agent
    agent_from_dict(_load_yaml_definition(yaml); kwargs...)
end

"""
    agent_to_yaml(agent::Agent) -> String

Serialize an Agent to a declarative YAML string.
"""
agent_to_yaml(agent::Agent)::String = YAML.write(agent_to_dict(agent))

"""
    agent_from_file(path; kwargs...) -> Agent

Create an Agent from a declarative JSON or YAML file.
"""
function agent_from_file(path::String; kwargs...)::Agent
    definition, _ = _read_declarative_definition(path)
    return agent_from_dict(definition; kwargs...)
end

"""
    agent_to_file(agent::Agent, path::String)

Serialize an Agent to a declarative JSON or YAML file based on its extension.
"""
function agent_to_file(agent::Agent, path::String)
    _write_declarative_file(agent_to_dict(agent), path)
end
