# Tool abstractions for AgentFramework.jl
# Mirrors the Python FunctionTool and @tool decorator.

"""
    FunctionTool <: AbstractTool

Wraps a Julia function as an AI-invocable tool with JSON Schema metadata.

# Fields
- `name::String`: Tool name (used in LLM function calling).
- `description::String`: Human-readable description for the LLM.
- `func::Function`: The underlying Julia function.
- `parameters::Dict{String, Any}`: JSON Schema for the function parameters.
- `strict::Bool`: Whether strict parameter validation is enforced.

# Examples
```julia
# Using the @tool macro
@tool function get_weather(location::String, unit::String="celsius")
    "Get the current weather for a location."
    return "Sunny, 22°C in \$location"
end

# Manual construction
t = FunctionTool(
    name = "get_weather",
    description = "Get weather for a location",
    func = my_func,
    parameters = Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "City name"),
        ),
        "required" => ["location"],
    ),
)
```
"""
Base.@kwdef mutable struct FunctionTool <: AbstractTool
    name::String
    description::String
    func::Union{Nothing, Function} = nothing
    parameters::Dict{String, Any} = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}())
    strict::Bool = false
    approval_mode::Symbol = :never_require  # :always_require or :never_require
    max_invocations::Union{Nothing, Int} = nothing
    max_invocation_exceptions::Union{Nothing, Int} = nothing
    invocation_count::Int = 0
    invocation_exception_count::Int = 0
    result_parser::Union{Nothing, Function} = nothing  # (raw_result) -> String
    kind::Union{Nothing, String} = nothing
    additional_properties::Dict{String, Any} = Dict{String, Any}()
end

"""Whether this tool is declaration-only (no implementation to execute)."""
is_declaration_only(t::FunctionTool) = t.func === nothing

function Base.show(io::IO, t::FunctionTool)
    suffix = is_declaration_only(t) ? " [declaration]" : ""
    print(io, "FunctionTool(\"", t.name, "\"", suffix, ")")
end

"""
    invoke_tool(tool::FunctionTool, arguments::Dict{String, Any}) -> Any

Invoke a tool with parsed arguments. Arguments are matched to function parameters by name.
Tracks invocation count and enforces max_invocations / max_invocation_exceptions limits.
"""
function invoke_tool(tool::FunctionTool, arguments::Dict{String, Any})
    is_declaration_only(tool) && throw(ToolExecutionError(
        "Cannot invoke declaration-only tool '$(tool.name)': no implementation provided."))

    if tool.max_invocations !== nothing && tool.invocation_count >= tool.max_invocations
        throw(ToolExecutionError(
            "Tool '$(tool.name)' exceeded max_invocations limit ($(tool.max_invocations))."))
    end

    tool.invocation_count += 1

    raw_result = try
        m = first(methods(tool.func))
        param_names = Base.method_argnames(m)[2:end]
        args = Any[]
        for pname in param_names
            key = String(pname)
            if haskey(arguments, key)
                push!(args, arguments[key])
            end
        end
        tool.func(args...)
    catch e
        tool.invocation_exception_count += 1
        if tool.max_invocation_exceptions !== nothing &&
           tool.invocation_exception_count >= tool.max_invocation_exceptions
            throw(ToolExecutionError(
                "Tool '$(tool.name)' exceeded max_invocation_exceptions limit ($(tool.max_invocation_exceptions))."))
        end
        rethrow(e)
    end

    # Apply result_parser if provided
    if tool.result_parser !== nothing
        return tool.result_parser(raw_result)
    end
    return raw_result
end

"""
    invoke_tool(tool::FunctionTool, arguments_json::AbstractString) -> Any

Invoke a tool with JSON-encoded arguments.
"""
function invoke_tool(tool::FunctionTool, arguments_json::AbstractString)
    args = JSON3.read(arguments_json, Dict{String, Any})
    return invoke_tool(tool, args)
end

"""
    parse_result(value) -> String

Convert a raw tool return value to a string suitable for the LLM.
Handles strings, dicts, vectors, Content objects, and arbitrary types.
"""
function parse_result(value)::String
    value isa AbstractString && return String(value)
    value isa Content && return get_text(value)
    value isa AbstractDict && return JSON3.write(value)
    value isa AbstractVector && return JSON3.write(value)
    value === nothing && return ""
    return string(value)
end

"""
    reset_invocation_count!(tool::FunctionTool)

Reset invocation and exception counters to zero.
"""
function reset_invocation_count!(tool::FunctionTool)
    tool.invocation_count = 0
    tool.invocation_exception_count = 0
    return tool
end

"""
    tool_to_schema(tool::FunctionTool) -> Dict{String, Any}

Convert a FunctionTool to the OpenAI function-calling JSON schema format.
"""
function tool_to_schema(tool::FunctionTool)::Dict{String, Any}
    schema = Dict{String, Any}(
        "type" => "function",
        "function" => Dict{String, Any}(
            "name" => tool.name,
            "description" => tool.description,
            "parameters" => tool.parameters,
        ),
    )
    if tool.strict
        schema["function"]["strict"] = true
    end
    return schema
end

# ── Julia Type → JSON Schema Mapping ─────────────────────────────────────────

const _JULIA_TO_JSON_TYPE = Dict{Type, String}(
    String => "string",
    Int => "integer",
    Int64 => "integer",
    Int32 => "integer",
    Float64 => "number",
    Float32 => "number",
    Bool => "boolean",
    Any => "string",
)

function _julia_type_to_json_schema(T::Type)::Dict{String, Any}
    if T === Nothing
        return Dict{String, Any}("type" => "null")
    end
    # Handle Union{Nothing, X} as optional
    if T isa Union
        types = Base.uniontypes(T)
        non_nothing = filter(t -> t !== Nothing, types)
        if length(non_nothing) == 1
            return _julia_type_to_json_schema(non_nothing[1])
        end
    end
    if T <: AbstractVector
        eltype_schema = _julia_type_to_json_schema(eltype(T))
        return Dict{String, Any}("type" => "array", "items" => eltype_schema)
    end
    if T <: AbstractDict
        return Dict{String, Any}("type" => "object")
    end
    return Dict{String, Any}("type" => get(_JULIA_TO_JSON_TYPE, T, "string"))
end

# ── @tool Macro ──────────────────────────────────────────────────────────────

"""
    @tool function name(args...)
        "Description string"
        ...
    end

Define a `FunctionTool` from a Julia function definition. The first string literal
in the function body is used as the tool description. Parameter types are converted
to JSON Schema automatically.

# Example
```julia
@tool function search_web(query::String, max_results::Int=5)
    "Search the web and return results."
    # ... implementation
end
# Creates: search_web::FunctionTool
```
"""
macro tool(funcdef)
    funcdef.head == :function || error("@tool must be applied to a function definition")

    sig = funcdef.args[1]
    body = funcdef.args[2]

    # Extract function name
    if sig isa Expr && sig.head == :call
        fname = sig.args[1]
        raw_params = sig.args[2:end]
    else
        error("@tool: cannot parse function signature")
    end

    # Extract description from first string in body
    description = ""
    if body isa Expr && body.head == :block
        for (i, stmt) in enumerate(body.args)
            if stmt isa String
                description = stmt
                break
            elseif stmt isa Expr && stmt.head == :string
                description = string(stmt.args...)
                break
            end
        end
    end

    # Build parameter schema at macro expansion time
    param_exprs = Expr[]
    required_names = String[]

    for p in raw_params
        if p isa Expr && p.head == :(::)
            # name::Type
            pname = String(p.args[1])
            ptype = p.args[2]
            push!(param_exprs, :(props[$pname] = _julia_type_to_json_schema($ptype)))
            push!(required_names, pname)
        elseif p isa Expr && p.head == :kw
            # name::Type = default (keyword with default)
            inner = p.args[1]
            if inner isa Expr && inner.head == :(::)
                pname = String(inner.args[1])
                ptype = inner.args[2]
                push!(param_exprs, :(props[$pname] = _julia_type_to_json_schema($ptype)))
            elseif inner isa Symbol
                pname = String(inner)
                push!(param_exprs, :(props[$pname] = Dict{String, Any}("type" => "string")))
            end
            # Parameters with defaults are not required
        elseif p isa Symbol
            # name (no type annotation)
            pname = String(p)
            push!(param_exprs, :(props[$pname] = Dict{String, Any}("type" => "string")))
            push!(required_names, pname)
        elseif p isa Expr && p.head == :parameters
            # Handle keyword arguments after semicolon
            for kw in p.args
                if kw isa Expr && kw.head == :kw
                    inner = kw.args[1]
                    if inner isa Expr && inner.head == :(::)
                        pname = String(inner.args[1])
                        ptype = inner.args[2]
                        push!(param_exprs, :(props[$pname] = _julia_type_to_json_schema($ptype)))
                    elseif inner isa Symbol
                        pname = String(inner)
                        push!(param_exprs, :(props[$pname] = Dict{String, Any}("type" => "string")))
                    end
                end
            end
        end
    end

    impl_name = gensym(fname)
    impl_sig = deepcopy(sig)
    impl_sig.args[1] = impl_name
    impl_funcdef = Expr(:function, impl_sig, body)
    fname_str = String(fname)

    quote
        # Define the implementation under a private name so module-scope bindings
        # can still use the public name for the FunctionTool object.
        $(esc(impl_funcdef))

        $(esc(fname)) = let props = Dict{String, Any}()
            $(Expr(:block, param_exprs...))
            params = Dict{String, Any}(
                "type" => "object",
                "properties" => props,
            )
            req = String[$(required_names...)]
            if !isempty(req)
                params["required"] = req
            end
            FunctionTool(
                name = $fname_str,
                description = $description,
                func = $(esc(impl_name)),
                parameters = params,
            )
        end
    end
end

# ── Tool Collection Utilities ────────────────────────────────────────────────

"""
    normalize_tools(tools) -> Vector{FunctionTool}

Normalize a collection of tools to a flat vector of FunctionTool.
"""
function normalize_tools(tools::Vector)::Vector{FunctionTool}
    result = FunctionTool[]
    seen = Set{String}()
    for t in tools
        if t isa FunctionTool
            if t.name in seen
                throw(ToolError("Duplicate tool name: $(t.name)"))
            end
            push!(seen, t.name)
            push!(result, t)
        end
    end
    return result
end

normalize_tools(::Nothing) = FunctionTool[]

"""
    find_tool(tools::Vector{FunctionTool}, name::String) -> Union{Nothing, FunctionTool}

Find a tool by name in a collection.
"""
function find_tool(tools::Vector{FunctionTool}, name::String)::Union{Nothing, FunctionTool}
    idx = findfirst(t -> t.name == name, tools)
    return idx === nothing ? nothing : tools[idx]
end
