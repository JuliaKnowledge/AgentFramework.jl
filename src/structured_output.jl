# Structured output support for AgentFramework.jl
# Enables type-safe parsing of LLM responses into Julia structs.

"""
    StructuredOutput{T}

Wrapper holding a parsed Julia struct from an LLM response, along with the raw text.

# Fields
- `value::T`: The parsed struct.
- `raw_text::String`: The raw JSON text from the LLM.
"""
struct StructuredOutput{T}
    value::T
    raw_text::String
end

Base.show(io::IO, s::StructuredOutput{T}) where T = print(io, "StructuredOutput{", T, "}(", s.value, ")")

"""
    schema_from_type(::Type{T}) -> Dict{String, Any}

Generate an OpenAI-compatible JSON Schema from a Julia struct type.
Supports `@kwdef` structs with `String`, `Int`, `Float64`, `Bool`, `Vector`, 
`Dict`, `Union{Nothing, T}`, and nested struct types.

# Example
```julia
@kwdef struct MovieReview
    title::String
    rating::Int
    summary::String
end

schema = schema_from_type(MovieReview)
# Dict("type"=>"object", "properties"=>Dict("title"=>..., "rating"=>..., "summary"=>...), "required"=>[...])
```
"""
function schema_from_type(::Type{T})::Dict{String, Any} where T
    props = Dict{String, Any}()
    required = String[]

    for (fname, ftype) in zip(fieldnames(T), fieldtypes(T))
        name = String(fname)
        is_optional, inner_type = _unwrap_optional(ftype)
        props[name] = _type_to_json_schema(inner_type)
        if !is_optional
            push!(required, name)
        end
    end

    schema = Dict{String, Any}(
        "type" => "object",
        "properties" => props,
    )
    if !isempty(required)
        schema["required"] = required
    end
    return schema
end

"""
    response_format_for(::Type{T}) -> Dict{String, Any}

Build the `response_format` dict for ChatOptions that requests JSON output
conforming to the schema of type `T`.

# Example
```julia
options = ChatOptions(response_format=response_format_for(MovieReview))
```
"""
function response_format_for(::Type{T})::Dict{String, Any} where T
    Dict{String, Any}(
        "type" => "json_schema",
        "json_schema" => Dict{String, Any}(
            "name" => String(nameof(T)),
            "strict" => true,
            "schema" => schema_from_type(T),
        ),
    )
end

"""
    parse_structured(::Type{T}, response::ChatResponse) -> StructuredOutput{T}
    parse_structured(::Type{T}, response::AgentResponse) -> StructuredOutput{T}
    parse_structured(::Type{T}, text::AbstractString) -> StructuredOutput{T}

Parse a chat/agent response into a typed Julia struct. Extracts JSON from
the response text (handling markdown code fences if present) and deserializes
into type `T`.

# Example
```julia
response = run_agent(agent, "Review The Matrix"; options=ChatOptions(response_format=response_format_for(MovieReview)))
result = parse_structured(MovieReview, response)
println(result.value.title)   # "The Matrix"
println(result.value.rating)  # 9
```
"""
function parse_structured(::Type{T}, response::ChatResponse)::StructuredOutput{T} where T
    parse_structured(T, get_text(response))
end

function parse_structured(::Type{T}, response::AgentResponse)::StructuredOutput{T} where T
    parse_structured(T, get_text(response))
end

function parse_structured(::Type{T}, text::AbstractString)::StructuredOutput{T} where T
    json_text = _extract_json(text)
    try
        data = JSON3.read(json_text, Dict{String, Any})
        value = _dict_to_struct(T, data)
        return StructuredOutput{T}(value, json_text)
    catch e
        throw(ContentError("Failed to parse structured output as $(nameof(T)): $(sprint(showerror, e))\nRaw text: $(text[1:min(500, length(text))])"))
    end
end

# ── JSON Extraction ──────────────────────────────────────────────────────────

"""Extract JSON from text that may be wrapped in markdown code fences."""
function _extract_json(text::AbstractString)::String
    stripped = strip(text)

    # Try to find JSON in markdown code fences: ```json ... ``` or ``` ... ```
    m = match(r"```(?:json)?\s*\n?(.*?)\n?\s*```"s, stripped)
    if m !== nothing
        return strip(m.captures[1])
    end

    # Already raw JSON (starts with { or [)
    if startswith(stripped, '{') || startswith(stripped, '[')
        return stripped
    end

    # Try to find first { ... } or [ ... ] block
    first_brace = findfirst('{', stripped)
    last_brace = findlast('}', stripped)
    if first_brace !== nothing && last_brace !== nothing
        return stripped[first_brace:last_brace]
    end

    return stripped
end

# ── Type Conversion ──────────────────────────────────────────────────────────

const _BASIC_JSON_TYPES = Dict{Type, String}(
    String => "string",
    Int => "integer",
    Int64 => "integer",
    Int32 => "integer",
    Float64 => "number",
    Float32 => "number",
    Bool => "boolean",
)

function _unwrap_optional(T::Type)
    if T isa Union
        types = Base.uniontypes(T)
        non_nothing = filter(t -> t !== Nothing, types)
        if length(non_nothing) == 1
            return (true, non_nothing[1])
        end
    end
    return (false, T)
end

function _type_to_json_schema(T::Type)::Dict{String, Any}
    if T === Nothing
        return Dict{String, Any}("type" => "null")
    end

    if T === Any || T === String
        return Dict{String, Any}("type" => "string")
    end

    if haskey(_BASIC_JSON_TYPES, T)
        return Dict{String, Any}("type" => _BASIC_JSON_TYPES[T])
    end

    # Vector types
    if T <: AbstractVector
        items_schema = _type_to_json_schema(eltype(T))
        return Dict{String, Any}("type" => "array", "items" => items_schema)
    end

    # Dict types
    if T <: AbstractDict
        return Dict{String, Any}("type" => "object")
    end

    # Nested struct types — recurse
    if isstructtype(T) && !(T <: Number) && !(T <: AbstractString)
        return schema_from_type(T)
    end

    # Fallback
    return Dict{String, Any}("type" => "string")
end

"""Convert a Dict to a Julia struct, handling nested types and type coercion."""
function _dict_to_struct(::Type{T}, data::Dict{String, Any})::T where T
    kwargs = Dict{Symbol, Any}()

    for (fname, ftype) in zip(fieldnames(T), fieldtypes(T))
        key = String(fname)
        if haskey(data, key)
            val = data[key]
            is_optional, inner_type = _unwrap_optional(ftype)
            if val === nothing
                kwargs[fname] = nothing
            else
                kwargs[fname] = _coerce_value(inner_type, val)
            end
        end
    end

    return T(; kwargs...)
end

function _coerce_value(::Type{T}, val) where T
    # Already the right type
    val isa T && return val

    # Number coercion
    if T <: Integer && val isa Number
        return T(round(Integer, val))
    end
    if T <: AbstractFloat && val isa Number
        return T(val)
    end

    # String coercion
    if T === String && !(val isa String)
        return string(val)
    end

    # Bool
    if T === Bool
        return Bool(val)
    end

    # Vector
    if T <: AbstractVector
        ET = eltype(T)
        return ET[_coerce_value(ET, v) for v in val]
    end

    # Dict passthrough
    if T <: AbstractDict
        return val
    end

    # Nested struct
    if isstructtype(T) && val isa Dict
        return _dict_to_struct(T, Dict{String, Any}(string(k) => v for (k, v) in pairs(val)))
    end

    # Fallback — try direct conversion
    return convert(T, val)
end
