# Tools

Tools let agents interact with the world by calling Julia functions. When an LLM decides it needs external information or wants to perform an action, it generates a tool call request. AgentFramework.jl intercepts this, invokes the corresponding Julia function, and feeds the result back to the LLM.

## Overview

The tool system has three layers:

1. **[`@tool`](@ref) macro** — Define tools declaratively from function definitions
2. **[`FunctionTool`](@ref)** — The runtime representation of a tool with JSON Schema metadata
3. **Tool execution** — Automatic argument parsing, invocation, and result formatting

## The @tool Macro

The [`@tool`](@ref) macro is the recommended way to create tools. It inspects the function signature to generate JSON Schema metadata automatically:

```julia
using AgentFramework

@tool function get_weather(location::String, unit::String="celsius")
    "Get the current weather for a location."
    return "Sunny, 22°C in $(location)"
end
```

After this definition, `get_weather` is a [`FunctionTool`](@ref) (not a regular function). Here's what the macro does:

1. **Defines the function** — The underlying Julia function is created as normal
2. **Extracts the description** — The first string literal in the body becomes the tool description
3. **Builds parameter schema** — Parameter names and types are converted to JSON Schema
4. **Creates the FunctionTool** — Assigns it to the function name

### Type Mappings

The macro maps Julia types to JSON Schema types:

| Julia Type | JSON Schema Type |
|:-----------|:-----------------|
| `String` | `"string"` |
| `Int`, `Int64`, `Int32` | `"integer"` |
| `Float64`, `Float32` | `"number"` |
| `Bool` | `"boolean"` |
| `Vector{T}` | `"array"` with typed items |
| `Dict` | `"object"` |
| `Any` (or untyped) | `"string"` |

### Required vs Optional Parameters

Parameters with default values are optional in the schema; parameters without defaults are required:

```julia
@tool function search(query::String, max_results::Int=10, language::String="en")
    "Search for documents."
    # query is required; max_results and language are optional
    return "Results for: $(query)"
end
```

Generated schema:
```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "max_results": {"type": "integer"},
    "language": {"type": "string"}
  },
  "required": ["query"]
}
```

### Keyword Arguments

The `@tool` macro also handles keyword arguments (after a semicolon):

```julia
@tool function create_file(path::String; content::String="", overwrite::Bool=false)
    "Create a file at the given path."
    # ...
end
```

## Manual FunctionTool Creation

For full control over the schema, create a [`FunctionTool`](@ref) directly:

```julia
my_tool = FunctionTool(
    name = "lookup_user",
    description = "Look up a user by their ID or email address.",
    func = (id) -> "User: Alice (id=$(id))",
    parameters = Dict{String, Any}(
        "type" => "object",
        "properties" => Dict{String, Any}(
            "id" => Dict{String, Any}(
                "type" => "string",
                "description" => "User ID or email address",
            ),
        ),
        "required" => ["id"],
    ),
)
```

This is useful when:
- You need custom property descriptions
- The schema requires enums, patterns, or nested objects
- You're wrapping an external API with a specific schema

## Tool Schemas

Every tool has a JSON Schema that describes its parameters. You can inspect it:

```julia
@tool function greet(name::String)
    "Greet someone by name."
    return "Hello, $(name)!"
end

# View the tool's schema
println(greet.name)          # "greet"
println(greet.description)   # "Greet someone by name."
println(greet.parameters)    # Dict with JSON Schema
```

Convert to the OpenAI function-calling format with [`tool_to_schema`](@ref):

```julia
schema = tool_to_schema(greet)
# Dict("type" => "function", "function" => Dict("name" => "greet", ...))
```

## Tool Invocation Lifecycle

When the LLM generates a tool call, the framework handles the full lifecycle:

```
LLM Response (with tool call)
    │
    ├─ 1. Parse tool call: extract name, arguments, call_id
    ├─ 2. Find tool: match name against agent's tool list
    ├─ 3. Parse arguments: JSON string → Dict{String, Any}
    ├─ 4. Function middleware: run through function middleware pipeline
    ├─ 5. Invoke function: call tool.func with matched arguments
    ├─ 6. Parse result: convert return value to string
    └─ 7. Send back: add function_result_content to messages, loop to LLM
```

You can intercept step 4 with function middleware (see [Middleware](@ref)).

### Direct Invocation

You can invoke tools programmatically:

```julia
# With a Dict
result = invoke_tool(greet, Dict{String, Any}("name" => "Alice"))

# With a JSON string
result = invoke_tool(greet, """{"name": "Alice"}""")
```

## Declaration-Only Tools

A tool without an implementation (no `func`) is declaration-only. It tells the LLM about a capability without the framework handling execution. This is useful for service-backed tools where the provider handles execution:

```julia
declaration = FunctionTool(
    name = "code_interpreter",
    description = "Execute Python code in a sandboxed environment.",
    parameters = Dict{String, Any}(
        "type" => "object",
        "properties" => Dict{String, Any}(
            "code" => Dict{String, Any}("type" => "string"),
        ),
        "required" => ["code"],
    ),
    # No func — declaration only
)

is_declaration_only(declaration)  # true
```

## Tool Options

[`FunctionTool`](@ref) supports several advanced options:

### Invocation Limits

Control how many times a tool can be called:

```julia
limited_tool = FunctionTool(
    name = "expensive_api",
    description = "Call an expensive external API.",
    func = call_api,
    parameters = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}()),
    max_invocations = 3,              # Max 3 calls total
    max_invocation_exceptions = 2,    # Stop after 2 errors
)
```

Reset counters with [`reset_invocation_count!`](@ref):

```julia
reset_invocation_count!(limited_tool)
```

### Strict Mode

Enable strict parameter validation:

```julia
strict_tool = FunctionTool(
    name = "precise_tool",
    description = "A tool with strict validation.",
    func = my_func,
    parameters = params,
    strict = true,  # Adds "strict": true to the schema
)
```

### Approval Mode

Require human approval before tool execution:

```julia
dangerous_tool = FunctionTool(
    name = "delete_file",
    description = "Delete a file from the filesystem.",
    func = rm,
    parameters = params,
    approval_mode = :always_require,  # Requires approval before execution
)
```

### Custom Result Parsing

Override how return values are converted to strings:

```julia
@tool function get_data(query::String)
    "Fetch structured data."
    return Dict("count" => 42, "items" => ["a", "b", "c"])
end

# Default: result is JSON-serialized
# Custom: extract just what matters
get_data.result_parser = result -> "Found $(result["count"]) items"
```

## Tool Collection Utilities

### Finding Tools

```julia
@tool function calculate(expression::String)
    "Evaluate a numeric expression."
    return "42"
end

@tool function current_time()
    "Get the current time."
    return "12:00"
end

tools = [get_weather, calculate, current_time]
tool = find_tool(tools, "calculate")  # Returns the FunctionTool or nothing
```

### Normalizing Tools

[`normalize_tools`](@ref) deduplicates and validates a tool collection:

```julia
tools = normalize_tools([get_weather, calculate])
# Throws ToolError if duplicate names are found
```

## Next Steps

- [Middleware](@ref) — Intercept tool invocations with function middleware
- [Agents](@ref) — Learn how agents use tools in the execution loop
- [Advanced Topics](@ref) — MCP integration for connecting external tool servers
