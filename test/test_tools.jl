using AgentFramework
using Test

@testset "Tools" begin
    @testset "FunctionTool construction" begin
        t = FunctionTool(
            name = "greet",
            description = "Say hello",
            func = (name) -> "Hello, $name!",
            parameters = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "name" => Dict{String, Any}("type" => "string"),
                ),
                "required" => ["name"],
            ),
        )
        @test t.name == "greet"
        @test t.description == "Say hello"
    end

    @testset "invoke_tool with Dict args" begin
        add_func(a, b) = a + b
        t = FunctionTool(
            name = "add",
            description = "Add two numbers",
            func = add_func,
            parameters = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "a" => Dict{String, Any}("type" => "number"),
                    "b" => Dict{String, Any}("type" => "number"),
                ),
                "required" => ["a", "b"],
            ),
        )
        result = invoke_tool(t, Dict{String, Any}("a" => 3, "b" => 4))
        @test result == 7
    end

    @testset "invoke_tool with JSON string" begin
        greet_fn(name) = string("Hello, ", name, "!")
        t = FunctionTool(
            name = "greet",
            description = "Greet someone",
            func = greet_fn,
            parameters = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "name" => Dict{String, Any}("type" => "string"),
                ),
                "required" => ["name"],
            ),
        )
        result = invoke_tool(t, """{"name": "Alice"}""")
        @test result == "Hello, Alice!"
    end

    @testset "tool_to_schema" begin
        t = FunctionTool(
            name = "search",
            description = "Search the web",
            func = identity,
            parameters = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "query" => Dict{String, Any}("type" => "string"),
                ),
                "required" => ["query"],
            ),
        )
        schema = tool_to_schema(t)
        @test schema["type"] == "function"
        @test schema["function"]["name"] == "search"
        @test schema["function"]["description"] == "Search the web"
        @test haskey(schema["function"], "parameters")
    end

    @testset "normalize_tools" begin
        t1 = FunctionTool(name="a", description="A", func=identity)
        t2 = FunctionTool(name="b", description="B", func=identity)
        tools = normalize_tools([t1, t2])
        @test length(tools) == 2

        # Duplicate names
        t3 = FunctionTool(name="a", description="A2", func=identity)
        @test_throws ToolError normalize_tools([t1, t3])

        # Nothing
        @test isempty(normalize_tools(nothing))
    end

    @testset "find_tool" begin
        t1 = FunctionTool(name="alpha", description="A", func=identity)
        t2 = FunctionTool(name="beta", description="B", func=identity)
        @test find_tool([t1, t2], "alpha") === t1
        @test find_tool([t1, t2], "gamma") === nothing
    end

    @testset "Julia type to JSON schema" begin
        @test AgentFramework._julia_type_to_json_schema(String)["type"] == "string"
        @test AgentFramework._julia_type_to_json_schema(Int)["type"] == "integer"
        @test AgentFramework._julia_type_to_json_schema(Float64)["type"] == "number"
        @test AgentFramework._julia_type_to_json_schema(Bool)["type"] == "boolean"
        @test AgentFramework._julia_type_to_json_schema(Nothing)["type"] == "null"
        @test AgentFramework._julia_type_to_json_schema(Vector{String})["type"] == "array"
    end
end
