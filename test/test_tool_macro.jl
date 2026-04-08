# @tool macro tests for AgentFramework.jl

using AgentFramework
using Test

@testset "Tool Macro" begin

    @testset "Basic @tool function" begin
        @tool function greet(name::String)
            "Greet someone by name."
            return "Hello, " * name * "!"
        end

        @test greet isa FunctionTool
        @test greet.name == "greet"
        @test greet.description == "Greet someone by name."
        @test haskey(greet.parameters, "properties")
        @test greet.parameters["properties"]["name"]["type"] == "string"
        @test "name" in greet.parameters["required"]

        # Invoke
        result = invoke_tool(greet, Dict{String, Any}("name" => "Alice"))
        @test result == "Hello, Alice!"
    end

    @testset "@tool with multiple typed params" begin
        @tool function add_numbers(a::Int, b::Int)
            "Add two integers."
            return a + b
        end

        @test add_numbers isa FunctionTool
        @test add_numbers.parameters["properties"]["a"]["type"] == "integer"
        @test add_numbers.parameters["properties"]["b"]["type"] == "integer"
        @test Set(add_numbers.parameters["required"]) == Set(["a", "b"])

        result = invoke_tool(add_numbers, Dict{String, Any}("a" => 3, "b" => 7))
        @test result == 10
    end

    @testset "@tool with default parameter" begin
        @tool function search(query::String, max_results::Int=5)
            "Search with optional limit."
            return "Searching '$query' (max $max_results)"
        end

        @test search isa FunctionTool
        @test "query" in search.parameters["required"]
        # max_results has a default so should NOT be required
        @test !("max_results" in get(search.parameters, "required", String[]))
    end

    @testset "@tool with Float64 param" begin
        @tool function set_temp(temperature::Float64)
            "Set temperature."
            return temperature
        end

        @test set_temp isa FunctionTool
        @test set_temp.parameters["properties"]["temperature"]["type"] == "number"
    end

    @testset "@tool with Bool param" begin
        @tool function toggle(enabled::Bool)
            "Toggle a setting."
            return !enabled
        end

        @test toggle isa FunctionTool
        @test toggle.parameters["properties"]["enabled"]["type"] == "boolean"
    end

    @testset "@tool works in local scope" begin
        scoped_tool = let
            @tool function local_greet(name::String)
                "Greet from a local scope."
                return "Hello, " * name
            end
            local_greet
        end

        @test scoped_tool isa FunctionTool
        @test scoped_tool.name == "local_greet"
        @test invoke_tool(scoped_tool, Dict{String, Any}("name" => "Dana")) == "Hello, Dana"
    end

    @testset "@tool works at module scope" begin
        mod = Module(:ToolMacroModuleScope)
        Core.eval(mod, :(using AgentFramework))
        Core.eval(mod, quote
            @tool function module_greet(name::String)
                "Greet from module scope."
                return "Hello, " * name
            end
        end)

        module_tool = Core.eval(mod, :module_greet)
        @test module_tool isa FunctionTool
        @test module_tool.name == "module_greet"
        @test invoke_tool(module_tool, Dict{String, Any}("name" => "Mia")) == "Hello, Mia"
    end

    @testset "@tool schema format" begin
        @tool function my_tool(x::String)
            "A tool."
            return x
        end

        schema = tool_to_schema(my_tool)
        @test schema["type"] == "function"
        @test schema["function"]["name"] == "my_tool"
        @test schema["function"]["description"] == "A tool."
        @test schema["function"]["parameters"]["type"] == "object"
    end

    @testset "@tool with no type annotation" begin
        @tool function untyped(value)
            "Handle any value."
            return string(value)
        end

        @test untyped isa FunctionTool
        # Untyped params default to "string"
        @test untyped.parameters["properties"]["value"]["type"] == "string"
    end

    @testset "invoke_tool with JSON string" begin
        @tool function concat(a::String, b::String)
            "Concatenate strings."
            return a * b
        end

        result = invoke_tool(concat, """{"a": "hello", "b": " world"}""")
        @test result == "hello world"
    end

    @testset "find_tool" begin
        @tool function tool_a(x::String)
            "Tool A."
            return x
        end

        @tool function tool_b(x::Int)
            "Tool B."
            return x * 2
        end

        tools = [tool_a, tool_b]
        @test find_tool(tools, "tool_a") === tool_a
        @test find_tool(tools, "tool_b") === tool_b
        @test find_tool(tools, "nonexistent") === nothing
    end

    @testset "normalize_tools" begin
        t1 = FunctionTool(name="a", description="A", func=identity)
        t2 = FunctionTool(name="b", description="B", func=identity)

        normalized = normalize_tools([t1, t2])
        @test length(normalized) == 2

        # Duplicate detection
        @test_throws ToolError normalize_tools([t1, FunctionTool(name="a", description="dup", func=identity)])
    end
end
