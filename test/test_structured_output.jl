using AgentFramework
using Test

@testset "Structured Output" begin

    @testset "Schema generation" begin
        # Simple struct
        @kwdef struct SimpleMovie
            title::String
            year::Int
            rating::Float64
        end

        schema = schema_from_type(SimpleMovie)
        @test schema["type"] == "object"
        @test haskey(schema, "properties")
        @test schema["properties"]["title"]["type"] == "string"
        @test schema["properties"]["year"]["type"] == "integer"
        @test schema["properties"]["rating"]["type"] == "number"
        @test Set(schema["required"]) == Set(["title", "year", "rating"])
    end

    @testset "Schema with optional fields" begin
        @kwdef struct OptionalFields
            name::String
            age::Union{Nothing, Int} = nothing
            active::Bool = true
        end

        schema = schema_from_type(OptionalFields)
        @test "name" in schema["required"]
        # Optional fields should not be in required
        @test !("age" in get(schema, "required", String[]))
    end

    @testset "Schema with nested structs" begin
        @kwdef struct Address
            street::String
            city::String
        end

        @kwdef struct Person
            name::String
            address::Address
        end

        schema = schema_from_type(Person)
        @test schema["properties"]["name"]["type"] == "string"
        addr_schema = schema["properties"]["address"]
        @test addr_schema["type"] == "object"
        @test haskey(addr_schema, "properties")
        @test addr_schema["properties"]["street"]["type"] == "string"
    end

    @testset "Schema with vectors" begin
        @kwdef struct TaggedItem
            name::String
            tags::Vector{String}
            scores::Vector{Float64}
        end

        schema = schema_from_type(TaggedItem)
        @test schema["properties"]["tags"]["type"] == "array"
        @test schema["properties"]["tags"]["items"]["type"] == "string"
        @test schema["properties"]["scores"]["type"] == "array"
        @test schema["properties"]["scores"]["items"]["type"] == "number"
    end

    @testset "response_format_for" begin
        @kwdef struct TestOutput
            answer::String
            confidence::Float64
        end

        rf = response_format_for(TestOutput)
        @test rf["type"] == "json_schema"
        @test rf["json_schema"]["name"] == "TestOutput"
        @test rf["json_schema"]["strict"] == true
        @test rf["json_schema"]["schema"]["type"] == "object"
    end

    @testset "Parse structured from JSON" begin
        @kwdef struct ParseTarget
            name::String
            count::Int
            active::Bool
        end

        json_text = """{"name": "test", "count": 42, "active": true}"""
        result = parse_structured(ParseTarget, json_text)
        @test result isa StructuredOutput{ParseTarget}
        @test result.value.name == "test"
        @test result.value.count == 42
        @test result.value.active == true
        @test result.raw_text == json_text
    end

    @testset "Parse with markdown code fence" begin
        @kwdef struct FencedTarget
            x::Int
            y::Int
        end

        text = """Here is the result:
```json
{"x": 10, "y": 20}
```
"""
        result = parse_structured(FencedTarget, text)
        @test result.value.x == 10
        @test result.value.y == 20
    end

    @testset "Parse with embedded JSON" begin
        @kwdef struct EmbeddedTarget
            value::String
        end

        text = """The answer is {"value": "hello"} as requested."""
        result = parse_structured(EmbeddedTarget, text)
        @test result.value.value == "hello"
    end

    @testset "Parse with type coercion" begin
        @kwdef struct CoercionTarget
            count::Int
            score::Float64
            label::String
        end

        # JSON numbers might come as Float64 for Int fields
        json_text = """{"count": 5.0, "score": 3, "label": 42}"""
        result = parse_structured(CoercionTarget, json_text)
        @test result.value.count == 5
        @test result.value.score == 3.0
        @test result.value.label == "42"
    end

    @testset "Parse with nested struct" begin
        @kwdef struct Inner
            value::String
        end

        @kwdef struct Outer
            inner::Inner
            name::String
        end

        json_text = """{"inner": {"value": "deep"}, "name": "top"}"""
        result = parse_structured(Outer, json_text)
        @test result.value.name == "top"
        @test result.value.inner.value == "deep"
    end

    @testset "Parse with vectors" begin
        @kwdef struct VecTarget
            items::Vector{String}
            numbers::Vector{Int}
        end

        json_text = """{"items": ["a", "b", "c"], "numbers": [1, 2, 3]}"""
        result = parse_structured(VecTarget, json_text)
        @test result.value.items == ["a", "b", "c"]
        @test result.value.numbers == [1, 2, 3]
    end

    @testset "Parse from ChatResponse" begin
        @kwdef struct ChatTarget
            answer::String
        end

        resp = ChatResponse(
            messages = [Message(:assistant, """{"answer": "Paris"}""")],
        )
        result = parse_structured(ChatTarget, resp)
        @test result.value.answer == "Paris"
    end

    @testset "Parse from AgentResponse" begin
        @kwdef struct AgentTarget
            result::String
        end

        resp = AgentResponse(
            messages = [Message(:assistant, """{"result": "done"}""")],
        )
        result = parse_structured(AgentTarget, resp)
        @test result.value.result == "done"
    end

    @testset "Parse failure throws ContentError" begin
        @kwdef struct StrictTarget
            required_field::String
        end

        @test_throws ContentError parse_structured(StrictTarget, "not json at all {{{")
    end

    @testset "StructuredOutput display" begin
        @kwdef struct DisplayTarget
            x::Int
        end

        result = parse_structured(DisplayTarget, """{"x": 1}""")
        s = sprint(show, result)
        @test contains(s, "StructuredOutput")
        @test contains(s, "DisplayTarget")
    end
end
