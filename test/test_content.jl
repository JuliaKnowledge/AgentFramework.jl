using AgentFramework
using Test

@testset "Content Types" begin
    @testset "text_content" begin
        c = text_content("hello world")
        @test c.type == AgentFramework.TEXT
        @test c.text == "hello world"
        @test is_text(c)
        @test !is_function_call(c)
        @test get_text(c) == "hello world"
    end

    @testset "data_content" begin
        c = data_content("base64data", "image/png")
        @test c.type == AgentFramework.DATA
        @test c.media_type == "image/png"
    end

    @testset "uri_content" begin
        c = uri_content("https://example.com/img.png"; media_type="image/png")
        @test c.type == AgentFramework.URI
        @test c.uri == "https://example.com/img.png"
        @test c.media_type == "image/png"
    end

    @testset "error_content" begin
        c = error_content("something failed"; error_code="E001")
        @test c.type == AgentFramework.ERROR_CONTENT
        @test c.message == "something failed"
        @test c.error_code == "E001"
    end

    @testset "function_call_content" begin
        c = function_call_content("call_1", "get_weather", """{"location": "London"}""")
        @test c.type == AgentFramework.FUNCTION_CALL
        @test is_function_call(c)
        @test c.call_id == "call_1"
        @test c.name == "get_weather"

        args = parse_arguments(c)
        @test args !== nothing
        @test args["location"] == "London"
    end

    @testset "function_result_content" begin
        c = function_result_content("call_1", "Sunny, 22°C")
        @test c.type == AgentFramework.FUNCTION_RESULT
        @test is_function_result(c)
        @test c.result == "Sunny, 22°C"
    end

    @testset "usage_content" begin
        ud = UsageDetails(input_tokens=10, output_tokens=20, total_tokens=30)
        c = usage_content(ud)
        @test c.type == AgentFramework.USAGE
        @test c.usage_details.total_tokens == 30
    end

    @testset "hosted_file_content" begin
        c = hosted_file_content("file-abc123")
        @test c.type == AgentFramework.HOSTED_FILE
        @test c.file_id == "file-abc123"
    end

    @testset "UsageDetails addition" begin
        a = UsageDetails(input_tokens=10, output_tokens=20)
        b = UsageDetails(input_tokens=5, output_tokens=10, total_tokens=15)
        merged = add_usage_details(a, b)
        @test merged.input_tokens == 15
        @test merged.output_tokens == 30
        @test merged.total_tokens == 15

        @test add_usage_details(nothing, b) === b
        @test add_usage_details(a, nothing) === a
        @test add_usage_details(nothing, nothing) === nothing
    end

    @testset "ContentType parsing" begin
        @test parse_content_type("text") == AgentFramework.TEXT
        @test parse_content_type("function_call") == AgentFramework.FUNCTION_CALL
        @test content_type_string(AgentFramework.TEXT) == "text"
        @test_throws ContentError parse_content_type("invalid_type")
    end

    @testset "Content serialization round-trip" begin
        original = function_call_content("call_1", "search", """{"q": "test"}""")
        d = content_to_dict(original)
        @test d["type"] == "function_call"
        @test d["name"] == "search"

        restored = content_from_dict(d)
        @test restored.type == original.type
        @test restored.name == original.name
        @test restored.call_id == original.call_id
    end

    @testset "Content equality" begin
        a = text_content("hello")
        b = text_content("hello")
        c = text_content("world")
        @test a == b
        @test a != c
    end

    @testset "Content show" begin
        c = text_content("hello")
        s = sprint(show, c)
        @test contains(s, "text")
        @test contains(s, "hello")
    end
end
