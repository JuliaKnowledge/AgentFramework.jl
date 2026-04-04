using AgentFramework
using Test

@testset "Settings" begin

    # ── SecretString ─────────────────────────────────────────────────────

    @testset "SecretString redacts on show" begin
        s = SecretString("my-api-key")
        buf = IOBuffer()
        show(buf, s)
        @test String(take!(buf)) == "SecretString(\"***\")"
    end

    @testset "SecretString.value accessible" begin
        s = SecretString("secret-value")
        @test s.value == "secret-value"
        @test string(s) == "secret-value"
    end

    @testset "SecretString isempty" begin
        @test isempty(SecretString("")) == true
        @test isempty(SecretString("x")) == false
        @test length(SecretString("abc")) == 3
    end

    # ── Settings construction ────────────────────────────────────────────

    @testset "Settings construction" begin
        s = Settings()
        @test isempty(s.values)
        s2 = Settings(values=Dict("A" => "1"))
        @test s2.values["A"] == "1"
    end

    # ── load_from_env! ───────────────────────────────────────────────────

    @testset "load_from_env! with prefix" begin
        # Set env vars with prefix
        ENV["AGENTFRAMEWORK_TEST_KEY"] = "test_value"
        ENV["AGENTFRAMEWORK_OTHER"] = "other_value"
        try
            s = Settings()
            load_from_env!(s)
            @test s.values["TEST_KEY"] == "test_value"
            @test s.values["OTHER"] == "other_value"
        finally
            delete!(ENV, "AGENTFRAMEWORK_TEST_KEY")
            delete!(ENV, "AGENTFRAMEWORK_OTHER")
        end
    end

    # ── load_from_dotenv! ────────────────────────────────────────────────

    @testset "load_from_dotenv! basic KEY=VALUE" begin
        mktempdir() do dir
            path = joinpath(dir, ".env")
            write(path, "API_KEY=abc123\nDB_HOST=localhost\n")
            s = Settings()
            load_from_dotenv!(s, path)
            @test s.values["API_KEY"] == "abc123"
            @test s.values["DB_HOST"] == "localhost"
        end
    end

    @testset "load_from_dotenv! with comments" begin
        mktempdir() do dir
            path = joinpath(dir, ".env")
            write(path, "# This is a comment\nKEY=value\n# Another comment\n\nKEY2=value2\n")
            s = Settings()
            load_from_dotenv!(s, path)
            @test length(s.values) == 2
            @test s.values["KEY"] == "value"
            @test s.values["KEY2"] == "value2"
        end
    end

    @testset "load_from_dotenv! with quoted values" begin
        mktempdir() do dir
            path = joinpath(dir, ".env")
            write(path, "DOUBLE=\"hello world\"\nSINGLE='single quoted'\n")
            s = Settings()
            load_from_dotenv!(s, path)
            @test s.values["DOUBLE"] == "hello world"
            @test s.values["SINGLE"] == "single quoted"
        end
    end

    @testset "load_from_dotenv! nonexistent file — no error" begin
        s = Settings()
        result = load_from_dotenv!(s, "/tmp/nonexistent_env_file_12345")
        @test result === s
        @test isempty(s.values)
    end

    # ── get_setting / get_secret / has_setting ───────────────────────────

    @testset "get_setting with default" begin
        s = Settings(values=Dict("A" => "1"))
        @test get_setting(s, "A") == "1"
        @test get_setting(s, "B") === nothing
        @test get_setting(s, "B", "fallback") == "fallback"
    end

    @testset "get_secret returns SecretString" begin
        s = Settings(values=Dict("TOKEN" => "secret123"))
        secret = get_secret(s, "TOKEN")
        @test secret !== nothing
        @test secret isa SecretString
        @test secret.value == "secret123"
        @test get_secret(s, "MISSING") === nothing
    end

    @testset "has_setting" begin
        s = Settings(values=Dict("EXISTS" => "yes"))
        @test has_setting(s, "EXISTS") == true
        @test has_setting(s, "NOPE") == false
    end

    # ── load_from_toml! ──────────────────────────────────────────────────

    @testset "load_from_toml! basic" begin
        mktempdir() do dir
            path = joinpath(dir, "config.toml")
            write(path, """
            # Config file
            [settings]
            model = "gpt-4"
            temperature = "0.7"
            api_key = "sk-test"
            """)
            s = Settings()
            load_from_toml!(s, path)
            @test s.values["model"] == "gpt-4"
            @test s.values["temperature"] == "0.7"
            @test s.values["api_key"] == "sk-test"
        end
    end

end
