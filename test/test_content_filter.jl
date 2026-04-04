# Tests for content filtering

using AgentFramework
using Test

@testset "Content Filtering" begin

    @testset "ContentFilterResult Construction" begin
        r = ContentFilterResult(category=FILTER_HATE, severity=FILTER_LOW, filtered=false)
        @test r.category == FILTER_HATE
        @test r.severity == FILTER_LOW
        @test r.filtered == false
        @test r.details === nothing
    end

    @testset "ContentFilterResults Aggregation" begin
        results = ContentFilterResults()
        @test !is_blocked(results)
        @test isempty(get_filtered_categories(results))
        @test max_severity(results) == FILTER_SAFE
    end

    @testset "Blocked Results" begin
        r1 = ContentFilterResult(category=FILTER_VIOLENCE, severity=FILTER_HIGH, filtered=true)
        r2 = ContentFilterResult(category=FILTER_HATE, severity=FILTER_LOW, filtered=false)
        results = ContentFilterResults(results=[r1, r2], blocked=true)
        @test is_blocked(results)
        @test get_filtered_categories(results) == [FILTER_VIOLENCE]
        @test max_severity(results) == FILTER_HIGH
    end

    @testset "Multiple Filtered Categories" begin
        r1 = ContentFilterResult(category=FILTER_VIOLENCE, severity=FILTER_HIGH, filtered=true)
        r2 = ContentFilterResult(category=FILTER_HATE, severity=FILTER_MEDIUM, filtered=true)
        r3 = ContentFilterResult(category=FILTER_SEXUAL, severity=FILTER_LOW, filtered=false)
        results = ContentFilterResults(results=[r1, r2, r3], blocked=true)
        cats = get_filtered_categories(results)
        @test length(cats) == 2
        @test FILTER_VIOLENCE in cats
        @test FILTER_HATE in cats
    end

    @testset "ContentFilteredException" begin
        results = ContentFilterResults(blocked=true, reason="content policy violation")
        e = ContentFilteredException("Blocked", results)
        @test e.message == "Blocked"
        @test is_blocked(e.results)
        buf = IOBuffer()
        showerror(buf, e)
        @test occursin("ContentFilteredException", String(take!(buf)))
    end

    @testset "Parse OpenAI Content Filter — Clean" begin
        data = Dict(
            "hate" => Dict("severity" => "safe", "filtered" => false),
            "violence" => Dict("severity" => "safe", "filtered" => false),
        )
        results = parse_openai_content_filter(data)
        @test !is_blocked(results)
        @test max_severity(results) == FILTER_SAFE
    end

    @testset "Parse OpenAI Content Filter — Blocked" begin
        data = Dict(
            "hate" => Dict("severity" => "high", "filtered" => true),
            "violence" => Dict("severity" => "low", "filtered" => false),
            "self_harm" => Dict("severity" => "safe", "filtered" => false),
        )
        results = parse_openai_content_filter(data)
        @test is_blocked(results)
        cats = get_filtered_categories(results)
        @test FILTER_HATE in cats
    end

    @testset "Parse OpenAI Content Filter — Boolean Format" begin
        data = Dict("jailbreak" => true)
        results = parse_openai_content_filter(data)
        @test is_blocked(results)
        @test FILTER_JAILBREAK in get_filtered_categories(results)
    end

    @testset "Parse OpenAI Content Filter — Empty" begin
        results = parse_openai_content_filter(Dict())
        @test !is_blocked(results)
        @test isempty(results.results)
    end

    @testset "Severity Ordering" begin
        @test FILTER_SAFE < FILTER_LOW
        @test FILTER_LOW < FILTER_MEDIUM
        @test FILTER_MEDIUM < FILTER_HIGH
    end

    @testset "Category Strings" begin
        @test AgentFramework.FILTER_CATEGORY_STRINGS[FILTER_HATE] == "hate"
        @test AgentFramework.FILTER_CATEGORY_STRINGS[FILTER_VIOLENCE] == "violence"
        @test AgentFramework.FILTER_CATEGORY_STRINGS[FILTER_JAILBREAK] == "jailbreak"
    end
end
