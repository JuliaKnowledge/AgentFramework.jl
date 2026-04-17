using AgentFramework
using Test

# Clear the once-per-process warning registry so test ordering doesn't
# swallow warnings from earlier fixtures.
empty!(AgentFramework._FEATURE_STAGE_WARNED)

@testset "feature_stage" begin
    @testset "experimental function emits warning once and registers" begin
        @experimental feature_id=:TEST_EXP1 function _exp_fn(x)
            return 2x
        end

        @test AgentFramework.feature_stage(_exp_fn) == (:experimental, :TEST_EXP1)

        # First call warns; result is correct.
        result = @test_logs (:warn, r"TEST_EXP1") _exp_fn(5)
        @test result == 10

        # Second call does NOT emit the warning again.
        @test_logs _exp_fn(6)
    end

    @testset "release_candidate function registers but does not warn" begin
        @release_candidate feature_id=:TEST_RC1 function _rc_fn(x)
            return x + 1
        end

        @test AgentFramework.feature_stage(_rc_fn) == (:release_candidate, :TEST_RC1)
        # No runtime warning expected.
        @test_logs _rc_fn(3)
    end

    @testset "experimental struct registers metadata" begin
        @experimental feature_id=:TEST_EXP2 struct _ExpThing
            value::Int
        end

        @test AgentFramework.feature_stage(_ExpThing) == (:experimental, :TEST_EXP2)
        # Constructor still works normally.
        thing = _ExpThing(7)
        @test thing.value == 7
    end

    @testset "experimental mutable struct registers metadata" begin
        @experimental feature_id=:TEST_EXP3 mutable struct _ExpMutable
            counter::Int
        end

        @test AgentFramework.feature_stage(_ExpMutable) == (:experimental, :TEST_EXP3)
        obj = _ExpMutable(0)
        obj.counter += 1
        @test obj.counter == 1
    end

    @testset "experimental abstract type registers metadata" begin
        @experimental feature_id=:TEST_EXP4 abstract type _ExpAbstract end
        @test AgentFramework.feature_stage(_ExpAbstract) == (:experimental, :TEST_EXP4)
    end

    @testset "ExperimentalFeature inventory module is accessible" begin
        @test ExperimentalFeature.EVALS == :EVALS
        @test ExperimentalFeature.SKILLS == :SKILLS
    end

    @testset "feature_stage returns nothing for unmarked objects" begin
        @test AgentFramework.feature_stage(sum) === nothing
    end

    @testset "missing feature_id raises at macroexpand time" begin
        # Expanding without feature_id should throw ArgumentError.
        @test_throws LoadError @eval @experimental function _bad() end
    end
end
