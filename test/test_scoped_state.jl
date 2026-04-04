# Tests for scoped state management

using AgentFramework
using Test

@testset "Scoped State" begin

    @testset "Local State" begin
        store = ScopedStateStore()
        set_local!(store, "exec1", "counter", 0)
        @test get_local(store, "exec1", "counter") == 0

        set_local!(store, "exec1", "counter", 42)
        @test get_local(store, "exec1", "counter") == 42

        # Different executor can't see it
        @test get_local(store, "exec2", "counter") === nothing
        @test get_local(store, "exec2", "counter"; default=0) == 0
    end

    @testset "Local State Keys" begin
        store = ScopedStateStore()
        set_local!(store, "exec1", "a", 1)
        set_local!(store, "exec1", "b", 2)
        set_local!(store, "exec2", "c", 3)

        keys1 = list_local_keys(store, "exec1")
        @test Set(keys1) == Set(["a", "b"])

        keys2 = list_local_keys(store, "exec2")
        @test keys2 == ["c"]

        @test isempty(list_local_keys(store, "exec3"))
    end

    @testset "Broadcast State" begin
        store = ScopedStateStore()
        set_broadcast!(store, "exec1", "shared_config", "value1")
        @test get_broadcast(store, "shared_config") == "value1"

        # Owner can update
        set_broadcast!(store, "exec1", "shared_config", "value2")
        @test get_broadcast(store, "shared_config") == "value2"

        # Other executor cannot write
        @test_throws WorkflowError set_broadcast!(store, "exec2", "shared_config", "hijack")

        # But can read
        @test get_broadcast(store, "shared_config") == "value2"
    end

    @testset "Broadcast State Keys" begin
        store = ScopedStateStore()
        set_broadcast!(store, "exec1", "key1", "v1")
        set_broadcast!(store, "exec2", "key2", "v2")

        keys = list_broadcast_keys(store)
        @test Set(keys) == Set(["key1", "key2"])
    end

    @testset "Workflow State" begin
        store = ScopedStateStore()
        set_workflow_state!(store, "iteration", 0)
        @test get_workflow_state(store, "iteration") == 0

        set_workflow_state!(store, "iteration", 1; executor_id="exec1")
        @test get_workflow_state(store, "iteration") == 1

        # Any executor can write
        set_workflow_state!(store, "iteration", 2; executor_id="exec2")
        @test get_workflow_state(store, "iteration") == 2

        @test get_workflow_state(store, "nonexistent") === nothing
        @test get_workflow_state(store, "nonexistent"; default="fallback") == "fallback"
    end

    @testset "Clear Executor State" begin
        store = ScopedStateStore()
        set_local!(store, "exec1", "a", 1)
        set_local!(store, "exec1", "b", 2)
        set_broadcast!(store, "exec1", "shared", "data")
        set_local!(store, "exec2", "c", 3)

        clear_executor_state!(store, "exec1")

        @test get_local(store, "exec1", "a") === nothing
        @test get_local(store, "exec1", "b") === nothing
        @test get_broadcast(store, "shared") === nothing  # broadcast cleared too
        @test get_local(store, "exec2", "c") == 3  # untouched
    end

    @testset "Snapshot" begin
        store = ScopedStateStore()
        set_local!(store, "exec1", "count", 10)
        set_broadcast!(store, "exec1", "config", "abc")
        set_workflow_state!(store, "phase", "running")

        snap = AgentFramework.snapshot(store)
        @test snap isa Dict{String, Any}
        @test haskey(snap, "local")
        @test haskey(snap, "broadcast")
        @test haskey(snap, "workflow")
        @test snap["local"]["exec1"]["count"] == 10
        @test snap["broadcast"]["config"]["value"] == "abc"
        @test snap["broadcast"]["config"]["owner"] == "exec1"
        @test snap["workflow"]["phase"] == "running"
    end

    @testset "Thread Safety — Concurrent Access" begin
        store = ScopedStateStore()
        n = 100
        @sync begin
            for i in 1:n
                Threads.@spawn begin
                    set_local!(store, "exec1", "key_$i", i)
                end
            end
        end
        # All values should be set
        found = 0
        for i in 1:n
            val = get_local(store, "exec1", "key_$i")
            val !== nothing && (found += 1)
        end
        @test found == n
    end

    @testset "ScopedValue Versioning" begin
        store = ScopedStateStore()
        set_local!(store, "exec1", "x", "v1")
        set_local!(store, "exec1", "x", "v2")
        set_local!(store, "exec1", "x", "v3")
        # Value should be latest
        @test get_local(store, "exec1", "x") == "v3"
    end

    @testset "Scope Enum Values" begin
        @test SCOPE_LOCAL != SCOPE_BROADCAST
        @test SCOPE_BROADCAST != SCOPE_WORKFLOW
        @test SCOPE_LOCAL != SCOPE_WORKFLOW
    end
end
