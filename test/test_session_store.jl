using AgentFramework
using Test

Base.@kwdef struct StoredState
    label::String
    count::Int
end

@testset "Session Store" begin

    # ── InMemorySessionStore ─────────────────────────────────────────────

    @testset "InMemorySessionStore — save and load" begin
        store = InMemorySessionStore()
        session = AgentSession(id="test-1", state=Dict{String, Any}("key" => "value"))
        save_session!(store, session)
        loaded = load_session(store, "test-1")
        @test loaded !== nothing
        @test loaded.id == "test-1"
        @test loaded.state["key"] == "value"
    end

    @testset "InMemorySessionStore — load nonexistent returns nothing" begin
        store = InMemorySessionStore()
        @test load_session(store, "nonexistent") === nothing
    end

    @testset "InMemorySessionStore — delete existing" begin
        store = InMemorySessionStore()
        session = AgentSession(id="del-1")
        save_session!(store, session)
        @test delete_session!(store, "del-1") == true
        @test load_session(store, "del-1") === nothing
    end

    @testset "InMemorySessionStore — delete nonexistent returns false" begin
        store = InMemorySessionStore()
        @test delete_session!(store, "nope") == false
    end

    @testset "InMemorySessionStore — list_sessions" begin
        store = InMemorySessionStore()
        save_session!(store, AgentSession(id="a"))
        save_session!(store, AgentSession(id="b"))
        ids = list_sessions(store)
        @test sort(ids) == ["a", "b"]
    end

    @testset "InMemorySessionStore — has_session" begin
        store = InMemorySessionStore()
        save_session!(store, AgentSession(id="exists"))
        @test has_session(store, "exists") == true
        @test has_session(store, "missing") == false
    end

    # ── FileSessionStore ─────────────────────────────────────────────────

    @testset "FileSessionStore — save and load roundtrip" begin
        mktempdir() do dir
            store = FileSessionStore(dir)
            session = AgentSession(id="file-1", state=Dict{String, Any}("x" => 42), user_id="user-a")
            save_session!(store, session)
            loaded = load_session(store, "file-1")
            @test loaded !== nothing
            @test loaded isa AgentSession
            @test loaded.id == "file-1"
            @test loaded.user_id == "user-a"
        end
    end

    @testset "FileSessionStore — load nonexistent returns nothing" begin
        mktempdir() do dir
            store = FileSessionStore(dir)
            @test load_session(store, "nope") === nothing
        end
    end

    @testset "FileSessionStore — delete" begin
        mktempdir() do dir
            store = FileSessionStore(dir)
            save_session!(store, AgentSession(id="del-f"))
            @test delete_session!(store, "del-f") == true
            @test delete_session!(store, "del-f") == false
        end
    end

    @testset "FileSessionStore — list_sessions" begin
        mktempdir() do dir
            store = FileSessionStore(dir)
            save_session!(store, AgentSession(id="s1"))
            save_session!(store, AgentSession(id="s2"))
            ids = sort(list_sessions(store))
            @test ids == ["s1", "s2"]
        end
    end

    @testset "FileSessionStore — session state preserved" begin
        mktempdir() do dir
            store = FileSessionStore(dir)
            session = AgentSession(
                id="state-test",
                state=Dict{String, Any}("count" => 5, "name" => "test"),
                metadata=Dict{String, Any}("created" => "now"),
                thread_id="thread-123"
            )
            save_session!(store, session)
            loaded = load_session(store, "state-test")
            @test loaded.thread_id == "thread-123"
            @test loaded.metadata["created"] == "now"
        end
    end

    @testset "FileSessionStore — registered state types roundtrip" begin
        mktempdir() do dir
            register_state_type!(StoredState)
            store = FileSessionStore(dir)
            session = AgentSession(
                id = "typed-file",
                state = Dict{String, Any}(
                    "stored" => StoredState(label="alpha", count=2),
                ),
            )
            save_session!(store, session)
            loaded = load_session(store, "typed-file")
            @test loaded !== nothing
            @test loaded.state["stored"] isa StoredState
            @test loaded.state["stored"].label == "alpha"
            @test loaded.state["stored"].count == 2
        end
    end

    @testset "Thread safety with concurrent saves" begin
        store = InMemorySessionStore()
        tasks = Task[]
        for i in 1:50
            t = @async begin
                s = AgentSession(id="concurrent-$i")
                save_session!(store, s)
            end
            push!(tasks, t)
        end
        for t in tasks
            wait(t)
        end
        @test length(list_sessions(store)) == 50
    end

end
