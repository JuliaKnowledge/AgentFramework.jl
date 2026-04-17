using Test
using AgentFramework

@testset "AgentFrameworkMem0Ext (local Mem0.jl)" begin
    if Base.find_package("Mem0") === nothing
        @test_skip "Mem0.jl not available — skipping local-Mem0 extension tests"
        return
    end

    @eval Main using Mem0
    Mem0 = Main.Mem0

    ext = Base.get_extension(AgentFramework, :AgentFrameworkMem0Ext)
    @test ext !== nothing
    LocalProv = ext.LocalMem0ContextProvider

    # --- Minimal mock Mem0.Memory (no network) -----------------------------
    struct _StubLLM <: Mem0.AbstractLLM end
    struct _StubEmb <: Mem0.AbstractEmbedder
        dims::Int
    end
    _StubEmb() = _StubEmb(8)

    vs = Mem0.InMemoryVectorStore(collection_name="test", embedding_model_dims=8)
    mem = Mem0.Memory(
        Mem0.MemoryConfig(),
        _StubLLM(), _StubEmb(), vs, Mem0.HistoryManager(":memory:"),
        nothing, false, "test", nothing, nothing,
    )

    # --- Stubbed search/add so we don't need real LLM/embedder calls -------
    search_calls = Ref(0)
    add_calls = Ref(0)
    last_query = Ref{String}("")
    last_add_msgs = Ref{Any}(nothing)

    stub_search = function(_mem, query; kwargs...)
        search_calls[] += 1
        last_query[] = query
        return Dict{String, Any}(
            "results" => Any[
                Dict("memory" => "user likes haskell"),
                Dict("memory" => "user is based in edinburgh"),
            ],
        )
    end
    stub_add = function(_mem, messages; kwargs...)
        add_calls[] += 1
        last_add_msgs[] = messages
        return Dict{String, Any}("results" => Any[])
    end

    provider = LocalProv(
        mem;
        user_id = "alice",
        top_k = 3,
        search_fn = stub_search,
        add_fn = stub_add,
    )
    @test provider isa AgentFramework.BaseContextProvider

    # Build a SessionContext with an input message and invoke before_run!.
    session = AgentFramework.AgentSession(id = "s1")
    ctx = AgentFramework.SessionContext(
        input_messages = [AgentFramework.Message(AgentFramework.ROLE_USER, "where do i live?")],
    )
    state = Dict{String, Any}()

    AgentFramework.before_run!(provider, nothing, session, ctx, state)

    @test search_calls[] == 1
    @test occursin("where do i live?", last_query[])
    @test state["last_result_count"] == 2

    # The memories should have been injected via extend_messages!.
    injected = reduce(vcat, values(ctx.context_messages); init=AgentFramework.Message[])
    @test length(injected) == 1
    txt = AgentFramework.get_text(injected[1])
    @test occursin("haskell", txt)
    @test occursin("edinburgh", txt)

    # after_run! should persist input user message (at least).
    AgentFramework.after_run!(provider, nothing, session, ctx, state)
    @test add_calls[] == 1
    msgs = last_add_msgs[]
    @test msgs isa AbstractVector
    @test any(m -> get(m, "role", "") == "user" && occursin("where do i live?", get(m, "content", "")), msgs)

    # store=false should short-circuit after_run!.
    provider2 = LocalProv(mem; user_id="alice", store=false,
                          search_fn=stub_search, add_fn=stub_add)
    add_calls_before = add_calls[]
    AgentFramework.after_run!(provider2, nothing, session, ctx, Dict{String, Any}())
    @test add_calls[] == add_calls_before  # no additional call
end
