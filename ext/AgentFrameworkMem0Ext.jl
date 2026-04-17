module AgentFrameworkMem0Ext

using AgentFramework
using Mem0

using AgentFramework: BaseContextProvider, AgentSession, SessionContext,
                      Message, ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM,
                      extend_messages!, get_text

import AgentFramework: before_run!, after_run!

"""
    LocalMem0ContextProvider <: AgentFramework.BaseContextProvider

Context provider backed by a **local** `Mem0.Memory` instance from the
`Mem0.jl` package (SQLite vector store + optional Neo4j graph). This is
distinct from [`Mem0ContextProvider`](@ref AgentFramework.Mem0Integration),
which targets the **Mem0 cloud / SaaS** REST API at `api.mem0.ai`.

Use `LocalMem0ContextProvider` when you want fully-offline, self-hosted
agent memory; use the cloud-backed `Mem0ContextProvider` when you want
to delegate storage / retrieval to the Mem0 managed service.

# Construction

```julia
using AgentFramework, Mem0

mem = Memory()   # or Memory(config=MemoryConfig(...))
provider = LocalMem0ContextProvider(mem; user_id="alice")
```

# Behaviour

- **`before_run!`** — concatenates all user input messages into a search
  query, calls `Mem0.search` with the configured `user_id` / `agent_id`
  / `run_id`, renders the top-k memories as a markdown list, and
  injects the result via `extend_messages!`.
- **`after_run!`** — optionally calls `Mem0.add` on messages whose
  roles appear in `store_roles` (default `[:user, :assistant]`),
  persisting the turn into the local store.

# Fields

- `memory::Mem0.Memory` — underlying local memory store.
- `user_id`, `agent_id`, `run_id` — scope filters passed to
  `Mem0.search` / `Mem0.add`.
- `top_k::Int` — number of memories to retrieve per turn.
- `threshold::Union{Nothing, Float64}` — similarity cutoff (nil = no cutoff).
- `filters::Union{Nothing, Dict}` — extra `Mem0.search` filters.
- `context_prompt::String` — header inserted before the rendered memories.
- `store_roles::Vector{Symbol}` — message roles to persist in `after_run!`.
- `store::Bool` — master switch for `after_run!` persistence.
"""
mutable struct LocalMem0ContextProvider <: BaseContextProvider
    memory::Mem0.Memory
    user_id::Union{Nothing, String}
    agent_id::Union{Nothing, String}
    run_id::Union{Nothing, String}
    top_k::Int
    threshold::Union{Nothing, Float64}
    filters::Union{Nothing, Dict}
    context_prompt::String
    store_roles::Vector{Symbol}
    store::Bool
    search_fn::Function
    add_fn::Function
end

const _DEFAULT_PROMPT =
    "## Memories\nConsider the following memories retrieved from the local Mem0 store:"

function _nonempty(s)
    s === nothing && return nothing
    text = String(strip(String(s)))
    return isempty(text) ? nothing : text
end

function _normalize_role(role)::Symbol
    role isa Symbol && return role
    role isa AbstractString && return Symbol(lowercase(String(role)))
    return Symbol(string(role))
end

function LocalMem0ContextProvider(memory::Mem0.Memory;
                                  user_id = nothing,
                                  agent_id = nothing,
                                  run_id = nothing,
                                  top_k::Integer = 5,
                                  threshold::Union{Nothing, Real} = nothing,
                                  filters::Union{Nothing, Dict} = nothing,
                                  context_prompt::AbstractString = _DEFAULT_PROMPT,
                                  store_roles = [:user, :assistant],
                                  store::Bool = true,
                                  search_fn::Function = Mem0.search,
                                  add_fn::Function = Mem0.add)
    top_k > 0 || throw(ArgumentError("top_k must be positive"))
    roles = unique(_normalize_role.(collect(store_roles)))
    isempty(roles) && throw(ArgumentError("store_roles cannot be empty"))
    return LocalMem0ContextProvider(
        memory,
        _nonempty(user_id),
        _nonempty(agent_id),
        _nonempty(run_id),
        Int(top_k),
        threshold === nothing ? nothing : Float64(threshold),
        filters,
        String(context_prompt),
        roles,
        store,
        search_fn,
        add_fn,
    )
end

function Base.show(io::IO, p::LocalMem0ContextProvider)
    print(io, "LocalMem0ContextProvider(user_id=", repr(p.user_id),
              ", top_k=", p.top_k, ", store=", p.store, ")")
end

# ── Retrieval ────────────────────────────────────────────────────────────────

function _query_from_inputs(ctx::SessionContext)::Union{Nothing, String}
    parts = String[]
    for msg in ctx.input_messages
        t = strip(get_text(msg))
        isempty(t) || push!(parts, String(t))
    end
    isempty(parts) && return nothing
    return join(parts, "\n")
end

function _extract_mem_records(result)::Vector{Dict{String, Any}}
    out = Dict{String, Any}[]
    result isa AbstractDict || return out
    raw = get(result, "results", nothing)
    raw isa AbstractVector || return out
    for r in raw
        if r isa AbstractDict
            push!(out, Dict{String, Any}(string(k) => v for (k, v) in pairs(r)))
        end
    end
    return out
end

function _format_memories(records::Vector{Dict{String, Any}}, prompt::String)::Union{Nothing, String}
    isempty(records) && return nothing
    lines = String[]
    for (i, r) in enumerate(records)
        mem = get(r, "memory", get(r, "text", ""))
        mem === nothing && continue
        text = strip(String(mem))
        isempty(text) && continue
        push!(lines, string("- ", text))
    end
    isempty(lines) && return nothing
    return string(prompt, "\n", join(lines, "\n"))
end

function before_run!(
    provider::LocalMem0ContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    query = _query_from_inputs(ctx)
    query === nothing && return nothing
    state["last_query"] = query

    result = provider.search_fn(
        provider.memory, query;
        user_id = provider.user_id,
        agent_id = provider.agent_id,
        run_id = provider.run_id,
        limit = provider.top_k,
        filters = provider.filters,
        threshold = provider.threshold,
    )
    records = _extract_mem_records(result)
    state["last_result_count"] = length(records)

    body = _format_memories(records, provider.context_prompt)
    body === nothing && return nothing

    extend_messages!(ctx, provider, [Message(ROLE_USER, body)])
    return nothing
end

# ── Persistence ──────────────────────────────────────────────────────────────

function _collect_messages_for_store(provider::LocalMem0ContextProvider, ctx::SessionContext)
    entries = Tuple{String, String}[]  # (role, text)
    pairs = Any[]
    for msg in ctx.input_messages
        role = _normalize_role(msg.role)
        role in provider.store_roles || continue
        text = strip(get_text(msg))
        isempty(text) && continue
        push!(pairs, Dict("role" => String(role), "content" => String(text)))
    end
    # Include the assistant's output_messages if :assistant is in store_roles.
    if :assistant in provider.store_roles && hasproperty(ctx, :output_messages)
        for msg in getproperty(ctx, :output_messages)
            role = _normalize_role(msg.role)
            role == :assistant || continue
            text = strip(get_text(msg))
            isempty(text) && continue
            push!(pairs, Dict("role" => "assistant", "content" => String(text)))
        end
    end
    return pairs
end

function after_run!(
    provider::LocalMem0ContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    provider.store || return nothing
    messages = _collect_messages_for_store(provider, ctx)
    isempty(messages) && return nothing
    try
        provider.add_fn(
            provider.memory, messages;
            user_id = provider.user_id,
            agent_id = provider.agent_id,
            run_id = provider.run_id,
        )
    catch e
        @warn "LocalMem0ContextProvider: Mem0.add failed" exception=(e, catch_backtrace())
    end
    state["persisted_turn"] = get(state, "persisted_turn", 0) + 1
    return nothing
end

export LocalMem0ContextProvider

end # module AgentFrameworkMem0Ext
