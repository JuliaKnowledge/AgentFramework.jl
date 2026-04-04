# Scoped state management for workflow executors
# Implements executor-local state and broadcast scope (from C# patterns).

"""
    StateScope

Scope level for workflow state values.
"""
@enum StateScope begin
    SCOPE_LOCAL       # Only visible to the owning executor
    SCOPE_BROADCAST   # Visible to all executors (read-only for non-owners)
    SCOPE_WORKFLOW    # Global workflow-level state
end

"""
    ScopedValue{T}

A state value with scope metadata and ownership tracking.
"""
Base.@kwdef mutable struct ScopedValue
    value::Any
    scope::StateScope = SCOPE_LOCAL
    owner::String = ""  # executor ID that owns this value
    version::Int = 0
    updated_at::Float64 = time()
end

"""
    ScopedStateStore

State store that enforces scope-based access control.

Executors can:
- Read/write their own SCOPE_LOCAL values
- Read/write their own SCOPE_BROADCAST values
- Read (not write) other executors' SCOPE_BROADCAST values
- Read/write SCOPE_WORKFLOW values
"""
Base.@kwdef mutable struct ScopedStateStore
    local_state::Dict{String, Dict{String, ScopedValue}} = Dict()   # executor_id => key => value
    broadcast_state::Dict{String, ScopedValue} = Dict()              # key => value (with owner)
    workflow_state::Dict{String, ScopedValue} = Dict()               # key => value
    lock::ReentrantLock = ReentrantLock()
end

# ── Local State ──────────────────────────────────────────────────────────────

"""
    get_local(store, executor_id, key; default=nothing) -> Any

Get an executor-local state value.
"""
function get_local(store::ScopedStateStore, executor_id::String, key::String; default=nothing)
    lock(store.lock) do
        exec_state = get(store.local_state, executor_id, nothing)
        exec_state === nothing && return default
        sv = get(exec_state, key, nothing)
        sv === nothing && return default
        return sv.value
    end
end

"""
    set_local!(store, executor_id, key, value)

Set an executor-local state value.
"""
function set_local!(store::ScopedStateStore, executor_id::String, key::String, value)
    lock(store.lock) do
        if !haskey(store.local_state, executor_id)
            store.local_state[executor_id] = Dict{String, ScopedValue}()
        end
        store.local_state[executor_id][key] = ScopedValue(
            value=value, scope=SCOPE_LOCAL, owner=executor_id,
            version=_next_version(store.local_state[executor_id], key),
            updated_at=time(),
        )
    end
end

# ── Broadcast State ──────────────────────────────────────────────────────────

"""
    get_broadcast(store, key; default=nothing) -> Any

Get a broadcast state value (readable by all executors).
"""
function get_broadcast(store::ScopedStateStore, key::String; default=nothing)
    lock(store.lock) do
        sv = get(store.broadcast_state, key, nothing)
        sv === nothing && return default
        return sv.value
    end
end

"""
    set_broadcast!(store, executor_id, key, value)

Set a broadcast state value. Only the original owner can update it.
Throws WorkflowError if a different executor tries to write.
"""
function set_broadcast!(store::ScopedStateStore, executor_id::String, key::String, value)
    lock(store.lock) do
        existing = get(store.broadcast_state, key, nothing)
        if existing !== nothing && existing.owner != executor_id
            throw(WorkflowError("Executor '$executor_id' cannot modify broadcast state '$key' owned by '$(existing.owner)'"))
        end
        store.broadcast_state[key] = ScopedValue(
            value=value, scope=SCOPE_BROADCAST, owner=executor_id,
            version=_next_version(store.broadcast_state, key),
            updated_at=time(),
        )
    end
end

# ── Workflow State ───────────────────────────────────────────────────────────

"""
    get_workflow_state(store, key; default=nothing) -> Any

Get a workflow-level state value (readable/writable by all).
"""
function get_workflow_state(store::ScopedStateStore, key::String; default=nothing)
    lock(store.lock) do
        sv = get(store.workflow_state, key, nothing)
        sv === nothing && return default
        return sv.value
    end
end

"""
    set_workflow_state!(store, key, value; executor_id="")

Set a workflow-level state value.
"""
function set_workflow_state!(store::ScopedStateStore, key::String, value; executor_id::String="")
    lock(store.lock) do
        store.workflow_state[key] = ScopedValue(
            value=value, scope=SCOPE_WORKFLOW, owner=executor_id,
            version=_next_version(store.workflow_state, key),
            updated_at=time(),
        )
    end
end

# ── Queries ──────────────────────────────────────────────────────────────────

"""
    list_broadcast_keys(store) -> Vector{String}

List all broadcast state keys.
"""
function list_broadcast_keys(store::ScopedStateStore)::Vector{String}
    lock(store.lock) do
        return collect(keys(store.broadcast_state))
    end
end

"""
    list_local_keys(store, executor_id) -> Vector{String}

List all local state keys for an executor.
"""
function list_local_keys(store::ScopedStateStore, executor_id::String)::Vector{String}
    lock(store.lock) do
        exec_state = get(store.local_state, executor_id, nothing)
        exec_state === nothing && return String[]
        return collect(keys(exec_state))
    end
end

"""
    clear_executor_state!(store, executor_id)

Clear all local state for an executor.
"""
function clear_executor_state!(store::ScopedStateStore, executor_id::String)
    lock(store.lock) do
        delete!(store.local_state, executor_id)
        # Also remove any broadcast state owned by this executor
        for (key, sv) in collect(store.broadcast_state)
            if sv.owner == executor_id
                delete!(store.broadcast_state, key)
            end
        end
    end
end

"""
    snapshot(store) -> Dict{String, Any}

Create a serializable snapshot of all state (for checkpointing).
"""
function snapshot(store::ScopedStateStore)::Dict{String, Any}
    lock(store.lock) do
        return Dict{String, Any}(
            "local" => Dict(eid => Dict(k => sv.value for (k, sv) in kvs) for (eid, kvs) in store.local_state),
            "broadcast" => Dict(k => Dict("value" => sv.value, "owner" => sv.owner) for (k, sv) in store.broadcast_state),
            "workflow" => Dict(k => sv.value for (k, sv) in store.workflow_state),
        )
    end
end

# ── Internal ─────────────────────────────────────────────────────────────────

function _next_version(state_dict::Dict{String, ScopedValue}, key::String)::Int
    existing = get(state_dict, key, nothing)
    return existing === nothing ? 1 : existing.version + 1
end
