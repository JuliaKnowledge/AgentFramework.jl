# Edge types for AgentFramework.jl workflows.
# Edges define routing between executors in the workflow DAG.
# Uses tagged-union pattern inspired by C# (DirectEdgeData, FanOutEdgeData, FanInEdgeData).

"""
    EdgeKind

Discriminator for edge routing types.
"""
@enum EdgeKind begin
    DIRECT_EDGE      # 1:1 routing with optional condition
    FAN_OUT_EDGE     # 1:N broadcast
    FAN_IN_EDGE      # N:1 aggregation
end

"""
    Edge

A directed connection between executors with optional conditional routing.

# Fields
- `source_id::String`: Source executor ID.
- `target_id::String`: Target executor ID.
- `condition::Union{Nothing, Function}`: Optional predicate `(data) -> Bool`.
- `condition_name::Union{Nothing, String}`: Human-readable condition name.
"""
Base.@kwdef struct Edge
    source_id::String
    target_id::String
    condition::Union{Nothing, Function} = nothing
    condition_name::Union{Nothing, String} = nothing
end

function Base.show(io::IO, e::Edge)
    print(io, e.source_id, " → ", e.target_id)
    e.condition !== nothing && print(io, " [", something(e.condition_name, "conditional"), "]")
end

"""Test whether a message should be routed along this edge."""
function should_route(edge::Edge, data)::Bool
    edge.condition === nothing && return true
    return edge.condition(data)::Bool
end

# ── Edge Groups ──────────────────────────────────────────────────────────────

"""
    EdgeGroup

A group of edges that defines a routing pattern between executors.
This is the tagged-union container for different routing strategies.

# Variants (determined by `kind`)
- `DIRECT_EDGE`: Single edge, 1:1 routing.
- `FAN_OUT_EDGE`: One source to multiple targets (broadcast).
- `FAN_IN_EDGE`: Multiple sources to one target (aggregation).

# Fields
- `kind::EdgeKind`: Routing strategy.
- `edges::Vector{Edge}`: The edges in this group.
- `id::String`: Group identifier.
- `selection_func::Union{Nothing, Function}`: For FAN_OUT, optional target filter `(data, target_ids) -> Vector{String}`.
"""
Base.@kwdef struct EdgeGroup
    kind::EdgeKind
    edges::Vector{Edge}
    id::String = string(UUIDs.uuid4())
    selection_func::Union{Nothing, Function} = nothing
end

function Base.show(io::IO, g::EdgeGroup)
    print(io, "EdgeGroup(", g.kind, ", ", length(g.edges), " edges)")
end

"""Get all unique source executor IDs in this group."""
source_executor_ids(g::EdgeGroup) = unique([e.source_id for e in g.edges])

"""Get all unique target executor IDs in this group."""
target_executor_ids(g::EdgeGroup) = unique([e.target_id for e in g.edges])

# ── Edge Group Constructors ──────────────────────────────────────────────────

"""
    direct_edge(source_id, target_id; condition=nothing, condition_name=nothing) -> EdgeGroup

Create a direct 1:1 edge between two executors.

# Example
```julia
edge = direct_edge("processor", "output")
edge_conditional = direct_edge("router", "handler_a"; condition = d -> d isa String)
```
"""
function direct_edge(
    source_id::String,
    target_id::String;
    condition::Union{Nothing, Function} = nothing,
    condition_name::Union{Nothing, String} = nothing,
)::EdgeGroup
    EdgeGroup(
        kind = DIRECT_EDGE,
        edges = [Edge(source_id=source_id, target_id=target_id, condition=condition, condition_name=condition_name)],
    )
end

"""
    fan_out_edge(source_id, target_ids; selection_func=nothing) -> EdgeGroup

Create a fan-out edge: one source broadcasts to multiple targets.

# Example
```julia
edge = fan_out_edge("splitter", ["worker_1", "worker_2", "worker_3"])
# With selective routing:
edge = fan_out_edge("router", ["fast", "slow"]; selection_func = (data, ids) -> data.priority == :high ? ["fast"] : ["slow"])
```
"""
function fan_out_edge(
    source_id::String,
    target_ids::Vector{String};
    selection_func::Union{Nothing, Function} = nothing,
)::EdgeGroup
    edges = [Edge(source_id=source_id, target_id=tid) for tid in target_ids]
    EdgeGroup(kind=FAN_OUT_EDGE, edges=edges, selection_func=selection_func)
end

"""
    fan_in_edge(source_ids, target_id) -> EdgeGroup

Create a fan-in edge: multiple sources converge to one target.
The target executor receives an aggregated message once ALL sources have sent.

# Example
```julia
edge = fan_in_edge(["analyzer_1", "analyzer_2"], "aggregator")
```
"""
function fan_in_edge(
    source_ids::Vector{String},
    target_id::String,
)::EdgeGroup
    edges = [Edge(source_id=sid, target_id=target_id) for sid in source_ids]
    EdgeGroup(kind=FAN_IN_EDGE, edges=edges)
end

"""
    switch_edge(source_id, cases::Vector{Pair{Function, String}}; default=nothing) -> Vector{EdgeGroup}

Create *mutually exclusive* conditional routing from one source to multiple targets.
Cases are evaluated in order: a message is routed to the target of the FIRST matching
case, and to `default` only if no case matches. This mirrors Python's
`SwitchCaseEdgeGroup`, where exactly one branch is selected. Returns a vector of
EdgeGroups (one per case, plus default), each carrying an exclusion-aware condition so
overlapping cases can never both fire.

# Example
```julia
edges = switch_edge("classifier", [
    (d -> d.score > 0.8) => "high_confidence",
    (d -> d.score > 0.5) => "medium_confidence",
]; default="low_confidence")
# A message with score 0.9 routes ONLY to "high_confidence" (not also "medium_confidence").
```
"""
function switch_edge(
    source_id::String,
    cases::Vector{Pair{Function, String}};
    default::Union{Nothing, String} = nothing,
)::Vector{EdgeGroup}
    groups = EdgeGroup[]
    preceding = Function[]
    for (index, (condition, target_id)) in enumerate(cases)
        # This case fires only when it matches AND no earlier case matched.
        # A `let` block gives each closure its own fresh bindings (avoids the
        # closure-over-loop-variable boxing that would make every case share the
        # last iteration's `own`/`prior`).
        exclusive_condition = let own = condition, prior = copy(preceding)
            (data) -> own(data)::Bool && !any(c -> c(data)::Bool, prior)
        end
        push!(groups, direct_edge(source_id, target_id; condition=exclusive_condition, condition_name="case_$(index)"))
        push!(preceding, condition)
    end
    if default !== nothing
        # Default edge: matches only when no case condition does.
        default_condition = let prior = copy(preceding)
            (data) -> !any(c -> c(data)::Bool, prior)
        end
        push!(groups, direct_edge(source_id, default; condition=default_condition, condition_name="default"))
    end
    return groups
end

# ── Edge Routing Logic ───────────────────────────────────────────────────────

"""
    route_messages(group::EdgeGroup, messages::Vector{WorkflowMessage}) -> Dict{String, Vector{Any}}

Stateless, single-batch routing of `messages` through an edge group, returning a dict of
`target_id => [payloads]`.

For `DIRECT_EDGE` and `FAN_OUT_EDGE` this is the routing used by the engine. For
`FAN_IN_EDGE` this helper only aggregates messages **already present in a single batch**
(it delivers iff every source appears in `messages`, otherwise nothing); it does NOT
accumulate contributions across supersteps. Cross-superstep fan-in is handled separately
by the engine's stateful `_accumulate_fan_in!`, which the runner uses instead of this
function for `FAN_IN_EDGE` groups. Use this overload for fan-in only when all sources are
guaranteed to be in the same batch.
"""
function route_messages(group::EdgeGroup, messages::Vector{WorkflowMessage})::Dict{String, Vector{Any}}
    result = Dict{String, Vector{Any}}()

    if group.kind == DIRECT_EDGE
        edge = group.edges[1]
        for msg in messages
            if should_route(edge, msg.data)
                targets = get!(result, edge.target_id, Any[])
                push!(targets, msg.data)
            end
        end

    elseif group.kind == FAN_OUT_EDGE
        all_target_ids = target_executor_ids(group)
        for msg in messages
            targets = if group.selection_func !== nothing
                group.selection_func(msg.data, all_target_ids)
            else
                all_target_ids
            end
            for tid in targets
                target_msgs = get!(result, tid, Any[])
                push!(target_msgs, msg.data)
            end
        end

    elseif group.kind == FAN_IN_EDGE
        # Fan-in: collect from all sources. Target gets aggregated messages.
        target_id = group.edges[1].target_id
        sources_seen = Set{String}()
        aggregated = Any[]
        for msg in messages
            if msg.source_id in source_executor_ids(group)
                push!(sources_seen, msg.source_id)
                push!(aggregated, msg.data)
            end
        end
        # Only deliver if we have messages from ALL sources
        required_sources = Set(source_executor_ids(group))
        if sources_seen == required_sources && !isempty(aggregated)
            result[target_id] = aggregated
        end
    end

    return result
end
