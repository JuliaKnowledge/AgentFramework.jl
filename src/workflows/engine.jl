# Workflow engine for AgentFramework.jl
# Implements the Pregel-like superstep execution model.

"""
    Workflow

A DAG-based workflow that orchestrates message passing between executors.
Uses synchronous superstep execution: all ready executors run concurrently
within a superstep, messages are delivered at superstep boundaries.

# Fields
- `name::String`: Workflow name.
- `executors::Dict{String, ExecutorSpec}`: Executor registry (id → spec).
- `edge_groups::Vector{EdgeGroup}`: All routing groups.
- `start_executor_id::String`: Entry point executor.
- `output_executor_ids::Vector{String}`: Executors whose yield_output calls become workflow outputs.
- `max_iterations::Int`: Maximum supersteps before stopping.
- `state::Dict{String, Any}`: Shared workflow state.
- `_running::Bool`: Whether the workflow is currently executing (prevents concurrent runs).
- `_lock::ReentrantLock`: Lock guarding the `_running` flag.

# Example
```julia
workflow = WorkflowBuilder(
    name = "TextPipeline",
    start = upper_executor,
)  |> add_edge("upper", "reverse") |> add_output("reverse") |> build

result = run_workflow(workflow, "hello world")
outputs = get_outputs(result)  # ["DLROW OLLEH"]
```
"""
Base.@kwdef mutable struct Workflow
    name::String = "Workflow"
    executors::Dict{String, ExecutorSpec} = Dict{String, ExecutorSpec}()
    edge_groups::Vector{EdgeGroup} = EdgeGroup[]
    start_executor_id::String = ""
    output_executor_ids::Vector{String} = String[]
    max_iterations::Int = 100
    state::Dict{String, Any} = Dict{String, Any}()
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing
    graph_signature_hash::String = ""
    _running::Bool = false
    _lock::ReentrantLock = ReentrantLock()
end

function Base.show(io::IO, w::Workflow)
    print(io, "Workflow(\"", w.name, "\", ", length(w.executors), " executors, ", length(w.edge_groups), " edges)")
end

# ── Workflow Execution ───────────────────────────────────────────────────────

function _workflow_graph_signature(workflow::Workflow)::String
    executor_parts = sort([
        join((
            spec.id,
            string.(spec.input_types)...,
            "->",
            string.(spec.output_types)...,
            "=>",
            string.(spec.yield_types)...,
        ), ":")
        for spec in values(workflow.executors)
    ])

    edge_parts = sort([
        join((
            string(group.kind),
            edge.source_id,
            edge.target_id,
            something(edge.condition_name, edge.condition === nothing ? "direct" : "conditional"),
            group.selection_func === nothing ? "all" : "selected",
        ), ":")
        for group in workflow.edge_groups for edge in group.edges
    ])

    return join(vcat(
        ["name=$(workflow.name)", "start=$(workflow.start_executor_id)", "max=$(workflow.max_iterations)"],
        ["outputs=" * join(sort(workflow.output_executor_ids), ",")],
        executor_parts,
        edge_parts,
    ), "|")
end

function _workflow_graph_signature_hash(workflow::Workflow)::String
    bytes2hex(SHA.sha1(_workflow_graph_signature(workflow)))
end

function _copy_message_queue(queue::Dict{String, Vector{WorkflowMessage}})
    deepcopy(queue)
end

function _pending_request_dict!(state::Dict{String, Any})::Dict{String, WorkflowEvent}
    pending = get!(state, "_pending_requests", Dict{String, WorkflowEvent}())
    if pending isa Dict{String, WorkflowEvent}
        return pending
    elseif pending isa AbstractDict
        converted = Dict{String, WorkflowEvent}()
        for (req_id, raw) in pairs(pending)
            key = string(req_id)
            if raw isa WorkflowEvent
                converted[key] = raw
            elseif raw isa AbstractDict
                dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(raw))
                converted[key] = event_request_info(
                    key,
                    string(get(dict, "executor_id", "")),
                    _deserialize_any_value(get(dict, "data", nothing)),
                )
            else
                converted[key] = event_request_info(key, "", raw)
            end
        end
        state["_pending_requests"] = converted
        return converted
    else
        throw(WorkflowError("Workflow pending request state must be a dictionary"))
    end
end

function _resolve_checkpoint_storage(
    workflow::Workflow,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage},
)::Union{Nothing, AbstractCheckpointStorage}
    checkpoint_storage === nothing ? workflow.checkpoint_storage : checkpoint_storage
end

function _restore_checkpoint!(
    workflow::Workflow,
    storage::AbstractCheckpointStorage,
    checkpoint_id::String,
)
    checkpoint = try
        load(storage, checkpoint_id)
    catch e
        if e isa WorkflowError
            throw(WorkflowCheckpointError("Checkpoint '$checkpoint_id' could not be loaded: $(e.message)"))
        end
        rethrow()
    end
    if checkpoint.workflow_name != workflow.name
        throw(WorkflowCheckpointError("Checkpoint '$checkpoint_id' belongs to workflow '$(checkpoint.workflow_name)', not '$(workflow.name)'"))
    end

    expected_hash = isempty(workflow.graph_signature_hash) ? _workflow_graph_signature_hash(workflow) : workflow.graph_signature_hash
    if !isempty(checkpoint.graph_signature_hash) && checkpoint.graph_signature_hash != expected_hash
        throw(WorkflowCheckpointError("Checkpoint '$checkpoint_id' is not compatible with the current workflow graph"))
    end

    workflow.state = deepcopy(checkpoint.state)
    pending = Dict{String, WorkflowEvent}(evt.request_id => deepcopy(evt) for evt in checkpoint.pending_requests if evt.request_id !== nothing)
    if isempty(pending)
        delete!(workflow.state, "_pending_requests")
    else
        workflow.state["_pending_requests"] = pending
    end

    return checkpoint, _copy_message_queue(checkpoint.messages)
end

function _save_checkpoint!(
    storage::Union{Nothing, AbstractCheckpointStorage},
    workflow::Workflow,
    iteration::Int,
    message_queue::Dict{String, Vector{WorkflowMessage}},
    pending_requests::Dict{String, WorkflowEvent},
    previous_id::Union{Nothing, String},
)
    storage === nothing && return previous_id
    checkpoint = WorkflowCheckpoint(
        workflow_name = workflow.name,
        iteration = iteration,
        state = deepcopy(workflow.state),
        messages = _copy_message_queue(message_queue),
        pending_requests = [deepcopy(evt) for evt in values(pending_requests)],
        graph_signature_hash = isempty(workflow.graph_signature_hash) ? _workflow_graph_signature_hash(workflow) : workflow.graph_signature_hash,
        previous_id = previous_id,
    )
    return save!(storage, checkpoint)
end

"""
    run_workflow(workflow::Workflow, message=nothing; responses=nothing, stream=false, checkpoint_id=nothing, checkpoint_storage=nothing) -> WorkflowRunResult

Execute a workflow with the given initial message.

# Arguments
- `workflow::Workflow`: The workflow to execute.
- `message`: Initial message for the start executor.
- `responses::Union{Nothing, Dict{String, Any}}`: Responses for pending request_info events (HIL continuation).
- `stream::Bool`: If true, returns a Channel{WorkflowEvent} for streaming events.
- `checkpoint_id::Union{Nothing, String}`: Resume from a previously saved checkpoint.
- `checkpoint_storage`: Storage backend to use for checkpoint loading/saving.
"""
function run_workflow(
    workflow::Workflow,
    message = nothing;
    responses::Union{Nothing, Dict{String, Any}} = nothing,
    stream::Bool = false,
    checkpoint_id::Union{Nothing, String} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
)
    if stream
        return _run_workflow_streaming(workflow, message; responses=responses, checkpoint_id=checkpoint_id, checkpoint_storage=checkpoint_storage)
    else
        return _run_workflow_sync(workflow, message; responses=responses, checkpoint_id=checkpoint_id, checkpoint_storage=checkpoint_storage)
    end
end

function _run_workflow_sync(
    workflow::Workflow,
    message;
    responses::Union{Nothing, Dict{String, Any}} = nothing,
    checkpoint_id::Union{Nothing, String} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
)::WorkflowRunResult
    # Ownership check — prevent concurrent runs on the same workflow
    lock(workflow._lock) do
        if workflow._running
            throw(WorkflowError("Workflow '$(workflow.name)' is already running"))
        end
        workflow._running = true
    end

    try
        return _execute_workflow(workflow, message; responses=responses, checkpoint_id=checkpoint_id, checkpoint_storage=checkpoint_storage)
    finally
        lock(workflow._lock) do
            workflow._running = false
        end
    end
end

function _execute_workflow(
    workflow::Workflow,
    message;
    responses::Union{Nothing, Dict{String, Any}} = nothing,
    checkpoint_id::Union{Nothing, String} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
)::WorkflowRunResult
    events = WorkflowEvent[]
    storage = _resolve_checkpoint_storage(workflow, checkpoint_storage)

    push!(events, event_started())
    push!(events, event_status(WF_IN_PROGRESS))

    if message === nothing && responses === nothing && checkpoint_id === nothing
        throw(WorkflowError("run_workflow requires an initial message, responses, or a checkpoint_id"))
    end

    try
        message_queue = Dict{String, Vector{WorkflowMessage}}()
        pending_requests = Dict{String, WorkflowEvent}()
        previous_checkpoint_id = checkpoint_id
        iteration = 0

        if checkpoint_id !== nothing
            storage === nothing && throw(WorkflowCheckpointError("checkpoint_id requires checkpoint_storage"))
            checkpoint, message_queue = _restore_checkpoint!(workflow, storage, checkpoint_id)
            pending_requests = _pending_request_dict!(workflow.state)
            iteration = checkpoint.iteration
            previous_checkpoint_id = checkpoint.id
        else
            delete!(workflow.state, "_fan_in_buffers")
            if responses === nothing
                delete!(workflow.state, "_pending_requests")
            end
            pending_requests = _pending_request_dict!(workflow.state)
            isempty(pending_requests) && delete!(workflow.state, "_pending_requests")
        end

        if message !== nothing
            start_msgs = get!(message_queue, workflow.start_executor_id, WorkflowMessage[])
            push!(start_msgs, WorkflowMessage(data=message, source_id="__input__"))
        end

        if responses !== nothing
            pending = _pending_request_dict!(workflow.state)
            for (req_id, response_data) in responses
                if haskey(pending, req_id)
                    info = pending[req_id]
                    executor_id = something(info.executor_id, "")
                    msgs = get!(message_queue, executor_id, WorkflowMessage[])
                    push!(msgs, WorkflowMessage(
                        data=response_data,
                        source_id="__hil_response__",
                        type=RESPONSE_MESSAGE,
                    ))
                    delete!(pending, req_id)
                else
                    @warn "No pending request found for response id=$req_id"
                end
            end
            isempty(pending) && delete!(workflow.state, "_pending_requests")
        end

        while iteration < workflow.max_iterations && !isempty(message_queue)
            iteration += 1
            push!(events, event_superstep_started(iteration))

            next_queue = Dict{String, Vector{WorkflowMessage}}()

            # Run all ready executors concurrently within the superstep
            results_lock = ReentrantLock()
            executor_results = Vector{Any}()
            failed_ref = Ref{Union{Nothing, Tuple{String, WorkflowErrorDetails}}}(nothing)

            @sync for (executor_id, messages) in message_queue
                spec = get(workflow.executors, executor_id, nothing)
                if spec === nothing
                    @warn "No executor found for id=$executor_id, skipping"
                    continue
                end

                @async begin
                    local_events = WorkflowEvent[event_executor_invoked(executor_id)]
                    local_requests = WorkflowEvent[]
                    local_sent = Tuple{String, Vector{WorkflowMessage}}[]

                    try
                        for msg in messages
                            ctx = execute_handler(spec, msg.data, [msg.source_id], workflow.state)
                            append!(local_events, ctx._events)
                            append!(local_requests, ctx._request_infos)
                            if !isempty(ctx._sent_messages)
                                push!(local_sent, (executor_id, copy(ctx._sent_messages)))
                            end
                        end
                        push!(local_events, event_executor_completed(executor_id))
                    catch e
                        details = WorkflowErrorDetails(
                            error_type = string(typeof(e)),
                            message = sprint(showerror, e),
                            executor_id = executor_id,
                        )
                        push!(local_events, event_executor_failed(executor_id, details))
                        lock(results_lock) do
                            if failed_ref[] === nothing
                                failed_ref[] = (executor_id, details)
                            end
                        end
                    end

                    lock(results_lock) do
                        push!(executor_results, (local_events, local_requests, local_sent))
                    end
                end
            end

            # Merge results and route messages sequentially (safe for fan-in buffers)
            for (local_events, local_requests, local_sent) in executor_results
                append!(events, local_events)
                for req in local_requests
                    req.request_id === nothing && continue
                    pending_requests[req.request_id] = req
                end
                for (src_id, sent_msgs) in local_sent
                    _route_messages!(next_queue, workflow, src_id, sent_msgs)
                end
            end

            # Check for failures after the superstep
            if failed_ref[] !== nothing
                _, details = failed_ref[]
                push!(events, event_failed(details))
                push!(events, event_status(WF_FAILED))
                return WorkflowRunResult(events=events, state=WF_FAILED)
            end

            workflow.state["_pending_requests"] = pending_requests
            previous_checkpoint_id = _save_checkpoint!(storage, workflow, iteration, next_queue, pending_requests, previous_checkpoint_id)
            push!(events, event_superstep_completed(iteration))
            message_queue = next_queue
        end

        if isempty(pending_requests)
            delete!(workflow.state, "_pending_requests")
        else
            workflow.state["_pending_requests"] = pending_requests
        end

        # Determine final state
        final_state = if !isempty(pending_requests)
            WF_IDLE_WITH_PENDING_REQUESTS
        else
            WF_IDLE
        end

        if iteration >= workflow.max_iterations && !isempty(message_queue)
            @warn "Workflow reached max iterations ($iteration)"
        end

        push!(events, event_status(final_state))
        return WorkflowRunResult(events=events, state=final_state)

    catch e
        if e isa WorkflowCheckpointError
            rethrow()
        end
        details = WorkflowErrorDetails(
            error_type = string(typeof(e)),
            message = sprint(showerror, e),
        )
        push!(events, event_failed(details))
        push!(events, event_status(WF_FAILED))
        return WorkflowRunResult(events=events, state=WF_FAILED)
    end
end

function _run_workflow_streaming(
    workflow::Workflow,
    message;
    responses::Union{Nothing, Dict{String, Any}} = nothing,
    checkpoint_id::Union{Nothing, String} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
)::Channel{WorkflowEvent}
    channel = Channel{WorkflowEvent}(64)

    Threads.@spawn begin
        try
            result = _run_workflow_sync(workflow, message; responses=responses, checkpoint_id=checkpoint_id, checkpoint_storage=checkpoint_storage)
            for evt in result.events
                put!(channel, evt)
            end
        catch e
            if !(e isa InvalidStateException)
                @error "Workflow streaming error" exception=(e, catch_backtrace())
            end
        finally
            close(channel)
        end
    end

    return channel
end

# ── Internal Routing ─────────────────────────────────────────────────────────

"""Route messages from an executor through the workflow's edge groups."""
function _route_messages!(
    queue::Dict{String, Vector{WorkflowMessage}},
    workflow::Workflow,
    source_executor_id::String,
    messages::Vector{WorkflowMessage},
)
    # Separate targeted and untargeted messages
    targeted = WorkflowMessage[]
    untargeted = WorkflowMessage[]
    for msg in messages
        if msg.target_id !== nothing
            push!(targeted, msg)
        else
            push!(untargeted, msg)
        end
    end

    # Route targeted messages directly (exactly once)
    for msg in targeted
        target_msgs = get!(queue, msg.target_id, WorkflowMessage[])
        push!(target_msgs, msg)
    end

    # Route untargeted messages through matching edge groups
    if !isempty(untargeted)
        for group in workflow.edge_groups
            if source_executor_id in source_executor_ids(group)
                if group.kind == FAN_IN_EDGE
                    _accumulate_fan_in!(queue, workflow, group, source_executor_id, untargeted)
                else
                    routed = route_messages(group, untargeted)
                    for (target_id, payloads) in routed
                        target_msgs = get!(queue, target_id, WorkflowMessage[])
                        for data in payloads
                            push!(target_msgs, WorkflowMessage(data=data, source_id=source_executor_id))
                        end
                    end
                end
            end
        end
    end
end

"""Accumulate fan-in messages across supersteps, delivering when all sources have contributed."""
function _accumulate_fan_in!(
    queue::Dict{String, Vector{WorkflowMessage}},
    workflow::Workflow,
    group::EdgeGroup,
    source_executor_id::String,
    messages::Vector{WorkflowMessage},
)
    buffers = get!(workflow.state, "_fan_in_buffers", Dict{String, Dict{String, Vector{Any}}}())
    group_buf = get!(buffers, group.id, Dict{String, Vector{Any}}())
    source_buf = get!(group_buf, source_executor_id, Any[])

    for msg in messages
        push!(source_buf, msg.data)
    end

    # Check if all required sources have contributed
    required = Set(source_executor_ids(group))
    contributed = Set(k for (k, v) in group_buf if !isempty(v))

    if required ⊆ contributed
        target_id = group.edges[1].target_id
        aggregated = Any[]
        for src_id in source_executor_ids(group)
            if haskey(group_buf, src_id)
                append!(aggregated, group_buf[src_id])
            end
        end

        target_msgs = get!(queue, target_id, WorkflowMessage[])
        for data in aggregated
            push!(target_msgs, WorkflowMessage(data=data, source_id="__fan_in__"))
        end

        # Clear buffers after delivery
        for (_, v) in group_buf
            empty!(v)
        end
    end
end
