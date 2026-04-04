# Checkpoint support for AgentFramework.jl workflows.
# Enables saving and restoring workflow state for resumable execution.

"""
    WorkflowCheckpoint

A snapshot of workflow state at a superstep boundary.

# Fields
- `id::String`: Unique checkpoint identifier.
- `workflow_name::String`: Name of the workflow.
- `iteration::Int`: Superstep number when checkpoint was taken.
- `state::Dict{String, Any}`: Committed workflow state.
- `messages::Dict{String, Vector{WorkflowMessage}}`: In-transit messages.
- `pending_requests::Vector{WorkflowEvent}`: Pending request_info events (HIL).
- `previous_id::Union{Nothing, String}`: Previous checkpoint in chain.
- `timestamp::String`: ISO 8601 timestamp.
- `metadata::Dict{String, Any}`: Custom metadata.
"""
Base.@kwdef struct WorkflowCheckpoint
    id::String = string(UUIDs.uuid4())
    workflow_name::String
    iteration::Int = 0
    state::Dict{String, Any} = Dict{String, Any}()
    messages::Dict{String, Vector{WorkflowMessage}} = Dict{String, Vector{WorkflowMessage}}()
    pending_requests::Vector{WorkflowEvent} = WorkflowEvent[]
    graph_signature_hash::String = ""
    previous_id::Union{Nothing, String} = nothing
    timestamp::String = Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat) * "Z"
    metadata::Dict{String, Any} = Dict{String, Any}()
end

function Base.show(io::IO, c::WorkflowCheckpoint)
    print(io, "WorkflowCheckpoint(\"", c.workflow_name, "\", iteration=", c.iteration, ", id=", c.id[1:min(8, length(c.id))], "...)")
end

# ── Abstract Storage Interface ───────────────────────────────────────────────

"""
    AbstractCheckpointStorage

Abstract type for checkpoint persistence backends.
Concrete implementations must implement: `save!`, `load`, `list_checkpoints`,
`get_latest`, `delete!`.
"""
abstract type AbstractCheckpointStorage end

"""
    save!(storage, checkpoint) -> String

Save a checkpoint and return its ID.
"""
function save! end

"""
    load(storage, checkpoint_id) -> WorkflowCheckpoint

Load a checkpoint by ID.
"""
function load end

"""
    list_checkpoints(storage, workflow_name) -> Vector{WorkflowCheckpoint}

List all checkpoints for a workflow.
"""
function list_checkpoints end

"""
    get_latest(storage, workflow_name) -> Union{Nothing, WorkflowCheckpoint}

Get the most recent checkpoint for a workflow.
"""
function get_latest end

# delete!(storage, checkpoint_id) -> Bool
# Delete a checkpoint. Returns true if deleted.
# Methods are added via Base.delete! on concrete storage types below.

# ── In-Memory Implementation ─────────────────────────────────────────────────

"""
    InMemoryCheckpointStorage <: AbstractCheckpointStorage

Simple in-memory checkpoint storage for development and testing.
Checkpoints are lost when the process exits.
"""
mutable struct InMemoryCheckpointStorage <: AbstractCheckpointStorage
    checkpoints::Dict{String, WorkflowCheckpoint}
    _lock::ReentrantLock
end

InMemoryCheckpointStorage() = InMemoryCheckpointStorage(
    Dict{String, WorkflowCheckpoint}(),
    ReentrantLock(),
)

function save!(storage::InMemoryCheckpointStorage, checkpoint::WorkflowCheckpoint)::String
    lock(storage._lock) do
        storage.checkpoints[checkpoint.id] = deepcopy(checkpoint)
    end
    return checkpoint.id
end

function load(storage::InMemoryCheckpointStorage, checkpoint_id::String)::WorkflowCheckpoint
    lock(storage._lock) do
        if !haskey(storage.checkpoints, checkpoint_id)
            throw(WorkflowError("Checkpoint not found: $checkpoint_id"))
        end
        return deepcopy(storage.checkpoints[checkpoint_id])
    end
end

function list_checkpoints(storage::InMemoryCheckpointStorage, workflow_name::String)::Vector{WorkflowCheckpoint}
    lock(storage._lock) do
        return [deepcopy(cp) for cp in values(storage.checkpoints) if cp.workflow_name == workflow_name]
    end
end

function get_latest(storage::InMemoryCheckpointStorage, workflow_name::String)::Union{Nothing, WorkflowCheckpoint}
    lock(storage._lock) do
        matching = [cp for cp in values(storage.checkpoints) if cp.workflow_name == workflow_name]
        isempty(matching) && return nothing
        return deepcopy(sort(matching; by=cp -> cp.timestamp, rev=true)[1])
    end
end

function Base.delete!(storage::InMemoryCheckpointStorage, checkpoint_id::String)::Bool
    lock(storage._lock) do
        if haskey(storage.checkpoints, checkpoint_id)
            Base.delete!(storage.checkpoints, checkpoint_id)
            return true
        end
        return false
    end
end

# ── File-Based Implementation ────────────────────────────────────────────────

"""
    FileCheckpointStorage <: AbstractCheckpointStorage

File-based checkpoint storage using JSON serialization.
Each checkpoint is saved as a separate JSON file.

# Fields
- `directory::String`: Directory to store checkpoint files.
"""
struct FileCheckpointStorage <: AbstractCheckpointStorage
    directory::String

    function FileCheckpointStorage(directory::String)
        mkpath(directory)
        new(directory)
    end
end

function _checkpoint_path(storage::FileCheckpointStorage, id::String)
    joinpath(storage.directory, "$id.json")
end

function _serialize_checkpoint(cp::WorkflowCheckpoint)::Dict{String, Any}
    function serialize_workflow_message(msg::WorkflowMessage)
        Dict{String, Any}(
            "data" => _serialize_any_value(msg.data),
            "source_id" => msg.source_id,
            "target_id" => msg.target_id,
            "type" => string(msg.type),
        )
    end

    function serialize_error_details(details::WorkflowErrorDetails)
        Dict{String, Any}(
            "error_type" => details.error_type,
            "message" => details.message,
            "executor_id" => details.executor_id,
            "stacktrace" => details.stacktrace,
        )
    end

    function serialize_workflow_event(event::WorkflowEvent)
        data = Dict{String, Any}(
            "type" => string(event.type),
            "timestamp" => event.timestamp,
        )
        event.data !== nothing && (data["data"] = _serialize_any_value(event.data))
        event.executor_id !== nothing && (data["executor_id"] = event.executor_id)
        event.state !== nothing && (data["state"] = string(event.state))
        event.details !== nothing && (data["details"] = serialize_error_details(event.details))
        event.iteration !== nothing && (data["iteration"] = event.iteration)
        event.request_id !== nothing && (data["request_id"] = event.request_id)
        return data
    end

    Dict{String, Any}(
        "id" => cp.id,
        "workflow_name" => cp.workflow_name,
        "iteration" => cp.iteration,
        "state" => _serialize_any_value(cp.state),
        "messages" => Dict(k => [serialize_workflow_message(m) for m in v] for (k, v) in cp.messages),
        "pending_requests" => [serialize_workflow_event(evt) for evt in cp.pending_requests],
        "graph_signature_hash" => cp.graph_signature_hash,
        "previous_id" => cp.previous_id,
        "timestamp" => cp.timestamp,
        "metadata" => _serialize_any_value(cp.metadata),
    )
end

function _deserialize_checkpoint(data::Dict{String, Any})::WorkflowCheckpoint
    function parse_workflow_message_type(raw)
        raw_str = string(raw)
        raw_str == string(RESPONSE_MESSAGE) && return RESPONSE_MESSAGE
        return STANDARD_MESSAGE
    end

    function parse_workflow_event_type(raw)
        raw_str = string(raw)
        for candidate in instances(WorkflowEventType)
            if string(candidate) == raw_str
                return candidate
            end
        end
        throw(WorkflowError("Unknown workflow event type in checkpoint: $raw_str"))
    end

    function parse_workflow_run_state(raw)
        raw === nothing && return nothing
        raw_str = string(raw)
        for candidate in instances(WorkflowRunState)
            if string(candidate) == raw_str
                return candidate
            end
        end
        throw(WorkflowError("Unknown workflow run state in checkpoint: $raw_str"))
    end

    function deserialize_error_details(raw)
        raw === nothing && return nothing
        dict = raw isa Dict ? raw : Dict{String, Any}(string(k) => v for (k, v) in pairs(raw))
        WorkflowErrorDetails(
            error_type = string(get(dict, "error_type", "")),
            message = string(get(dict, "message", "")),
            executor_id = let v = get(dict, "executor_id", nothing); v === nothing ? nothing : string(v) end,
            stacktrace = let v = get(dict, "stacktrace", nothing); v === nothing ? nothing : string(v) end,
        )
    end

    function deserialize_workflow_event(raw)
        dict = raw isa Dict ? raw : Dict{String, Any}(string(k) => v for (k, v) in pairs(raw))
        WorkflowEvent{Any}(
            type = parse_workflow_event_type(get(dict, "type", EVT_REQUEST_INFO)),
            data = _deserialize_any_value(get(dict, "data", nothing)),
            executor_id = let v = get(dict, "executor_id", nothing); v === nothing ? nothing : string(v) end,
            state = parse_workflow_run_state(get(dict, "state", nothing)),
            details = deserialize_error_details(get(dict, "details", nothing)),
            iteration = haskey(dict, "iteration") ? Int(dict["iteration"]) : nothing,
            request_id = let v = get(dict, "request_id", nothing); v === nothing ? nothing : string(v) end,
            timestamp = Float64(get(dict, "timestamp", time())),
        )
    end

    messages = Dict{String, Vector{WorkflowMessage}}()
    if haskey(data, "messages") && data["messages"] !== nothing
        msg_data = data["messages"]
        msg_dict = msg_data isa Dict ? msg_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(msg_data))
        for (k, v) in msg_dict
            msgs = WorkflowMessage[]
            v_vec = v isa AbstractVector ? v : [v]
            for m in v_vec
                m_dict = m isa Dict ? m : Dict{String, Any}(string(mk) => mv for (mk, mv) in pairs(m))
                push!(msgs, WorkflowMessage(
                    data = _deserialize_any_value(get(m_dict, "data", nothing)),
                    source_id = string(get(m_dict, "source_id", "")),
                    target_id = let t = get(m_dict, "target_id", nothing); t === nothing ? nothing : string(t) end,
                    type = parse_workflow_message_type(get(m_dict, "type", STANDARD_MESSAGE)),
                ))
            end
            messages[string(k)] = msgs
        end
    end

    state = _deserialize_any_value(get(data, "state", Dict{String, Any}()))

    pending_requests = WorkflowEvent[]
    if haskey(data, "pending_requests") && data["pending_requests"] !== nothing
        for raw in data["pending_requests"]
            push!(pending_requests, deserialize_workflow_event(raw))
        end
    end

    metadata = _deserialize_any_value(get(data, "metadata", Dict{String, Any}()))

    WorkflowCheckpoint(
        id = string(get(data, "id", "")),
        workflow_name = string(get(data, "workflow_name", "")),
        iteration = Int(get(data, "iteration", 0)),
        state = state,
        messages = messages,
        pending_requests = pending_requests,
        graph_signature_hash = string(get(data, "graph_signature_hash", "")),
        previous_id = let p = get(data, "previous_id", nothing); p === nothing ? nothing : string(p) end,
        timestamp = string(get(data, "timestamp", "")),
        metadata = metadata,
    )
end

function save!(storage::FileCheckpointStorage, checkpoint::WorkflowCheckpoint)::String
    path = _checkpoint_path(storage, checkpoint.id)
    data = _serialize_checkpoint(checkpoint)
    open(path, "w") do io
        JSON3.pretty(io, data)
    end
    return checkpoint.id
end

function load(storage::FileCheckpointStorage, checkpoint_id::String)::WorkflowCheckpoint
    path = _checkpoint_path(storage, checkpoint_id)
    if !isfile(path)
        throw(WorkflowError("Checkpoint file not found: $path"))
    end
    data = JSON3.read(read(path, String), Dict{String, Any})
    return _deserialize_checkpoint(data)
end

function list_checkpoints(storage::FileCheckpointStorage, workflow_name::String)::Vector{WorkflowCheckpoint}
    checkpoints = WorkflowCheckpoint[]
    for f in readdir(storage.directory)
        endswith(f, ".json") || continue
        try
            path = joinpath(storage.directory, f)
            data = JSON3.read(read(path, String), Dict{String, Any})
            cp = _deserialize_checkpoint(data)
            if cp.workflow_name == workflow_name
                push!(checkpoints, cp)
            end
        catch
            # Skip corrupted files
        end
    end
    return sort(checkpoints; by=cp -> cp.timestamp)
end

function get_latest(storage::FileCheckpointStorage, workflow_name::String)::Union{Nothing, WorkflowCheckpoint}
    cps = list_checkpoints(storage, workflow_name)
    isempty(cps) && return nothing
    return last(cps)
end

function Base.delete!(storage::FileCheckpointStorage, checkpoint_id::String)::Bool
    path = _checkpoint_path(storage, checkpoint_id)
    if isfile(path)
        rm(path)
        return true
    end
    return false
end
