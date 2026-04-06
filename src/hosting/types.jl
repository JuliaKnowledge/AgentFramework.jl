abstract type AbstractHostedRunStore end

function load_run end
function save_run! end
function delete_run! end
function list_runs end

_utcnow() = Dates.now(Dates.UTC)
_timestamp_format() = dateformat"yyyy-mm-ddTHH:MM:SS.sss"
_format_timestamp(dt::DateTime) = Dates.format(dt, _timestamp_format()) * "Z"
_parse_timestamp(value) = DateTime(replace(String(value), "Z" => ""), _timestamp_format())

function _materialize_json(value)
    if value isa AbstractDict
        return Dict{String, Any}(String(k) => _materialize_json(v) for (k, v) in pairs(value))
    elseif value isa AbstractVector
        return Any[_materialize_json(v) for v in value]
    end
    return value
end

function _jsonish(value)
    if value === nothing || value isa AbstractString || value isa Number || value isa Bool
        return value
    elseif value isa AbstractDict
        return Dict{String, Any}(String(k) => _jsonish(v) for (k, v) in pairs(value))
    elseif value isa Tuple || value isa AbstractVector
        return Any[_jsonish(v) for v in value]
    elseif value isa Symbol
        return String(value)
    elseif value isa DateTime
        return _format_timestamp(value)
    elseif applicable(AgentFramework.serialize_to_dict, value)
        return _jsonish(AgentFramework.serialize_to_dict(value))
    end
    return string(value)
end

function _parse_workflow_state(raw)::AgentFramework.WorkflowRunState
    expected = String(raw)
    for candidate in instances(AgentFramework.WorkflowRunState)
        string(candidate) == expected && return candidate
    end
    throw(ArgumentError("Unknown workflow state: $(raw)"))
end

Base.@kwdef mutable struct HostedWorkflowRun
    id::String = string(uuid4())
    workflow_name::String
    internal_workflow_name::String
    state::AgentFramework.WorkflowRunState = AgentFramework.WF_STARTED
    checkpoint_id::Union{Nothing, String} = nothing
    outputs::Vector{Any} = Any[]
    pending_requests::Vector{Dict{String, Any}} = Dict{String, Any}[]
    events::Vector{Dict{String, Any}} = Dict{String, Any}[]
    error::Union{Nothing, String} = nothing
    created_at::DateTime = _utcnow()
    updated_at::DateTime = _utcnow()
    metadata::Dict{String, Any} = Dict{String, Any}()
end

function hosted_workflow_run_to_dict(run::HostedWorkflowRun)
    return Dict{String, Any}(
        "id" => run.id,
        "workflow_name" => run.workflow_name,
        "internal_workflow_name" => run.internal_workflow_name,
        "state" => string(run.state),
        "checkpoint_id" => run.checkpoint_id,
        "outputs" => deepcopy(run.outputs),
        "pending_requests" => deepcopy(run.pending_requests),
        "events" => deepcopy(run.events),
        "error" => run.error,
        "created_at" => _format_timestamp(run.created_at),
        "updated_at" => _format_timestamp(run.updated_at),
        "metadata" => deepcopy(run.metadata),
    )
end

function _workflow_run_from_dict(data::AbstractDict)
    return HostedWorkflowRun(
        id = String(data["id"]),
        workflow_name = String(data["workflow_name"]),
        internal_workflow_name = String(data["internal_workflow_name"]),
        state = _parse_workflow_state(data["state"]),
        checkpoint_id = get(data, "checkpoint_id", nothing),
        outputs = Any[_materialize_json(v) for v in get(data, "outputs", Any[])],
        pending_requests = Dict{String, Any}[_materialize_json(v) for v in get(data, "pending_requests", Any[])],
        events = Dict{String, Any}[_materialize_json(v) for v in get(data, "events", Any[])],
        error = get(data, "error", nothing),
        created_at = _parse_timestamp(data["created_at"]),
        updated_at = _parse_timestamp(data["updated_at"]),
        metadata = Dict{String, Any}(_materialize_json(get(data, "metadata", Dict{String, Any}()))),
    )
end

Base.@kwdef mutable struct InMemoryHostedRunStore <: AbstractHostedRunStore
    runs::Dict{String, HostedWorkflowRun} = Dict{String, HostedWorkflowRun}()
    lock::ReentrantLock = ReentrantLock()
end

function load_run(store::InMemoryHostedRunStore, run_id::AbstractString)
    lock(store.lock) do
        run = get(store.runs, String(run_id), nothing)
        return run === nothing ? nothing : deepcopy(run)
    end
end

function save_run!(store::InMemoryHostedRunStore, run::HostedWorkflowRun)
    lock(store.lock) do
        store.runs[run.id] = deepcopy(run)
    end
    return run.id
end

function delete_run!(store::InMemoryHostedRunStore, run_id::AbstractString)
    lock(store.lock) do
        return pop!(store.runs, String(run_id), nothing) !== nothing
    end
end

function list_runs(store::InMemoryHostedRunStore, workflow_name::Union{Nothing, AbstractString}=nothing)
    lock(store.lock) do
        runs = HostedWorkflowRun[
            deepcopy(run) for run in values(store.runs)
            if workflow_name === nothing || run.workflow_name == workflow_name
        ]
        sort!(runs, by = run -> (run.created_at, run.id))
        return runs
    end
end

Base.@kwdef mutable struct FileHostedRunStore <: AbstractHostedRunStore
    directory::String
    lock::ReentrantLock = ReentrantLock()
end

function FileHostedRunStore(directory::AbstractString)
    root = abspath(String(directory))
    mkpath(root)
    return FileHostedRunStore(directory = root)
end

_run_path(store::FileHostedRunStore, run_id::AbstractString) = joinpath(store.directory, String(run_id) * ".json")

function load_run(store::FileHostedRunStore, run_id::AbstractString)
    path = _run_path(store, run_id)
    isfile(path) || return nothing
    lock(store.lock) do
        data = JSON3.read(read(path, String), Dict{String, Any})
        return _workflow_run_from_dict(_materialize_json(data))
    end
end

function save_run!(store::FileHostedRunStore, run::HostedWorkflowRun)
    path = _run_path(store, run.id)
    payload = JSON3.write(hosted_workflow_run_to_dict(run))
    lock(store.lock) do
        mkpath(store.directory)
        write(path, payload)
    end
    return run.id
end

function delete_run!(store::FileHostedRunStore, run_id::AbstractString)
    path = _run_path(store, run_id)
    lock(store.lock) do
        isfile(path) || return false
        rm(path)
        return true
    end
end

function list_runs(store::FileHostedRunStore, workflow_name::Union{Nothing, AbstractString}=nothing)
    lock(store.lock) do
        runs = HostedWorkflowRun[]
        for entry in sort(readdir(store.directory))
            endswith(entry, ".json") || continue
            run = load_run(store, first(splitext(entry)))
            run === nothing && continue
            workflow_name === nothing || run.workflow_name == workflow_name || continue
            push!(runs, run)
        end
        sort!(runs, by = run -> (run.created_at, run.id))
        return runs
    end
end
