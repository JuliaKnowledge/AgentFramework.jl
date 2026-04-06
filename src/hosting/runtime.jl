Base.@kwdef mutable struct HostedRuntime
    agents::Dict{String, AgentFramework.Agent} = Dict{String, AgentFramework.Agent}()
    workflows::Dict{String, AgentFramework.Workflow} = Dict{String, AgentFramework.Workflow}()
    session_store::AgentFramework.AbstractSessionStore = AgentFramework.InMemorySessionStore()
    history_provider::AgentFramework.AbstractHistoryProvider = AgentFramework.InMemoryHistoryProvider(source_id = "hosting_history")
    checkpoint_storage::AgentFramework.AbstractCheckpointStorage = AgentFramework.InMemoryCheckpointStorage()
    run_store::AbstractHostedRunStore = InMemoryHostedRunStore()
    lock::ReentrantLock = ReentrantLock()
end

function HostedRuntime(directory::AbstractString)
    root = abspath(String(directory))
    mkpath(root)
    return HostedRuntime(
        session_store = AgentFramework.FileSessionStore(joinpath(root, "sessions")),
        history_provider = AgentFramework.FileHistoryProvider(directory = joinpath(root, "history")),
        checkpoint_storage = AgentFramework.FileCheckpointStorage(joinpath(root, "checkpoints")),
        run_store = FileHostedRunStore(joinpath(root, "runs")),
    )
end

function _require_name(name::AbstractString, kind::AbstractString)
    isempty(strip(String(name))) && throw(ArgumentError("$(kind) name cannot be empty"))
    return String(name)
end

function _get_registered_agent(runtime::HostedRuntime, name::AbstractString)
    agent = get(runtime.agents, String(name), nothing)
    agent === nothing && throw(KeyError("Unknown hosted agent: $(name)"))
    return agent
end

function _get_registered_workflow(runtime::HostedRuntime, name::AbstractString)
    workflow = get(runtime.workflows, String(name), nothing)
    workflow === nothing && throw(KeyError("Unknown hosted workflow: $(name)"))
    return workflow
end

function _agent_history_provider(agent::AgentFramework.Agent)
    for provider in agent.context_providers
        provider isa AgentFramework.AbstractHistoryProvider && return provider
    end
    return nothing
end

function _ensure_history_provider!(runtime::HostedRuntime, agent::AgentFramework.Agent)
    _agent_history_provider(agent) === nothing && push!(agent.context_providers, runtime.history_provider)
    return agent
end

function list_registered_agents(runtime::HostedRuntime)
    return sort!(collect(keys(runtime.agents)))
end

function list_registered_workflows(runtime::HostedRuntime)
    return sort!(collect(keys(runtime.workflows)))
end

function register_agent!(runtime::HostedRuntime, agent::AgentFramework.Agent; name::AbstractString = agent.name)
    agent_name = _require_name(name, "Agent")
    lock(runtime.lock) do
        haskey(runtime.agents, agent_name) && throw(ArgumentError("Hosted agent already registered: $(agent_name)"))
        _ensure_history_provider!(runtime, agent)
        runtime.agents[agent_name] = agent
    end
    return agent
end

function register_workflow!(runtime::HostedRuntime, workflow::AgentFramework.Workflow; name::AbstractString = workflow.name)
    workflow_name = _require_name(name, "Workflow")
    lock(runtime.lock) do
        haskey(runtime.workflows, workflow_name) && throw(ArgumentError("Hosted workflow already registered: $(workflow_name)"))
        runtime.workflows[workflow_name] = workflow
    end
    return workflow
end

function _session_agent_name(session::AgentFramework.AgentSession)
    value = get(session.metadata, "agent_name", nothing)
    return value === nothing ? nothing : String(value)
end

function _load_or_create_session!(runtime::HostedRuntime, agent_name::AbstractString, session_id::Union{Nothing, AbstractString})
    if session_id !== nothing
        session = AgentFramework.load_session(runtime.session_store, String(session_id))
        if session === nothing
            session = AgentFramework.AgentSession(id = String(session_id), metadata = Dict{String, Any}("agent_name" => String(agent_name)))
            AgentFramework.save_session!(runtime.session_store, session)
            return session
        end
        existing_agent = _session_agent_name(session)
        existing_agent === nothing && (session.metadata["agent_name"] = String(agent_name))
        _session_agent_name(session) == agent_name || throw(ArgumentError("Session $(session.id) belongs to hosted agent $(existing_agent), not $(agent_name)"))
        return session
    end
    session = AgentFramework.AgentSession(metadata = Dict{String, Any}("agent_name" => String(agent_name)))
    AgentFramework.save_session!(runtime.session_store, session)
    return session
end

function _load_history(agent::AgentFramework.Agent, session_id::AbstractString)
    provider = _agent_history_provider(agent)
    provider === nothing && return AgentFramework.Message[]
    return AgentFramework.get_messages(provider, String(session_id))
end

function run_agent!(runtime::HostedRuntime, agent_name::AbstractString, inputs = nothing; session_id::Union{Nothing, AbstractString}=nothing, options=nothing)
    agent = _get_registered_agent(runtime, agent_name)
    session = _load_or_create_session!(runtime, String(agent_name), session_id)
    response = AgentFramework.run_agent(agent, inputs; session = session, options = options)
    AgentFramework.save_session!(runtime.session_store, session)
    return (response = response, session = session, history = _load_history(agent, session.id))
end

function get_agent_session(runtime::HostedRuntime, agent_name::AbstractString, session_id::AbstractString)
    agent = _get_registered_agent(runtime, agent_name)
    session = AgentFramework.load_session(runtime.session_store, String(session_id))
    session === nothing && throw(KeyError("Unknown hosted session: $(session_id)"))
    _session_agent_name(session) == agent_name || throw(ArgumentError("Session $(session_id) does not belong to hosted agent $(agent_name)"))
    return (session = session, history = _load_history(agent, session.id))
end

function list_agent_sessions(runtime::HostedRuntime, agent_name::AbstractString)
    _get_registered_agent(runtime, agent_name)
    sessions = AgentFramework.AgentSession[]
    for session_id in AgentFramework.list_sessions(runtime.session_store)
        session = AgentFramework.load_session(runtime.session_store, session_id)
        session === nothing && continue
        _session_agent_name(session) == agent_name || continue
        push!(sessions, session)
    end
    sort!(sessions, by = session -> session.id)
    return sessions
end

function _clear_history!(provider::AgentFramework.InMemoryHistoryProvider, session_id::AbstractString)
    delete!(provider.store, String(session_id))
    return nothing
end

function _clear_history!(provider::AgentFramework.FileHistoryProvider, session_id::AbstractString)
    path = joinpath(provider.directory, String(session_id) * ".json")
    isfile(path) && rm(path)
    return nothing
end

_clear_history!(provider::AgentFramework.AbstractHistoryProvider, session_id::AbstractString) = nothing

function delete_agent_session!(runtime::HostedRuntime, agent_name::AbstractString, session_id::AbstractString)
    agent = _get_registered_agent(runtime, agent_name)
    session = AgentFramework.load_session(runtime.session_store, String(session_id))
    session === nothing && return false
    _session_agent_name(session) == agent_name || throw(ArgumentError("Session $(session_id) does not belong to hosted agent $(agent_name)"))
    AgentFramework.delete_session!(runtime.session_store, session.id)
    provider = _agent_history_provider(agent)
    provider === nothing || _clear_history!(provider, session.id)
    return true
end

_workflow_internal_name(workflow_name::AbstractString, run_id::AbstractString) = String(workflow_name) * "::" * String(run_id)

function _instantiate_workflow(runtime::HostedRuntime, workflow_name::AbstractString, run_id::AbstractString)
    blueprint = _get_registered_workflow(runtime, workflow_name)
    workflow = deepcopy(blueprint)
    workflow.name = _workflow_internal_name(workflow_name, run_id)
    return workflow
end

function _workflow_error_string(result::AgentFramework.WorkflowRunResult)
    for event in reverse(result.events)
        event.type == AgentFramework.EVT_FAILED && event.details !== nothing && return event.details.message
        event.type == AgentFramework.EVT_ERROR && event.data !== nothing && return string(event.data)
    end
    return nothing
end

function _workflow_event_to_dict(event::AgentFramework.WorkflowEvent)
    payload = Dict{String, Any}(
        "type" => string(event.type),
        "timestamp" => _jsonish(event.timestamp),
    )
    event.executor_id === nothing || (payload["executor_id"] = event.executor_id)
    event.iteration === nothing || (payload["iteration"] = event.iteration)
    event.state === nothing || (payload["state"] = string(event.state))
    event.data === nothing || (payload["data"] = _jsonish(event.data))
    event.request_id === nothing || (payload["request_id"] = event.request_id)
    if event.details !== nothing
        payload["details"] = Dict{String, Any}(
            "message" => event.details.message,
            "error_type" => event.details.error_type,
            "stacktrace" => event.details.stacktrace,
        )
    end
    return payload
end

function _record_workflow_run(
    run_id::AbstractString,
    workflow_name::AbstractString,
    internal_name::AbstractString,
    result::AgentFramework.WorkflowRunResult;
    checkpoint_id::Union{Nothing, String}=nothing,
    created_at::DateTime=_utcnow(),
    metadata::Dict{String, Any}=Dict{String, Any}(),
)
    return HostedWorkflowRun(
        id = String(run_id),
        workflow_name = String(workflow_name),
        internal_workflow_name = String(internal_name),
        state = AgentFramework.get_final_state(result),
        checkpoint_id = checkpoint_id,
        outputs = Any[_jsonish(output) for output in AgentFramework.get_outputs(result)],
        pending_requests = Dict{String, Any}[_workflow_event_to_dict(event) for event in AgentFramework.get_request_info_events(result)],
        events = Dict{String, Any}[_workflow_event_to_dict(event) for event in result.events],
        error = _workflow_error_string(result),
        created_at = created_at,
        updated_at = _utcnow(),
        metadata = deepcopy(metadata),
    )
end

function _latest_checkpoint_id(runtime::HostedRuntime, internal_name::AbstractString)
    checkpoint = AgentFramework.get_latest(runtime.checkpoint_storage, String(internal_name))
    checkpoint === nothing && return nothing
    return checkpoint.id
end

function start_workflow_run!(runtime::HostedRuntime, workflow_name::AbstractString, input; run_id::Union{Nothing, AbstractString}=nothing, metadata::AbstractDict=Dict{String, Any}())
    _get_registered_workflow(runtime, workflow_name)
    run_id_string = run_id === nothing ? string(uuid4()) : String(run_id)
    workflow = _instantiate_workflow(runtime, workflow_name, run_id_string)
    result = AgentFramework.run_workflow(workflow, input; checkpoint_storage = runtime.checkpoint_storage)
    checkpoint_id = _latest_checkpoint_id(runtime, workflow.name)
    record = _record_workflow_run(
        run_id_string,
        String(workflow_name),
        workflow.name,
        result;
        checkpoint_id = checkpoint_id,
        metadata = Dict{String, Any}(_materialize_json(metadata)),
    )
    save_run!(runtime.run_store, record)
    return record
end

function get_workflow_run(runtime::HostedRuntime, workflow_name::AbstractString, run_id::AbstractString)
    _get_registered_workflow(runtime, workflow_name)
    run = load_run(runtime.run_store, run_id)
    run === nothing && throw(KeyError("Unknown workflow run: $(run_id)"))
    run.workflow_name == workflow_name || throw(ArgumentError("Run $(run_id) belongs to hosted workflow $(run.workflow_name), not $(workflow_name)"))
    return run
end

function list_workflow_runs(runtime::HostedRuntime, workflow_name::Union{Nothing, AbstractString}=nothing)
    workflow_name === nothing || _get_registered_workflow(runtime, workflow_name)
    return list_runs(runtime.run_store, workflow_name)
end

function resume_workflow_run!(runtime::HostedRuntime, workflow_name::AbstractString, run_id::AbstractString, responses::AbstractDict)
    run = get_workflow_run(runtime, workflow_name, run_id)
    run.checkpoint_id === nothing && throw(ArgumentError("Workflow run $(run_id) cannot be resumed because it has no checkpoint"))
    workflow = _instantiate_workflow(runtime, workflow_name, run.id)
    workflow.name = run.internal_workflow_name
    result = AgentFramework.run_workflow(
        workflow;
        checkpoint_id = run.checkpoint_id,
        checkpoint_storage = runtime.checkpoint_storage,
        responses = Dict{String, Any}(_materialize_json(responses)),
    )
    updated = _record_workflow_run(
        run.id,
        run.workflow_name,
        run.internal_workflow_name,
        result;
        checkpoint_id = _latest_checkpoint_id(runtime, run.internal_workflow_name),
        created_at = run.created_at,
        metadata = run.metadata,
    )
    save_run!(runtime.run_store, updated)
    return updated
end
