# Workflow type definitions for AgentFramework.jl
# Core types for the DAG-based workflow system.
# Design draws from both Python (Pregel-style supersteps) and C# (typed executors, tagged-union edges).

# ── Workflow Message ─────────────────────────────────────────────────────────

"""
    WorkflowMessageType

Type of message flowing between executors.
"""
@enum WorkflowMessageType begin
    STANDARD_MESSAGE
    RESPONSE_MESSAGE
end

"""
    WorkflowMessage

A message flowing between executors in a workflow.

# Fields
- `data::Any`: The actual payload.
- `source_id::String`: Sending executor ID.
- `target_id::Union{Nothing, String}`: Specific target (nothing = use edge routing).
- `type::WorkflowMessageType`: Standard or response message.
"""
Base.@kwdef struct WorkflowMessage
    data::Any
    source_id::String
    target_id::Union{Nothing, String} = nothing
    type::WorkflowMessageType = STANDARD_MESSAGE
end

# ── Workflow Run State ───────────────────────────────────────────────────────

"""
    WorkflowRunState

State of a workflow execution.
"""
@enum WorkflowRunState begin
    WF_STARTED
    WF_IN_PROGRESS
    WF_IDLE
    WF_IDLE_WITH_PENDING_REQUESTS
    WF_FAILED
    WF_CANCELLED
end

# ── Workflow Events ──────────────────────────────────────────────────────────

"""
    WorkflowEventType

Type of workflow event for observability.
"""
@enum WorkflowEventType begin
    EVT_STARTED
    EVT_STATUS
    EVT_FAILED
    EVT_OUTPUT
    EVT_DATA
    EVT_REQUEST_INFO
    EVT_WARNING
    EVT_ERROR
    EVT_SUPERSTEP_STARTED
    EVT_SUPERSTEP_COMPLETED
    EVT_EXECUTOR_INVOKED
    EVT_EXECUTOR_COMPLETED
    EVT_EXECUTOR_FAILED
end

"""
    WorkflowErrorDetails

Details about a workflow or executor error.

# Fields
- `error_type::String`: Exception type name.
- `message::String`: Error message.
- `executor_id::Union{Nothing, String}`: Executor that failed.
- `stacktrace::Union{Nothing, String}`: Full stacktrace.
"""
Base.@kwdef struct WorkflowErrorDetails
    error_type::String
    message::String
    executor_id::Union{Nothing, String} = nothing
    stacktrace::Union{Nothing, String} = nothing
end

function WorkflowErrorDetails(e::Exception; executor_id::Union{Nothing, String}=nothing)
    WorkflowErrorDetails(
        error_type = string(typeof(e)),
        message = sprint(showerror, e),
        executor_id = executor_id,
        stacktrace = sprint(Base.show_backtrace, catch_backtrace()),
    )
end

"""
    WorkflowEvent{T}

An event emitted during workflow execution for observability and streaming.

# Fields
- `type::WorkflowEventType`: Event category.
- `data::T`: Event-specific payload.
- `executor_id::Union{Nothing, String}`: Related executor.
- `state::Union{Nothing, WorkflowRunState}`: For status events.
- `details::Union{Nothing, WorkflowErrorDetails}`: For error events.
- `iteration::Union{Nothing, Int}`: For superstep events.
- `request_id::Union{Nothing, String}`: For request_info events.
- `timestamp::Float64`: Event timestamp.
"""
Base.@kwdef struct WorkflowEvent{T}
    type::WorkflowEventType
    data::T = nothing
    executor_id::Union{Nothing, String} = nothing
    state::Union{Nothing, WorkflowRunState} = nothing
    details::Union{Nothing, WorkflowErrorDetails} = nothing
    iteration::Union{Nothing, Int} = nothing
    request_id::Union{Nothing, String} = nothing
    timestamp::Float64 = time()
end

# Event factory methods
event_started() = WorkflowEvent{Nothing}(type=EVT_STARTED)
event_status(state::WorkflowRunState) = WorkflowEvent{Nothing}(type=EVT_STATUS, state=state)
event_failed(details::WorkflowErrorDetails) = WorkflowEvent{Nothing}(type=EVT_FAILED, details=details)
event_output(executor_id::String, data) = WorkflowEvent{Any}(type=EVT_OUTPUT, executor_id=executor_id, data=data)
event_executor_invoked(executor_id::String) = WorkflowEvent{Nothing}(type=EVT_EXECUTOR_INVOKED, executor_id=executor_id)
event_executor_completed(executor_id::String, data=nothing) = WorkflowEvent{Any}(type=EVT_EXECUTOR_COMPLETED, executor_id=executor_id, data=data)
event_executor_failed(executor_id::String, details::WorkflowErrorDetails) = WorkflowEvent{Nothing}(type=EVT_EXECUTOR_FAILED, executor_id=executor_id, details=details)
event_superstep_started(iteration::Int) = WorkflowEvent{Nothing}(type=EVT_SUPERSTEP_STARTED, iteration=iteration)
event_superstep_completed(iteration::Int) = WorkflowEvent{Nothing}(type=EVT_SUPERSTEP_COMPLETED, iteration=iteration)
event_request_info(request_id::String, executor_id::String, data) = WorkflowEvent{Any}(type=EVT_REQUEST_INFO, request_id=request_id, executor_id=executor_id, data=data)
event_warning(msg::String) = WorkflowEvent{String}(type=EVT_WARNING, data=msg)
event_error(e::Exception) = WorkflowEvent{Exception}(type=EVT_ERROR, data=e)

function Base.show(io::IO, e::WorkflowEvent)
    print(io, "WorkflowEvent(", e.type)
    e.executor_id !== nothing && print(io, ", executor=", e.executor_id)
    e.state !== nothing && print(io, ", state=", e.state)
    e.iteration !== nothing && print(io, ", iteration=", e.iteration)
    print(io, ")")
end

# ── Workflow Run Result ──────────────────────────────────────────────────────

"""
    WorkflowRunResult

Container for all events from a workflow execution.

# Fields
- `events::Vector{WorkflowEvent}`: All events in order.
- `state::WorkflowRunState`: Final workflow state.
"""
Base.@kwdef struct WorkflowRunResult
    events::Vector{WorkflowEvent} = WorkflowEvent[]
    state::WorkflowRunState = WF_IDLE
end

"""Get all output data from a workflow run."""
function get_outputs(result::WorkflowRunResult)
    [e.data for e in result.events if e.type == EVT_OUTPUT]
end

"""Get all request_info events (for human-in-the-loop)."""
function get_request_info_events(result::WorkflowRunResult)
    [e for e in result.events if e.type == EVT_REQUEST_INFO]
end

"""Get the final state."""
function get_final_state(result::WorkflowRunResult)
    for e in reverse(result.events)
        if e.type == EVT_STATUS && e.state !== nothing
            return e.state
        end
    end
    return result.state
end
