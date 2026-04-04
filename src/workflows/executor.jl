# Executor abstraction for AgentFramework.jl workflows.
# An executor is a node in the workflow DAG that processes messages.
# Inspired by C#'s Executor<TInput, TOutput> and Python's Executor + @handler.

"""
    AbstractExecutorHandler

Abstract type for executor handler implementations. Concrete executors
should subtype this and implement `handle!`.
"""
abstract type AbstractExecutorHandler end

"""
    ExecutorSpec

Specification for an executor — declares what message types it handles
and what types it produces, plus the handler function.

# Fields
- `id::String`: Unique executor identifier.
- `description::String`: Human-readable description.
- `input_types::Vector{DataType}`: Message types this executor can handle.
- `output_types::Vector{DataType}`: Types sent via `send_message`.
- `yield_types::Vector{DataType}`: Types yielded as workflow output.
- `handler::Function`: The handler function `(data, context) -> nothing`.
"""
Base.@kwdef struct ExecutorSpec
    id::String
    description::String = ""
    input_types::Vector{DataType} = DataType[Any]
    output_types::Vector{DataType} = DataType[Any]
    yield_types::Vector{DataType} = DataType[Any]
    handler::Function
end

function Base.show(io::IO, e::ExecutorSpec)
    print(io, "ExecutorSpec(\"", e.id, "\")")
end

"""
    WorkflowContext

Context passed to executor handlers, providing access to messaging,
state, and workflow output.

# Fields
- `executor_id::String`: ID of the current executor.
- `source_ids::Vector{String}`: IDs of executors that sent the current message.
- `_sent_messages::Vector{WorkflowMessage}`: Messages sent during this handler invocation.
- `_yielded_outputs::Vector{Any}`: Outputs yielded during this handler invocation.
- `_events::Vector{WorkflowEvent}`: Events emitted during this handler invocation.
- `_state::Dict{String, Any}`: Shared workflow state.
- `_request_infos::Vector{WorkflowEvent}`: Request info events for HIL.
"""
Base.@kwdef mutable struct WorkflowContext
    executor_id::String
    source_ids::Vector{String} = String[]
    _sent_messages::Vector{WorkflowMessage} = WorkflowMessage[]
    _yielded_outputs::Vector{Any} = Any[]
    _events::Vector{WorkflowEvent} = WorkflowEvent[]
    _state::Dict{String, Any} = Dict{String, Any}()
    _request_infos::Vector{WorkflowEvent} = WorkflowEvent[]
end

"""
    send_message(ctx::WorkflowContext, data; target_id=nothing)

Send a message to downstream executors. If `target_id` is specified,
the message is routed to that specific executor; otherwise, edge routing applies.
"""
function send_message(ctx::WorkflowContext, data; target_id::Union{Nothing, String}=nothing)
    msg = WorkflowMessage(
        data = data,
        source_id = ctx.executor_id,
        target_id = target_id,
    )
    push!(ctx._sent_messages, msg)
    return nothing
end

"""
    yield_output(ctx::WorkflowContext, data)

Yield data as a workflow-level output. This data will appear in the
`WorkflowRunResult.events` as an `EVT_OUTPUT` event.
"""
function yield_output(ctx::WorkflowContext, data)
    push!(ctx._yielded_outputs, data)
    push!(ctx._events, event_output(ctx.executor_id, data))
    return nothing
end

"""
    get_state(ctx::WorkflowContext, key::String, default=nothing)

Read a value from the shared workflow state.
"""
function get_state(ctx::WorkflowContext, key::String, default=nothing)
    get(ctx._state, key, default)
end

"""
    set_state!(ctx::WorkflowContext, key::String, value)

Write a value to the shared workflow state. The value becomes visible
to other executors in the next superstep.
"""
function set_state!(ctx::WorkflowContext, key::String, value)
    ctx._state[key] = value
    return nothing
end

"""
    request_info(ctx::WorkflowContext, request_data; response_type=Any, request_id=nothing)

Request external information (human-in-the-loop). The workflow will pause
after the current superstep completes, emitting a `request_info` event.
The caller can then resume the workflow with a response.
"""
function request_info(ctx::WorkflowContext, request_data; response_type::Type=Any, request_id::Union{Nothing, String}=nothing)
    rid = request_id !== nothing ? request_id : string(UUIDs.uuid4())
    evt = event_request_info(rid, ctx.executor_id, request_data)
    push!(ctx._request_infos, evt)
    push!(ctx._events, evt)
    return rid
end

# ── Executor Execution ───────────────────────────────────────────────────────

"""
    execute_handler(spec::ExecutorSpec, message, source_ids, state) -> WorkflowContext

Execute an executor's handler with the given message, returning the
context containing sent messages, yielded outputs, and events.
"""
function execute_handler(
    spec::ExecutorSpec,
    message,
    source_ids::Vector{String},
    state::Dict{String, Any},
)::WorkflowContext
    ctx = WorkflowContext(
        executor_id = spec.id,
        source_ids = source_ids,
        _state = state,
    )

    spec.handler(message, ctx)

    return ctx
end

# ── Agent Executor ───────────────────────────────────────────────────────────

"""
    agent_executor(id::String, agent::Agent; kwargs...) -> ExecutorSpec

Create an executor that wraps an Agent. The handler runs the agent with
the incoming message and yields the response text as output.

# Arguments
- `id::String`: Executor ID.
- `agent::Agent`: The agent to wrap.
- `yield_response::Bool`: If true, yields the agent response as workflow output (default: false).
- `forward_response::Bool`: If true, sends the agent response text downstream (default: true).
"""
function agent_executor(
    id::String,
    agent::Agent;
    yield_response::Bool = false,
    forward_response::Bool = true,
)::ExecutorSpec
    ExecutorSpec(
        id = id,
        description = "Agent: $(agent.name)",
        input_types = DataType[String],
        output_types = DataType[String],
        yield_types = yield_response ? DataType[String] : DataType[],
        handler = (message, ctx) -> begin
            # Run the agent with the message
            input = message isa AbstractString ? message : JSON3.write(message)
            response = run_agent(agent, input)
            text = response.text

            if forward_response
                send_message(ctx, text)
            end
            if yield_response
                yield_output(ctx, text)
            end
        end,
    )
end
