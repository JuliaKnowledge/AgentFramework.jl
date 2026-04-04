# Enhanced AgentExecutor for workflow integration.
# Provides session persistence, message caching, middleware integration,
# and multi-type input handling — inspired by the Python/C# AgentExecutor classes.

"""
    AgentExecutorRequest

Request type for AgentExecutor — carries messages and control flags.

# Fields
- `messages::Vector{Message}`: Messages to add to the conversation.
- `should_respond::Bool`: Whether the agent should generate a response (default: true).
- `metadata::Dict{String, Any}`: Arbitrary metadata for the request.
"""
Base.@kwdef struct AgentExecutorRequest
    messages::Vector{Message} = Message[]
    should_respond::Bool = true
    metadata::Dict{String, Any} = Dict{String, Any}()
end

"""
    AgentExecutorResponse

Response type from AgentExecutor — carries agent response and conversation context.

# Fields
- `agent_response::AgentResponse`: The underlying agent response.
- `full_conversation::Vector{Message}`: Complete conversation history up to this point.
- `executor_id::String`: ID of the executor that produced this response.
- `metadata::Dict{String, Any}`: Arbitrary metadata.
"""
Base.@kwdef struct AgentExecutorResponse
    agent_response::AgentResponse
    full_conversation::Vector{Message}
    executor_id::String
    metadata::Dict{String, Any} = Dict{String, Any}()
end

"""
    AgentExecutor

Enhanced executor wrapping an Agent for use in workflows. Provides:
- Session persistence across workflow invocations
- Message history caching for conversation continuity
- Middleware integration (agent, chat, function layers)
- Multiple input handling (String, Message, AgentExecutorRequest, AgentExecutorResponse)
- HIL request forwarding

# Fields
- `id::String`: Unique executor identifier.
- `agent::Agent`: The wrapped agent.
- `session::AgentSession`: Persistent session across invocations.
- `yield_response::Bool`: If true, yields AgentExecutorResponse as workflow output.
- `forward_response::Bool`: If true, sends AgentExecutorResponse downstream.
- `auto_respond::Bool`: If true, automatically responds when handling a chained response.
- `_message_cache::Vector{Message}`: Accumulated input messages for the next agent run.
- `_full_conversation::Vector{Message}`: Full conversation history (inputs + outputs).
"""
Base.@kwdef mutable struct AgentExecutor
    id::String
    agent::Agent
    session::AgentSession = AgentSession()
    yield_response::Bool = false
    forward_response::Bool = true
    auto_respond::Bool = true
    _message_cache::Vector{Message} = Message[]
    _full_conversation::Vector{Message} = Message[]
end

"""
    AgentExecutor(id::String, agent::Agent; kwargs...)

Construct an AgentExecutor with the given ID and agent.
"""
function AgentExecutor(id::String, agent::Agent;
    session::AgentSession = AgentSession(),
    yield_response::Bool = false,
    forward_response::Bool = true,
    auto_respond::Bool = true,
)
    AgentExecutor(
        id = id,
        agent = agent,
        session = session,
        yield_response = yield_response,
        forward_response = forward_response,
        auto_respond = auto_respond,
    )
end

"""
    to_executor_spec(ae::AgentExecutor) -> ExecutorSpec

Convert an AgentExecutor into an ExecutorSpec for use in workflow builders.
The handler dispatches based on the incoming message type.
"""
function to_executor_spec(ae::AgentExecutor)::ExecutorSpec
    handler = (message, ctx) -> begin
        if message isa AgentExecutorRequest
            handle_request!(ae, message, ctx)
        elseif message isa AgentExecutorResponse
            handle_response!(ae, message, ctx)
        elseif message isa AbstractString
            handle_string!(ae, string(message), ctx)
        elseif message isa Message
            handle_message!(ae, message, ctx)
        else
            handle_string!(ae, string(message), ctx)
        end
    end

    ExecutorSpec(
        id = ae.id,
        description = "AgentExecutor: $(ae.agent.name)",
        input_types = DataType[String, Message, AgentExecutorRequest, AgentExecutorResponse],
        output_types = DataType[AgentExecutorResponse],
        yield_types = ae.yield_response ? DataType[AgentExecutorResponse] : DataType[],
        handler = handler,
    )
end

"""
    handle_request!(ae::AgentExecutor, request::AgentExecutorRequest, ctx::WorkflowContext)

Handle an AgentExecutorRequest. Appends request messages to the cache and,
if `should_respond` is true, runs the agent.
"""
function handle_request!(ae::AgentExecutor, request::AgentExecutorRequest, ctx::WorkflowContext)
    append!(ae._message_cache, request.messages)
    if request.should_respond
        _run_and_emit!(ae, ctx)
    end
    return nothing
end

"""
    handle_string!(ae::AgentExecutor, input::String, ctx::WorkflowContext)

Handle a plain string input. Creates a user Message, caches it, and runs the agent.
"""
function handle_string!(ae::AgentExecutor, input::String, ctx::WorkflowContext)
    msg = Message(:user, input)
    push!(ae._message_cache, msg)
    _run_and_emit!(ae, ctx)
    return nothing
end

"""
    handle_message!(ae::AgentExecutor, msg::Message, ctx::WorkflowContext)

Handle a Message input. Caches the message and runs the agent.
"""
function handle_message!(ae::AgentExecutor, msg::Message, ctx::WorkflowContext)
    push!(ae._message_cache, msg)
    _run_and_emit!(ae, ctx)
    return nothing
end

"""
    handle_response!(ae::AgentExecutor, response::AgentExecutorResponse, ctx::WorkflowContext)

Handle a chained AgentExecutorResponse from a prior executor.
Replaces the message cache with the prior response's full conversation.
If `auto_respond` is true, runs the agent.
"""
function handle_response!(ae::AgentExecutor, response::AgentExecutorResponse, ctx::WorkflowContext)
    ae._message_cache = copy(response.full_conversation)
    if ae.auto_respond
        _run_and_emit!(ae, ctx)
    end
    return nothing
end

"""
    _run_and_emit!(ae::AgentExecutor, ctx::WorkflowContext)

Internal: Run the agent with cached messages, update conversation history,
and emit results according to forward_response / yield_response settings.
"""
function _run_and_emit!(ae::AgentExecutor, ctx::WorkflowContext)
    agent_response = run_agent(ae.agent, ae._message_cache; session=ae.session)

    # Append response messages to full conversation
    append!(ae._full_conversation, ae._message_cache)
    append!(ae._full_conversation, agent_response.messages)

    # Build executor response
    response = AgentExecutorResponse(
        agent_response = agent_response,
        full_conversation = copy(ae._full_conversation),
        executor_id = ae.id,
    )

    # Clear message cache after running — next invocation starts fresh
    empty!(ae._message_cache)

    if ae.forward_response
        send_message(ctx, response)
    end
    if ae.yield_response
        yield_output(ctx, response)
    end

    return nothing
end

"""
    reset!(ae::AgentExecutor)

Reset the executor state: clear message cache, full conversation,
and create a fresh session.
"""
function reset!(ae::AgentExecutor)
    empty!(ae._message_cache)
    empty!(ae._full_conversation)
    ae.session = AgentSession()
    return nothing
end

"""
    get_conversation(ae::AgentExecutor) -> Vector{Message}

Return a copy of the full conversation history.
"""
function get_conversation(ae::AgentExecutor)::Vector{Message}
    return copy(ae._full_conversation)
end
