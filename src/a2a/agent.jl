const A2A_CONTEXT_ID_STATE_KEY = "__a2a_context_id__"
const A2A_TASK_ID_STATE_KEY = "__a2a_task_id__"

mutable struct A2ARemoteAgent <: AbstractAgent
    id::String
    name::String
    description::String
    client::A2AClient
    context_providers::Vector{Any}
    poll_interval::Float64
    max_polls::Union{Nothing, Int}
end

function A2ARemoteAgent(;
    client::Union{Nothing, A2AClient} = nothing,
    url::Union{Nothing, AbstractString} = nothing,
    id::Union{Nothing, AbstractString} = nothing,
    name::AbstractString = "A2A Agent",
    description::AbstractString = "",
    context_providers::AbstractVector = Any[],
    poll_interval::Real = 1.0,
    max_polls::Union{Nothing, Integer} = nothing,
)
    client === nothing && url === nothing && throw(A2AError("Either `client` or `url` must be provided"))
    resolved_client = client === nothing ? A2AClient(base_url = String(url)) : client
    return A2ARemoteAgent(
        id === nothing ? string(UUIDs.uuid4()) : String(id),
        String(name),
        String(description),
        resolved_client,
        Any[context_providers...],
        Float64(poll_interval),
        max_polls === nothing ? nothing : Int(max_polls),
    )
end

function Base.show(io::IO, agent::A2ARemoteAgent)
    print(io, "A2ARemoteAgent(\"", agent.name, "\", \"", agent.client.base_url, "\")")
end

function create_session(agent::A2ARemoteAgent; session_id::Union{Nothing, String}=nothing)::AgentSession
    return AgentSession(
        id = session_id === nothing ? string(UUIDs.uuid4()) : session_id,
        state = Dict{String, Any}(
            A2A_CONTEXT_ID_STATE_KEY => nothing,
            A2A_TASK_ID_STATE_KEY => nothing,
        ),
    )
end

function _provider_state_key(provider)::String
    if hasproperty(provider, :source_id)
        source_id = getproperty(provider, :source_id)
        if source_id isa AbstractString && !isempty(source_id)
            return String(source_id)
        end
    end
    return string(typeof(provider))
end

function _provider_state_dict!(session::AgentSession, provider)::Dict{String, Any}
    key = _provider_state_key(provider)
    state = get!(session.state, key, Dict{String, Any}())
    state isa Dict{String, Any} && return state
    if state isa AbstractDict
        materialized = Dict{String, Any}(string(name) => value for (name, value) in pairs(state))
        session.state[key] = materialized
        return materialized
    end
    throw(AgentError("Context provider state for '$key' must be a dictionary"))
end

function _run_before_context_providers!(agent::A2ARemoteAgent, session::AgentSession, ctx::SessionContext)
    for provider in agent.context_providers
        before_run!(provider, agent, session, ctx, _provider_state_dict!(session, provider))
    end
end

function _run_after_context_providers!(agent::A2ARemoteAgent, session::AgentSession, ctx::SessionContext)
    for provider in reverse(agent.context_providers)
        after_run!(provider, agent, session, ctx, _provider_state_dict!(session, provider))
    end
end

function _session_context_id(session::AgentSession)::Union{Nothing, String}
    return _maybe_string(get(session.state, A2A_CONTEXT_ID_STATE_KEY, nothing))
end

function _session_task_id(session::AgentSession)::Union{Nothing, String}
    return _maybe_string(get(session.state, A2A_TASK_ID_STATE_KEY, nothing))
end

function _resolved_max_polls(agent::A2ARemoteAgent)::Union{Nothing, Int}
    return agent.max_polls === nothing ? agent.client.max_polls : agent.max_polls
end

function _update_session_from_response!(session::AgentSession, response::AgentResponse)
    if response.continuation_token isa A2AContinuationToken
        token = response.continuation_token
        token.context_id !== nothing && (session.state[A2A_CONTEXT_ID_STATE_KEY] = token.context_id)
        session.state[A2A_TASK_ID_STATE_KEY] = token.task_id
    end

    raw = response.raw_representation
    if raw isa A2ATask
        raw.context_id !== nothing && (session.state[A2A_CONTEXT_ID_STATE_KEY] = raw.context_id)
        session.state[A2A_TASK_ID_STATE_KEY] = raw.id
    elseif raw isa AbstractDict
        values = _dict(raw)
        kind = lowercase(string(get(values, "kind", "")))
        if kind == "message"
            context_id = _maybe_string(get(values, "contextId", nothing))
            context_id !== nothing && (session.state[A2A_CONTEXT_ID_STATE_KEY] = context_id)
        end
    end

    return session
end

function _new_or_resume_response(
    agent::A2ARemoteAgent,
    session::AgentSession,
    input_messages::Vector{Message},
    background::Bool,
)::AgentResponse
    if !isempty(input_messages)
        reference_task_ids = String[]
        existing_task_id = _session_task_id(session)
        existing_task_id !== nothing && push!(reference_task_ids, existing_task_id)

        return send_message(
            agent.client,
            input_messages[end];
            context_id = _session_context_id(session),
            reference_task_ids = reference_task_ids,
            background = background,
            poll_interval = agent.poll_interval,
            max_polls = _resolved_max_polls(agent),
        )
    end

    task_id = _session_task_id(session)
    task_id === nothing && throw(A2AError("At least one message is required unless the session already tracks an A2A task"))

    return get_task(
        agent.client,
        A2AContinuationToken(task_id = task_id, context_id = _session_context_id(session));
        background = background,
        poll_interval = agent.poll_interval,
        max_polls = _resolved_max_polls(agent),
    )
end

function _updates_from_response(response::AgentResponse)::Vector{AgentResponseUpdate}
    updates = if isempty(response.messages)
        [AgentResponseUpdate(role = :assistant, contents = Content[])]
    else
        [
            AgentResponseUpdate(
                role = message.role,
                contents = copy(message.contents),
                response_id = response.response_id,
                raw_representation = response.raw_representation,
            )
            for message in response.messages
        ]
    end

    last_update = last(updates)
    last_update.finish_reason = response.finish_reason
    last_update.model_id = response.model_id
    last_update.usage_details = response.usage_details
    last_update.response_id = response.response_id
    last_update.continuation_token = response.continuation_token
    last_update.raw_representation = response.raw_representation

    return updates
end

function _emit_response!(channel::Channel{AgentResponseUpdate}, response::AgentResponse)::Bool
    for update in _updates_from_response(response)
        try
            put!(channel, update)
        catch
            return false
        end
    end
    return true
end

function run_agent(
    agent::A2ARemoteAgent,
    inputs::Union{AgentRunInputs, Nothing} = nothing;
    session::Union{Nothing, AgentSession} = nothing,
    options = nothing,
)::AgentResponse
    sess = session === nothing ? create_session(agent) : session
    input_messages = normalize_messages(inputs)
    ctx = SessionContext(
        session_id = sess.id,
        input_messages = input_messages,
        options = options === nothing ? Dict{String, Any}() : Dict{String, Any}("options" => options),
    )

    _run_before_context_providers!(agent, sess, ctx)
    response = _new_or_resume_response(agent, sess, input_messages, false)
    _update_session_from_response!(sess, response)
    ctx.response = response
    _run_after_context_providers!(agent, sess, ctx)
    return response
end

function poll_task(
    agent::A2ARemoteAgent,
    token::A2AContinuationToken;
    session::Union{Nothing, AgentSession} = nothing,
    background::Bool = true,
)::AgentResponse
    response = get_task(
        agent.client,
        token;
        background = background,
        poll_interval = agent.poll_interval,
        max_polls = _resolved_max_polls(agent),
    )
    session !== nothing && _update_session_from_response!(session, response)
    return response
end

function run_agent_streaming(
    agent::A2ARemoteAgent,
    inputs::Union{AgentRunInputs, Nothing} = nothing;
    session::Union{Nothing, AgentSession} = nothing,
    options = nothing,
)::ResponseStream{AgentResponseUpdate}
    sess = session === nothing ? create_session(agent) : session
    input_messages = normalize_messages(inputs)
    ctx = SessionContext(
        session_id = sess.id,
        input_messages = input_messages,
        options = options === nothing ? Dict{String, Any}() : Dict{String, Any}("options" => options),
    )

    _run_before_context_providers!(agent, sess, ctx)

    channel = Channel{AgentResponseUpdate}(32)
    stream = ResponseStream{AgentResponseUpdate}(channel)

    task = Threads.@spawn begin
        try
            response = _new_or_resume_response(agent, sess, input_messages, true)
            _update_session_from_response!(sess, response)
            last_signature = nothing
            attempts = 0

            while true
                signature = _response_signature(response)
                if signature != last_signature
                    _emit_response!(channel, response) || break
                    last_signature = signature
                end

                if !(response.continuation_token isa A2AContinuationToken)
                    ctx.response = response
                    _run_after_context_providers!(agent, sess, ctx)
                    lock(stream._lock) do
                        stream.final_response = response
                    end
                    break
                end

                attempts += 1
                limit = _resolved_max_polls(agent)
                if limit !== nothing && attempts > limit
                    throw(A2ATimeoutError("A2A task $(response.continuation_token.task_id) did not complete after $(limit) polls"))
                end

                sleep(agent.poll_interval)
                response = get_task(agent.client, response.continuation_token; background = true)
                _update_session_from_response!(sess, response)
            end
        catch error
            lock(stream._lock) do
                stream.error = error
            end
        finally
            close(channel)
        end
    end

    lock(stream._lock) do
        stream.task = task
    end

    return stream
end
