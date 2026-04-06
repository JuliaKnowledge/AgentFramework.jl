# Agent implementation for AgentFramework.jl
# Mirrors Python Agent with middleware, tools, context providers, and tool execution loop.

const DEFAULT_MAX_TOOL_ITERATIONS = 10
const _SESSION_ID_OPTION_KEY = "_agentframework_session_id"
const _THREAD_ID_OPTION_KEY = "_agentframework_thread_id"
const _INPUT_MESSAGES_OPTION_KEY = "_agentframework_input_messages"

"""
    Agent <: AbstractAgent

Full-featured agent with middleware, context providers, and automatic tool execution.

# Fields
- `name::String`: Agent name.
- `description::String`: Agent description.
- `instructions::String`: System instructions prepended to every conversation.
- `client::AbstractChatClient`: The underlying LLM chat client.
- `tools::Vector{FunctionTool}`: Tools available to this agent.
- `context_providers::Vector{<:AbstractContextProvider}`: Context engineering pipeline.
- `agent_middlewares::Vector`: Agent-level middleware functions.
- `chat_middlewares::Vector`: Chat-level middleware functions.
- `function_middlewares::Vector`: Function invocation middleware functions.
- `options::ChatOptions`: Default chat options.
- `max_tool_iterations::Int`: Maximum tool call rounds before stopping.

# Examples
```julia
client = OllamaChatClient(model="qwen3:8b")

@tool function get_weather(location::String)
    "Get weather for a location."
    return "Sunny, 22°C in \$location"
end

agent = Agent(
    name = "WeatherBot",
    instructions = "You are a helpful weather assistant.",
    client = client,
    tools = [get_weather],
)

response = run_agent(agent, "What's the weather in London?")
println(response.text)
```
"""
Base.@kwdef mutable struct Agent <: AbstractAgent
    name::String = "Agent"
    description::String = ""
    instructions::String = ""
    client::AbstractChatClient
    tools::Vector = []                   # FunctionTool, HandoffTool, or mixed
    context_providers::Vector{Any} = Any[]
    agent_middlewares::Vector = []
    chat_middlewares::Vector = []
    function_middlewares::Vector = []
    options::ChatOptions = ChatOptions()
    max_tool_iterations::Int = DEFAULT_MAX_TOOL_ITERATIONS
end

function Base.show(io::IO, a::Agent)
    print(io, "Agent(\"", a.name, "\", ", length(a.tools), " tools)")
end

"""
    create_session(agent::Agent; session_id=nothing) -> AgentSession

Create a new session for this agent.
"""
function create_session(agent::Agent; session_id::Union{Nothing, String} = nothing)::AgentSession
    AgentSession(id = session_id !== nothing ? session_id : string(UUIDs.uuid4()))
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
    legacy_key = string(typeof(provider))

    if !haskey(session.state, key) && key != legacy_key && haskey(session.state, legacy_key)
        session.state[key] = session.state[legacy_key]
    end

    state = get!(session.state, key, Dict{String, Any}())
    if state isa Dict{String, Any}
        return state
    elseif state isa AbstractDict
        materialized = Dict{String, Any}(string(k) => v for (k, v) in pairs(state))
        session.state[key] = materialized
        return materialized
    end

    throw(AgentError("Context provider state for '$key' must be a dictionary"))
end

function _run_before_context_providers!(agent::Agent, session::AgentSession, ctx::SessionContext)
    for provider in agent.context_providers
        before_run!(provider, agent, session, ctx, _provider_state_dict!(session, provider))
    end
end

function _run_after_context_providers!(agent::Agent, session::AgentSession, ctx::SessionContext)
    for provider in reverse(agent.context_providers)
        after_run!(provider, agent, session, ctx, _provider_state_dict!(session, provider))
    end
end

function _prepare_chat_options(
    agent::Agent,
    ctx::SessionContext,
    options_override::Union{Nothing, ChatOptions},
)::Tuple{ChatOptions, Vector{FunctionTool}}
    chat_options = agent.options
    if options_override !== nothing
        chat_options = merge_chat_options(chat_options, options_override)
    end

    all_tools = vcat(agent.tools, ctx.tools)
    # Normalize: convert HandoffTools to FunctionTools for the LLM
    function_tools = FunctionTool[]
    for t in all_tools
        if t isa FunctionTool
            push!(function_tools, t)
        elseif t isa HandoffTool
            push!(function_tools, handoff_as_function_tool(t))
        end
    end
    effective_tools = isempty(function_tools) ? chat_options.tools : function_tools
    additional = copy(chat_options.additional)
    additional[_SESSION_ID_OPTION_KEY] = ctx.session_id
    additional[_THREAD_ID_OPTION_KEY] = ctx.service_session_id
    additional[_INPUT_MESSAGES_OPTION_KEY] = copy(ctx.input_messages)

    chat_options = ChatOptions(
        model = chat_options.model,
        temperature = chat_options.temperature,
        top_p = chat_options.top_p,
        max_tokens = chat_options.max_tokens,
        stop = chat_options.stop,
        tools = effective_tools,
        tool_choice = chat_options.tool_choice,
        response_format = chat_options.response_format,
        additional = additional,
    )

    return chat_options, function_tools
end

function _sync_service_session!(session::AgentSession, ctx::SessionContext, conversation_id)
    conversation_id === nothing && return nothing
    value = String(conversation_id)
    session.thread_id = value
    ctx.service_session_id = value
    return nothing
end

function _refresh_chat_options_thread_id(chat_options::ChatOptions, ctx::SessionContext)::ChatOptions
    additional = copy(chat_options.additional)
    additional[_THREAD_ID_OPTION_KEY] = ctx.service_session_id
    return ChatOptions(
        model = chat_options.model,
        temperature = chat_options.temperature,
        top_p = chat_options.top_p,
        max_tokens = chat_options.max_tokens,
        stop = chat_options.stop,
        tools = chat_options.tools,
        tool_choice = chat_options.tool_choice,
        response_format = chat_options.response_format,
        additional = additional,
    )
end

# ── Core Run Implementation ──────────────────────────────────────────────────

"""
    run_agent(agent::Agent, inputs; session=nothing, options=nothing) -> AgentResponse

Run the agent with the given inputs and return a complete response.

# Arguments
- `agent::Agent`: The agent to run.
- `inputs`: String, Content, Message, or Vector of these.
- `session::Union{Nothing, AgentSession}`: Session for conversation continuity.
- `options::Union{Nothing, ChatOptions}`: Per-call option overrides.
"""
function run_agent(
    agent::Agent,
    inputs::Union{AgentRunInputs, Nothing} = nothing;
    session::Union{Nothing, AgentSession} = nothing,
    options::Union{Nothing, ChatOptions} = nothing,
)::AgentResponse
    # Ensure session
    sess = session !== nothing ? session : create_session(agent)

    # Normalize inputs
    input_messages = normalize_messages(inputs)

    # Build session context
    ctx = SessionContext(
        session_id = sess.id,
        service_session_id = sess.thread_id,
        input_messages = input_messages,
        options = options !== nothing ? Dict{String, Any}("chat_options" => options) : Dict{String, Any}(),
    )

    # Run context providers (before)
    _run_before_context_providers!(agent, sess, ctx)

    # Build agent context for middleware
    agent_ctx = AgentContext(
        agent = agent,
        messages = input_messages,
        session = sess,
        options = ctx.options,
        stream = false,
    )

    # Define the core handler (what runs inside middleware)
    function core_handler(actx::AgentContext)
        response = _execute_agent_run(agent, sess, ctx, options)
        actx.result = response
        return response
    end

    # Execute through agent middleware
    response = apply_agent_middleware(agent.agent_middlewares, agent_ctx, core_handler)

    # Set response on context for after_run providers
    ctx.response = response

    # Run context providers (after)
    _run_after_context_providers!(agent, sess, ctx)

    return response
end

"""
    run_agent_streaming(agent::Agent, inputs; session=nothing, options=nothing) -> ResponseStream{AgentResponseUpdate}

Run the agent with streaming output. Returns a `ResponseStream` that yields
`AgentResponseUpdate` items.
"""
function run_agent_streaming(
    agent::Agent,
    inputs::Union{AgentRunInputs, Nothing} = nothing;
    session::Union{Nothing, AgentSession} = nothing,
    options::Union{Nothing, ChatOptions} = nothing,
)::ResponseStream{AgentResponseUpdate}
    sess = session !== nothing ? session : create_session(agent)
    input_messages = normalize_messages(inputs)

    ctx = SessionContext(
        session_id = sess.id,
        service_session_id = sess.thread_id,
        input_messages = input_messages,
        options = options !== nothing ? Dict{String, Any}("chat_options" => options) : Dict{String, Any}(),
    )

    _run_before_context_providers!(agent, sess, ctx)

    ch = Channel{AgentResponseUpdate}(32)
    stream = ResponseStream{AgentResponseUpdate}(ch)

    agent_ctx = AgentContext(
        agent = agent,
        messages = input_messages,
        session = sess,
        options = ctx.options,
        stream = true,
    )

    function core_handler(actx::AgentContext)
        response = _execute_agent_run_streaming(agent, sess, ctx, options, ch)
        actx.result = response
        return response
    end

    task = Threads.@spawn begin
        try
            response = apply_agent_middleware(agent.agent_middlewares, agent_ctx, core_handler)
            ctx.response = response
            _run_after_context_providers!(agent, sess, ctx)
            lock(stream._lock) do
                stream.final_response = response
            end
        catch e
            lock(stream._lock) do
                stream.error = e
            end
        finally
            close(ch)
        end
    end

    lock(stream._lock) do
        stream.task = task
    end

    return stream
end

# ── Internal Execution ───────────────────────────────────────────────────────

function _execute_agent_run(
    agent::Agent,
    session::AgentSession,
    ctx::SessionContext,
    options_override::Union{Nothing, ChatOptions},
)::AgentResponse
    # Build full message list
    all_messages = _build_message_list(agent, ctx)

    chat_options, all_tools = _prepare_chat_options(agent, ctx, options_override)

    # Tool execution loop
    iteration = 0
    while iteration < agent.max_tool_iterations
        iteration += 1

        # Call chat client through middleware
        chat_ctx = ChatContext(
            messages = all_messages,
            options = chat_options,
            stream = false,
        )

        function chat_handler(cctx::ChatContext)
            resp = get_response(agent.client, cctx.messages, cctx.options)
            cctx.result = resp
            return resp
        end

        chat_response = apply_chat_middleware(agent.chat_middlewares, chat_ctx, chat_handler)

        # Handle middleware termination (result is not a ChatResponse)
        if !(chat_response isa ChatResponse)
            term_text = string(chat_response)
            return AgentResponse(
                messages = [Message(role=:assistant, contents=[text_content(term_text)])],
                finish_reason = STOP,
            )
        end

        _sync_service_session!(session, ctx, chat_response.conversation_id)
        chat_options = _refresh_chat_options_thread_id(chat_options, ctx)

        # Check for tool calls in response
        tool_calls = Content[]
        for msg in chat_response.messages
            for c in msg.contents
                if is_function_call(c)
                    push!(tool_calls, c)
                end
            end
        end

        if isempty(tool_calls)
            # No tool calls — return final response
            return AgentResponse(
                messages = chat_response.messages,
                response_id = chat_response.response_id,
                conversation_id = chat_response.conversation_id,
                finish_reason = chat_response.finish_reason,
                usage_details = chat_response.usage_details,
                model_id = chat_response.model_id,
                additional_properties = chat_response.additional_properties,
                raw_representation = chat_response.raw_representation,
            )
        end

        # Execute tool calls
        append!(all_messages, chat_response.messages)
        tool_results = _execute_tool_calls(agent, all_tools, tool_calls)
        push!(all_messages, Message(role=:tool, contents=tool_results))
    end

    # Max iterations reached — return last response
    @warn "Agent reached max tool iterations ($iteration)"
    return AgentResponse(messages = Message[])
end

function _execute_agent_run_streaming(
    agent::Agent,
    session::AgentSession,
    ctx::SessionContext,
    options_override::Union{Nothing, ChatOptions},
    channel::Channel{AgentResponseUpdate},
)::AgentResponse
    all_messages = _build_message_list(agent, ctx)

    chat_options, all_tools = _prepare_chat_options(agent, ctx, options_override)

    iteration = 0
    while iteration < agent.max_tool_iterations
        iteration += 1

        chat_ctx = ChatContext(
            messages = all_messages,
            options = chat_options,
            stream = true,
        )

        function chat_handler(cctx::ChatContext)
            updates = get_response_streaming(agent.client, cctx.messages, cctx.options)
            cctx.result = updates
            return updates
        end

        updates_channel = apply_chat_middleware(agent.chat_middlewares, chat_ctx, chat_handler)

        # Handle middleware termination in streaming
        if !(updates_channel isa Channel)
            term_text = string(updates_channel)
            term_update = AgentResponseUpdate(
                role = :assistant,
                contents = [text_content(term_text)],
                finish_reason = STOP,
            )
            try; put!(channel, term_update); catch; end
            return
        end

        # Collect updates and forward to caller
        updates = ChatResponseUpdate[]
        for update in updates_channel
            push!(updates, update)
            # Forward as AgentResponseUpdate
            agent_update = AgentResponseUpdate(
                role = update.role,
                contents = update.contents,
                finish_reason = update.finish_reason,
                model_id = update.model_id,
                usage_details = update.usage_details,
                response_id = update.response_id,
                conversation_id = update.conversation_id,
                raw_representation = update.raw_representation,
            )
            try
                put!(channel, agent_update)
            catch
                # Channel closed
                break
            end
        end

        # Build complete response from updates
        chat_response = ChatResponse(updates)
        _sync_service_session!(session, ctx, chat_response.conversation_id)
        chat_options = _refresh_chat_options_thread_id(chat_options, ctx)

        # Check for tool calls
        tool_calls = Content[]
        for msg in chat_response.messages
            for c in msg.contents
                if is_function_call(c)
                    push!(tool_calls, c)
                end
            end
        end

        if isempty(tool_calls)
            return AgentResponse(
                messages = chat_response.messages,
                response_id = chat_response.response_id,
                conversation_id = chat_response.conversation_id,
                finish_reason = chat_response.finish_reason,
                usage_details = chat_response.usage_details,
                model_id = chat_response.model_id,
                additional_properties = chat_response.additional_properties,
                raw_representation = chat_response.raw_representation,
            )
        end

        # Execute tools and continue loop
        append!(all_messages, chat_response.messages)
        tool_results = _execute_tool_calls(agent, all_tools, tool_calls)
        push!(all_messages, Message(role=:tool, contents=tool_results))
    end

    @warn "Agent streaming reached max tool iterations ($iteration)"
    return AgentResponse(messages = Message[])
end

# ── Helper Functions ─────────────────────────────────────────────────────────

function _build_message_list(agent::Agent, ctx::SessionContext)::Vector{Message}
    messages = Message[]

    # System instructions
    if !isempty(agent.instructions)
        push!(messages, Message(:system, agent.instructions))
    end

    # Additional instructions from context providers
    for inst in ctx.instructions
        push!(messages, Message(:system, inst))
    end

    # Context messages (from providers like history)
    append!(messages, get_all_context_messages(ctx))

    # Input messages
    append!(messages, ctx.input_messages)

    return messages
end

function _execute_tool_calls(
    agent::Agent,
    tools::Vector{FunctionTool},
    tool_calls::Vector{Content},
)::Vector{Content}
    results = Content[]

    # Check if any tool calls need approval
    approval_needed = false
    for tc in tool_calls
        tool = find_tool(tools, something(tc.name, ""))
        if tool !== nothing && tool.approval_mode == :always_require
            approval_needed = true
            break
        end
    end

    if approval_needed
        # Return approval requests for all tool calls in this batch
        for tc in tool_calls
            push!(results, function_approval_request_content(
                something(tc.call_id, ""),
                tc,
            ))
        end
        return results
    end

    for tc in tool_calls
        # Handle approval responses (from a previous approval request round-trip)
        if is_approval_response(tc)
            if tc.approved == true && tc.function_call !== nothing
                # Execute the approved function call
                tc = tc.function_call  # unwrap to the original function call
            else
                # Rejected — return error result
                call_id = tc.function_call !== nothing ? something(tc.function_call.call_id, tc.id, "") : something(tc.id, "")
                push!(results, function_result_content(
                    call_id, nothing;
                    exception="Tool call invocation was rejected by user.",
                ))
                continue
            end
        end

        tool = find_tool(tools, something(tc.name, ""))
        if tool === nothing
            push!(results, function_result_content(
                something(tc.call_id, ""),
                nothing;
                exception = "Tool not found: $(tc.name)",
            ))
            continue
        end

        # Build invocation context for function middleware
        parsed_args = tc.arguments !== nothing ? JSON3.read(tc.arguments, Dict{String, Any}) : Dict{String, Any}()

        func_ctx = FunctionInvocationContext(
            tool = tool,
            arguments = parsed_args,
            call_id = something(tc.call_id, ""),
        )

        function func_handler(fctx::FunctionInvocationContext)
            try
                result = invoke_tool(fctx.tool, fctx.arguments)
                fctx.result = result
                return result
            catch e
                fctx.result = nothing
                rethrow(e)
            end
        end

        try
            result = apply_function_middleware(agent.function_middlewares, func_ctx, func_handler)
            result_str = result isa AbstractString ? result : JSON3.write(result)
            push!(results, function_result_content(
                something(tc.call_id, ""),
                result_str;
                name = tc.name,
            ))
        catch e
            @warn "Tool execution failed" tool=tc.name exception=e
            push!(results, function_result_content(
                something(tc.call_id, ""),
                nothing;
                name = tc.name,
                exception = sprint(showerror, e),
            ))
        end
    end

    return results
end

# ── Typed Agent Run (C#-style RunAsync<T>) ───────────────────────────────────

"""
    run_agent(::Type{T}, agent::Agent, inputs; session=nothing, options=nothing) -> StructuredOutput{T}

Run the agent and parse the response into a typed Julia struct.
Automatically sets `response_format` on the chat options to request
JSON output conforming to `T`'s schema.

# Example
```julia
@kwdef struct MovieReview
    title::String
    rating::Int
    summary::String
end

result = run_agent(MovieReview, agent, "Review The Matrix")
println(result.value.title)   # "The Matrix"
println(result.value.rating)  # 9
```
"""
function run_agent(
    ::Type{T},
    agent::Agent,
    inputs::Union{AgentRunInputs, Nothing} = nothing;
    session::Union{Nothing, AgentSession} = nothing,
    options::Union{Nothing, ChatOptions} = nothing,
)::StructuredOutput{T} where T
    # Build options with response_format for T
    rf = response_format_for(T)
    typed_options = if options !== nothing
        merge_chat_options(options, ChatOptions(response_format=rf))
    else
        ChatOptions(response_format=rf)
    end

    response = run_agent(agent, inputs; session=session, options=typed_options)
    return parse_structured(T, response)
end

# ── Agent Cloning ────────────────────────────────────────────────────────────

"""
    Base.deepcopy(agent::Agent) -> Agent

Create a deep copy of an agent. The chat client is shared (not copied) since
it typically holds connection state.
"""
function Base.deepcopy_internal(agent::Agent, stackdict::IdDict)
    Agent(
        name = agent.name,
        description = agent.description,
        instructions = agent.instructions,
        client = agent.client,  # shared
        tools = copy(agent.tools),
        context_providers = copy(agent.context_providers),
        agent_middlewares = copy(agent.agent_middlewares),
        chat_middlewares = copy(agent.chat_middlewares),
        function_middlewares = copy(agent.function_middlewares),
        options = deepcopy(agent.options),
        max_tool_iterations = agent.max_tool_iterations,
    )
end

"""
    with_instructions(agent::Agent, instructions::String) -> Agent

Create a copy of the agent with different instructions.
"""
function with_instructions(agent::Agent, instructions::String)::Agent
    new_agent = deepcopy(agent)
    new_agent.instructions = instructions
    return new_agent
end

"""
    with_tools(agent::Agent, tools::Vector{FunctionTool}) -> Agent

Create a copy of the agent with different tools.
"""
function with_tools(agent::Agent, tools::Vector{FunctionTool})::Agent
    new_agent = deepcopy(agent)
    new_agent.tools = tools
    return new_agent
end

"""
    with_name(agent::Agent, name::String) -> Agent

Create a copy of the agent with a different name.
"""
function with_name(agent::Agent, name::String)::Agent
    new_agent = deepcopy(agent)
    new_agent.name = name
    return new_agent
end

"""
    with_options(agent::Agent, options::ChatOptions) -> Agent

Create a copy of the agent with different chat options.
"""
function with_options(agent::Agent, options::ChatOptions)::Agent
    new_agent = deepcopy(agent)
    new_agent.options = options
    return new_agent
end

# ── Agent as Tool ────────────────────────────────────────────────────────────

"""
    as_tool(agent::Agent; description=nothing, propagate_session=false) -> FunctionTool

Convert an agent into a FunctionTool so it can be used as a tool by other agents.
When invoked, runs the agent with the input message and returns the text response.

# Arguments
- `agent`: The agent to wrap.
- `description`: Override description (defaults to agent.description or agent.name).
- `propagate_session`: If true, share the calling agent's session with this agent.

# Example
```julia
research_agent = Agent(name="researcher", client=client, instructions="Research topics thoroughly")
main_agent = Agent(name="main", client=client, instructions="You have a research assistant",
                   tools=[as_tool(research_agent)])
```
"""
function as_tool(agent::Agent;
                 description::Union{Nothing, String}=nothing,
                 propagate_session::Bool=false)::FunctionTool
    desc = if description !== nothing
        description
    elseif !isempty(agent.description)
        agent.description
    else
        "Run the $(agent.name) agent with the given input message."
    end
    tool_name = replace(lowercase(agent.name), r"[^a-z0-9_]" => "_")

    func = (input::String) -> begin
        session = propagate_session ? nothing : AgentSession()
        response = run_agent(agent, input; session=session)
        return get_text(response)
    end

    FunctionTool(
        name = tool_name,
        description = desc,
        func = func,
        parameters = Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "input" => Dict{String, Any}(
                    "type" => "string",
                    "description" => "The message to send to the $(agent.name) agent.",
                ),
            ),
            "required" => ["input"],
        ),
    )
end
