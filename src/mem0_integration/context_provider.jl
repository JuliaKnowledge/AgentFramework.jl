function _normalize_role(role)::Symbol
    return role isa Symbol ? role : Symbol(lowercase(string(role)))
end

function _session_metadata_string(session::AgentSession, keys::AbstractVector{String})::Union{Nothing, String}
    for key in keys
        haskey(session.metadata, key) || continue
        value = _nonempty_string(session.metadata[key])
        value !== nothing && return value
    end
    return nothing
end

function _resolved_user_id(provider, session::AgentSession)::Union{Nothing, String}
    provider.user_id !== nothing && return provider.user_id
    return _nonempty_string(session.user_id)
end

function _resolved_agent_id(provider, agent)::Union{Nothing, String}
    provider.agent_id !== nothing && return provider.agent_id
    agent === nothing && return nothing
    hasproperty(agent, :name) || return nothing
    return _nonempty_string(getproperty(agent, :name))
end

function _resolved_application_id(provider, session::AgentSession)::Union{Nothing, String}
    provider.application_id !== nothing && return provider.application_id
    return _session_metadata_string(session, ["application_id", "mem0_application_id", "app_id"])
end

function _mem0_scope(provider, agent, session::AgentSession)
    return (
        user_id = _resolved_user_id(provider, session),
        agent_id = _resolved_agent_id(provider, agent),
        application_id = _resolved_application_id(provider, session),
    )
end

function _mem0_filters(provider, agent, session::AgentSession)::Dict{String, Any}
    scope = _mem0_scope(provider, agent, session)
    filters = Dict{String, Any}()
    scope.user_id !== nothing && (filters["user_id"] = scope.user_id)
    scope.agent_id !== nothing && (filters["agent_id"] = scope.agent_id)
    scope.application_id !== nothing && (filters["app_id"] = scope.application_id)

    isempty(filters) && throw(
        ArgumentError("At least one of `agent_id`, `user_id`, or `application_id` is required for Mem0."),
    )

    return filters
end

function _mem0_search_path(client::Mem0Client)::String
    return client.deployment == MEM0_PLATFORM ? "v2/memories/search" : "search"
end

function _mem0_add_path(client::Mem0Client)::String
    return client.deployment == MEM0_PLATFORM ? "v1/memories" : "memories"
end

function _mem0_search_body(
    client::Mem0Client,
    query::AbstractString,
    filters::Dict{String, Any};
    top_k::Int,
    rerank::Union{Nothing, Bool},
)::Dict{String, Any}
    if client.deployment == MEM0_PLATFORM
        body = Dict{String, Any}(
            "query" => String(query),
            "filters" => filters,
            "top_k" => top_k,
        )
    else
        body = Dict{String, Any}("query" => String(query), "top_k" => top_k)
        merge!(body, filters)
    end

    rerank !== nothing && (body["rerank"] = rerank)
    return body
end

function _mem0_extract_search_results(payload)::Vector{Any}
    if payload === nothing
        return Any[]
    elseif payload isa AbstractVector
        return Any[item for item in payload]
    elseif payload isa AbstractDict && haskey(payload, "results")
        results = payload["results"]
        results isa AbstractVector || return Any[payload]
        return Any[item for item in results]
    end

    return Any[payload]
end

function _mem0_memory_text(memory)::Union{Nothing, String}
    if memory isa AbstractDict
        for key in ("memory", "content", "text")
            haskey(memory, key) || continue
            value = _nonempty_string(memory[key])
            value !== nothing && return value
        end
        return nothing
    end

    return _nonempty_string(memory)
end

function _format_mem0_context(memories::Vector{Any}, context_prompt::AbstractString)::Union{Nothing, String}
    memory_lines = String[]
    for memory in memories
        text = _mem0_memory_text(memory)
        text === nothing && continue
        push!(memory_lines, text)
    end

    isempty(memory_lines) && return nothing
    return String(context_prompt) * "\n" * join(memory_lines, "\n")
end

function _mem0_collect_messages(provider, ctx::SessionContext)::Vector{Dict{String, String}}
    allowed_roles = Set(provider.store_roles)
    payload = Dict{String, String}[]

    function append_message(message::Message)
        text = strip(message.text)
        isempty(text) && return nothing

        role = _normalize_role(message.role)
        role in allowed_roles || return nothing

        push!(payload, Dict{String, String}(
            "role" => String(role),
            "content" => text,
        ))
        return nothing
    end

    for message in ctx.input_messages
        append_message(message)
    end

    if ctx.response !== nothing && hasproperty(ctx.response, :messages)
        for message in ctx.response.messages
            append_message(message)
        end
    end

    return payload
end

function _mem0_store_body(
    provider,
    agent,
    session::AgentSession,
    messages::Vector{Dict{String, String}},
)::Dict{String, Any}
    scope = _mem0_scope(provider, agent, session)
    body = Dict{String, Any}("messages" => messages)

    scope.user_id !== nothing && (body["user_id"] = scope.user_id)
    scope.agent_id !== nothing && (body["agent_id"] = scope.agent_id)

    if scope.application_id !== nothing
        body["metadata"] = Dict{String, Any}("application_id" => scope.application_id)
    end

    return body
end

mutable struct Mem0ContextProvider <: BaseContextProvider
    client::Mem0Client
    source_id::String
    application_id::Union{Nothing, String}
    agent_id::Union{Nothing, String}
    user_id::Union{Nothing, String}
    context_prompt::String
    top_k::Int
    rerank::Union{Nothing, Bool}
    store_roles::Vector{Symbol}
end

function Mem0ContextProvider(;
    client::Union{Nothing, Mem0Client} = nothing,
    api_key::Union{Nothing, AbstractString} = nothing,
    base_url::Union{Nothing, AbstractString} = nothing,
    deployment::Symbol = MEM0_PLATFORM,
    request_runner::Function = _default_mem0_request,
    source_id::AbstractString = DEFAULT_MEM0_SOURCE_ID,
    application_id::Union{Nothing, AbstractString} = nothing,
    agent_id::Union{Nothing, AbstractString} = nothing,
    user_id::Union{Nothing, AbstractString} = nothing,
    context_prompt::Union{Nothing, AbstractString} = nothing,
    top_k::Int = 5,
    rerank::Union{Nothing, Bool} = nothing,
    store_roles::AbstractVector = [ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM],
)
    if client !== nothing
        has_connection_override = (
            api_key !== nothing ||
            base_url !== nothing ||
            deployment != MEM0_PLATFORM ||
            request_runner !== _default_mem0_request
        )
        has_connection_override && throw(
            ArgumentError("Provide either `client` or direct Mem0 connection settings, not both."),
        )
    else
        client = Mem0Client(;
            api_key = api_key,
            base_url = base_url,
            deployment = deployment,
            request_runner = request_runner,
        )
    end

    normalized_source_id = _nonempty_string(source_id)
    normalized_source_id === nothing && throw(ArgumentError("Mem0 source_id cannot be empty."))
    top_k > 0 || throw(ArgumentError("Mem0 top_k must be positive."))

    normalized_roles = Symbol[]
    for role in store_roles
        push!(normalized_roles, _normalize_role(role))
    end
    normalized_roles = unique(normalized_roles)
    isempty(normalized_roles) && throw(ArgumentError("Mem0 store_roles cannot be empty."))

    return Mem0ContextProvider(
        client,
        normalized_source_id,
        _nonempty_string(application_id),
        _nonempty_string(agent_id),
        _nonempty_string(user_id),
        something(_nonempty_string(context_prompt), DEFAULT_MEM0_CONTEXT_PROMPT),
        top_k,
        rerank,
        normalized_roles,
    )
end

function before_run!(
    provider::Mem0ContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    query_messages = [message for message in ctx.input_messages if !isempty(strip(message.text))]
    isempty(query_messages) && return nothing

    query = join((strip(message.text) for message in query_messages), "\n")
    filters = _mem0_filters(provider, agent, session)
    response = _mem0_request(
        provider.client,
        "POST",
        _mem0_search_path(provider.client);
        body = _mem0_search_body(
            provider.client,
            query,
            filters;
            top_k = provider.top_k,
            rerank = provider.rerank,
        ),
    )
    results = _mem0_extract_search_results(_read_mem0_payload(response))

    state["last_query"] = query
    state["last_result_count"] = length(results)

    context_text = _format_mem0_context(results, provider.context_prompt)
    context_text === nothing && return nothing

    extend_messages!(ctx, provider, [Message(ROLE_USER, context_text)])
    return nothing
end

function after_run!(
    provider::Mem0ContextProvider,
    agent,
    session::AgentSession,
    ctx::SessionContext,
    state::Dict{String, Any},
)
    _mem0_filters(provider, agent, session)

    messages = _mem0_collect_messages(provider, ctx)
    isempty(messages) && return nothing

    _mem0_request(
        provider.client,
        "POST",
        _mem0_add_path(provider.client);
        body = _mem0_store_body(provider, agent, session, messages),
    )
    state["stored_count"] = get(state, "stored_count", 0) + length(messages)
    return nothing
end

function Base.show(io::IO, provider::Mem0ContextProvider)
    print(io, "Mem0ContextProvider(source_id=\"", provider.source_id, "\", deployment=:", provider.client.deployment, ")")
end
