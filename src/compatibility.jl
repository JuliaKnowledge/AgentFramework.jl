# Compatibility helpers for Semantic Kernel and AutoGen migrations.

function _resolve_compat_client(
    client::Union{Nothing, AbstractChatClient},
    model_client::Union{Nothing, AbstractChatClient},
)::AbstractChatClient
    if client !== nothing && model_client !== nothing && client !== model_client
        throw(ArgumentError("Provide either `client` or `model_client`, not both."))
    end

    resolved = client !== nothing ? client : model_client
    resolved === nothing && throw(ArgumentError("Provide `client` or `model_client`."))
    return resolved
end

function _resolve_compat_instructions(
    instructions::Union{Nothing, AbstractString},
    system_message::Union{Nothing, AbstractString},
)::String
    if instructions !== nothing && system_message !== nothing &&
       String(instructions) != String(system_message)
        throw(ArgumentError("Provide either `instructions` or `system_message`, not both."))
    end

    if instructions !== nothing
        return String(instructions)
    elseif system_message !== nothing
        return String(system_message)
    end
    return ""
end

function _normalize_compatibility_tools(
    tools::AbstractVector,
    handoffs::AbstractVector,
)::Vector{FunctionTool}
    mixed = Any[]

    for tool in tools
        (tool isa FunctionTool || tool isa HandoffTool) ||
            throw(ArgumentError("Compatibility helpers only accept FunctionTool or HandoffTool items in `tools`; got $(typeof(tool))."))
        push!(mixed, tool)
    end

    for handoff in handoffs
        handoff isa HandoffTool ||
            throw(ArgumentError("Compatibility helper `handoffs` must contain only HandoffTool values; got $(typeof(handoff))."))
        push!(mixed, handoff)
    end

    function_tools, _ = normalize_agent_tools(mixed)
    return function_tools
end

function _compatibility_agent(;
    name::String,
    description::String = "",
    instructions::Union{Nothing, AbstractString} = nothing,
    system_message::Union{Nothing, AbstractString} = nothing,
    client::Union{Nothing, AbstractChatClient} = nothing,
    model_client::Union{Nothing, AbstractChatClient} = nothing,
    tools::AbstractVector = FunctionTool[],
    handoffs::AbstractVector = HandoffTool[],
    context_providers::AbstractVector = Any[],
    agent_middlewares::AbstractVector = Any[],
    chat_middlewares::AbstractVector = Any[],
    function_middlewares::AbstractVector = Any[],
    options::ChatOptions = ChatOptions(),
    max_tool_iterations::Int = DEFAULT_MAX_TOOL_ITERATIONS,
)::Agent
    Agent(
        name = name,
        description = description,
        instructions = _resolve_compat_instructions(instructions, system_message),
        client = _resolve_compat_client(client, model_client),
        tools = _normalize_compatibility_tools(tools, handoffs),
        context_providers = Any[context_providers...],
        agent_middlewares = collect(agent_middlewares),
        chat_middlewares = collect(chat_middlewares),
        function_middlewares = collect(function_middlewares),
        options = options,
        max_tool_iterations = max_tool_iterations,
    )
end

function _resolve_turn_limit(
    primary::Union{Nothing, Int},
    alias::Union{Nothing, Int},
    primary_name::String,
    alias_name::String,
)::Union{Nothing, Int}
    if primary !== nothing && alias !== nothing && primary != alias
        throw(ArgumentError("Provide either `$(primary_name)` or `$(alias_name)`, not both."))
    end
    return primary !== nothing ? primary : alias
end

function _resolve_selector_function(
    selection_func::Union{Nothing, Function},
    selector_func::Union{Nothing, Function},
)::Union{Nothing, Function}
    if selection_func !== nothing && selector_func !== nothing && selection_func !== selector_func
        throw(ArgumentError("Provide either `selection_func` or `selector_func`, not both."))
    end
    return selection_func !== nothing ? selection_func : selector_func
end

function _resolve_selector_agent(
    orchestrator_agent::Union{Nothing, Agent, AgentExecutor},
    selector_agent::Union{Nothing, Agent, AgentExecutor},
)::Union{Nothing, Agent, AgentExecutor}
    if orchestrator_agent !== nothing && selector_agent !== nothing && orchestrator_agent !== selector_agent
        throw(ArgumentError("Provide either `orchestrator_agent` or `selector_agent`, not both."))
    end
    return orchestrator_agent !== nothing ? orchestrator_agent : selector_agent
end

"""
    ChatCompletionAgent(; kwargs...) -> Agent

Compatibility constructor for Semantic Kernel migrations. This forwards into the
standard `Agent` type while accepting the common `model_client` and
`system_message` aliases.
"""
function ChatCompletionAgent(;
    name::String = "ChatCompletionAgent",
    kwargs...
)::Agent
    _compatibility_agent(; name = name, kwargs...)
end

"""
    AssistantAgent(; kwargs...) -> Agent

Compatibility constructor for AutoGen migrations. This forwards into the
standard `Agent` type while accepting the common `model_client` and
`system_message` aliases.
"""
function AssistantAgent(;
    name::String = "AssistantAgent",
    kwargs...
)::Agent
    _compatibility_agent(; name = name, kwargs...)
end

"""
    RoundRobinGroupChat(; participants, max_rounds=nothing, max_turns=nothing, ...) -> Workflow

Compatibility helper for AutoGen-style round-robin group chats. Builds a
`GroupChatBuilder` with the default round-robin participant selection policy.
"""
function RoundRobinGroupChat(;
    participants::AbstractVector,
    max_rounds::Union{Nothing, Int} = nothing,
    max_turns::Union{Nothing, Int} = nothing,
    termination_condition::Union{Nothing, Function} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    intermediate_outputs::Bool = false,
    name::String = "RoundRobinGroupChat",
    validate_types::Bool = true,
)::Workflow
    workflow = GroupChatBuilder(
        participants = participants,
        max_rounds = _resolve_turn_limit(max_rounds, max_turns, "max_rounds", "max_turns"),
        termination_condition = termination_condition,
        checkpoint_storage = checkpoint_storage,
        intermediate_outputs = intermediate_outputs,
        name = name,
    )
    return build(workflow; validate_types = validate_types)
end

"""
    SelectorGroupChat(; participants, selection_func=nothing, selector_func=nothing, ...) -> Workflow

Compatibility helper for selector-managed group chats. Supply either a Julia
selection function or an orchestrator agent via `orchestrator_agent` or
`selector_agent`.
"""
function SelectorGroupChat(;
    participants::AbstractVector,
    selection_func::Union{Nothing, Function} = nothing,
    selector_func::Union{Nothing, Function} = nothing,
    orchestrator_agent::Union{Nothing, Agent, AgentExecutor} = nothing,
    selector_agent::Union{Nothing, Agent, AgentExecutor} = nothing,
    max_rounds::Union{Nothing, Int} = nothing,
    max_turns::Union{Nothing, Int} = nothing,
    termination_condition::Union{Nothing, Function} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    intermediate_outputs::Bool = false,
    name::String = "SelectorGroupChat",
    validate_types::Bool = true,
)::Workflow
    resolved_selection = _resolve_selector_function(selection_func, selector_func)
    resolved_orchestrator = _resolve_selector_agent(orchestrator_agent, selector_agent)
    resolved_selection === nothing && resolved_orchestrator === nothing &&
        throw(ArgumentError("Provide `selection_func`, `selector_func`, `orchestrator_agent`, or `selector_agent`."))

    workflow = GroupChatBuilder(
        participants = participants,
        selection_func = resolved_selection,
        orchestrator_agent = resolved_orchestrator,
        max_rounds = _resolve_turn_limit(max_rounds, max_turns, "max_rounds", "max_turns"),
        termination_condition = termination_condition,
        checkpoint_storage = checkpoint_storage,
        intermediate_outputs = intermediate_outputs,
        name = name,
    )
    return build(workflow; validate_types = validate_types)
end

"""
    MagenticOneGroupChat(; participants, ...) -> Workflow

Compatibility helper for Magentic-One-style orchestrations. Builds and returns a
`MagenticBuilder` workflow directly.
"""
function MagenticOneGroupChat(;
    participants::AbstractVector,
    manager::AbstractMagenticManager = StandardMagenticManager(),
    enable_plan_review::Bool = false,
    max_stall_count::Int = 3,
    max_round_count::Union{Nothing, Int} = nothing,
    max_turns::Union{Nothing, Int} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    intermediate_outputs::Bool = false,
    name::String = "MagenticOneGroupChat",
    validate_types::Bool = true,
)::Workflow
    workflow = MagenticBuilder(
        participants = participants,
        manager = manager,
        enable_plan_review = enable_plan_review,
        max_stall_count = max_stall_count,
        max_round_count = _resolve_turn_limit(max_round_count, max_turns, "max_round_count", "max_turns"),
        checkpoint_storage = checkpoint_storage,
        intermediate_outputs = intermediate_outputs,
        name = name,
    )
    return build(workflow; validate_types = validate_types)
end
