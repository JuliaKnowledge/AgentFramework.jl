# Handoff orchestration — decentralised multi-agent routing.
#
# Port of `agent_framework_orchestrations._handoff` from the upstream Python
# AgentFramework. Agents in this orchestration decide who answers next by
# invoking declaration-only tools named `handoff_to_<target_id>`. The first
# such tool call in an agent's response wins; control transfers immediately
# without invoking the agent's normal tool-execution loop on the handoff tool.
#
# Notes vs. the Python reference:
# - We do not yet integrate with the `RequestInfo` (human-in-the-loop) pattern;
#   the autonomous-mode loop covers that gap for now.
# - We do not disable provider-side conversation persistence — AF.jl's chat
#   clients are stateless OpenAI-style by default. Providers that maintain
#   server-side threads should not be combined with handoff orchestration
#   without thread reset (caller's responsibility).

# ── Public types ────────────────────────────────────────────────────────────

"""
    HandoffSentEvent(source, target)

Event payload describing a single agent-to-agent handoff transfer. Emitted
through the workflow event stream when one participant transfers control to
another via a `handoff_to_<target>` tool call.
"""
struct HandoffSentEvent
    source::String
    target::String
end

"""
    HandoffConfiguration(target, description=nothing)

Routing rule allowing the source agent (declared in `HandoffBuilder.handoff_config`'s
key) to hand off to `target`. `description` is shown to the LLM as the
handoff tool's description; if `nothing`, the target agent's own
description is used.
"""
Base.@kwdef struct HandoffConfiguration
    target::String
    description::Union{Nothing, String} = nothing
end

Base.:(==)(a::HandoffConfiguration, b::HandoffConfiguration) = a.target == b.target
Base.hash(c::HandoffConfiguration, h::UInt) = hash(c.target, h)

"""
    get_handoff_tool_name(target_id::AbstractString) -> String

Standardised name for the synthetic tool an agent calls to transfer control
to `target_id`. Matches the Python reference (`handoff_to_<target_id>`).
"""
get_handoff_tool_name(target_id::AbstractString)::String = "handoff_to_" * String(target_id)

const _HANDOFF_TOOL_PREFIX = "handoff_to_"
const _HANDOFF_INPUT_ID = "__handoff_input__"
const _HANDOFF_OUTPUT_ID = "__handoff_end__"
const _HANDOFF_AUTONOMOUS_KEY = "__handoff_autonomous_turns__"
const _AUTONOMOUS_MODE_DEFAULT_PROMPT = "User did not respond. Continue assisting autonomously."
const _DEFAULT_AUTONOMOUS_TURN_LIMIT = 50

# ── Builder ────────────────────────────────────────────────────────────────

"""
    HandoffBuilder(; participants=nothing, name="HandoffWorkflow", ...)

Fluent builder for conversational handoff orchestrations. Build a [`Workflow`](@ref)
where multiple agents pass control among themselves by emitting
`handoff_to_<target>` tool calls.

Builder methods (chainable):

- [`participants`](@ref) — register agents (also accepts `participants=` kwarg)
- [`add_handoff`](@ref) — declare allowed source → targets routing
- [`with_start_agent`](@ref) — pick the agent that takes the first turn
- [`with_autonomous_mode`](@ref) — keep agents looping when no handoff occurs
- [`with_termination`](@ref) — supply a custom stop condition over the conversation
- [`with_checkpointing`](@ref) — enable workflow state persistence
- [`build`](@ref) — produce the final `Workflow`

If no `add_handoff` calls are made before `build()`, a complete mesh
(every agent can hand off to every other) is created.
"""
mutable struct HandoffBuilder
    participants_dict::Dict{String, Agent}
    participants_order::Vector{String}
    start_id::Union{Nothing, String}
    handoff_config::Dict{String, Vector{HandoffConfiguration}}
    autonomous_mode::Bool
    autonomous_prompts::Dict{String, String}
    autonomous_turn_limits::Dict{String, Int}
    autonomous_enabled_agents::Vector{String}
    termination_condition::Union{Nothing, Function}
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage}
    name::String
    description::Union{Nothing, String}
end

function HandoffBuilder(;
    participants::Union{Nothing, AbstractVector} = nothing,
    name::AbstractString = "HandoffWorkflow",
    description::Union{Nothing, AbstractString} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    termination_condition::Union{Nothing, Function} = nothing,
)
    builder = HandoffBuilder(
        Dict{String, Agent}(),
        String[],
        nothing,
        Dict{String, Vector{HandoffConfiguration}}(),
        false,
        Dict{String, String}(),
        Dict{String, Int}(),
        String[],
        termination_condition,
        checkpoint_storage,
        String(name),
        description === nothing ? nothing : String(description),
    )
    if participants !== nothing
        participants!(builder, participants)
    end
    return builder
end

"""
    participants!(builder, agents) -> HandoffBuilder

Register the agents that will participate. Each must be a concrete `Agent`
with a unique identifier (`agent.name` if non-empty, otherwise `agent.id`).
Raises if called twice or if duplicates / non-`Agent` values are passed.
"""
function participants!(builder::HandoffBuilder, agents::AbstractVector)::HandoffBuilder
    isempty(builder.participants_dict) || throw(ArgumentError("participants have already been assigned"))
    isempty(agents) && throw(ArgumentError("participants cannot be empty"))
    for participant in agents
        participant isa Agent || throw(ArgumentError(
            "Handoff participants must be `Agent` instances; got $(typeof(participant)). " *
            "Handoff requires cloning + tool injection, which is only available on `Agent`."
        ))
        pid = _participant_id(participant)
        haskey(builder.participants_dict, pid) && throw(ArgumentError("Duplicate participant identifier '$pid'"))
        builder.participants_dict[pid] = participant
        push!(builder.participants_order, pid)
    end
    return builder
end

"""
    add_handoff(builder, source, targets; description=nothing) -> HandoffBuilder

Declare that `source` may hand off to any agent in `targets`. May be called
multiple times for the same source — entries merge. If never called for any
source before [`build`](@ref), a full mesh topology is generated automatically.

`source` and each entry in `targets` must already be registered via
[`participants!`](@ref).
"""
function add_handoff(builder::HandoffBuilder, source, targets;
                     description::Union{Nothing, AbstractString} = nothing)::HandoffBuilder
    isempty(builder.participants_dict) &&
        throw(ArgumentError("Call participants!(...) before add_handoff(...)"))

    source_id = _resolve_handoff_id(builder, source)
    targets_iter = targets isa AbstractVector ? targets : (targets,)
    isempty(targets_iter) && throw(ArgumentError("targets cannot be empty"))

    desc = description === nothing ? nothing : String(description)
    for tgt in targets_iter
        target_id = _resolve_handoff_id(builder, tgt)
        target_id == source_id && throw(ArgumentError(
            "Self-handoff not allowed (source == target == '$source_id')"))
        bucket = get!(builder.handoff_config, source_id, HandoffConfiguration[])
        cfg = HandoffConfiguration(target = target_id, description = desc)
        existing = findfirst(c -> c.target == target_id, bucket)
        if existing === nothing
            push!(bucket, cfg)
        else
            bucket[existing] = cfg  # overwrite (e.g. updated description)
        end
    end
    return builder
end

"""
    with_start_agent(builder, agent) -> HandoffBuilder

Set the agent that takes the first turn. Defaults to the first registered
participant if not called.
"""
function with_start_agent(builder::HandoffBuilder, agent)::HandoffBuilder
    isempty(builder.participants_dict) &&
        throw(ArgumentError("Call participants!(...) before with_start_agent(...)"))
    builder.start_id = _resolve_handoff_id(builder, agent)
    return builder
end

"""
    with_autonomous_mode(builder; agents=nothing, prompts=nothing, turn_limits=nothing) -> HandoffBuilder

Enable autonomous mode. When an agent finishes its turn without invoking a
handoff tool and autonomous mode is enabled for it, the workflow appends a
"continue" prompt and re-runs the same agent — up to its turn limit. When
disabled (default) or the limit is reached, the conversation is yielded as
the workflow output.

- `agents`: limit autonomous mode to these participants (default: all).
- `prompts`: per-agent custom continue prompts.
- `turn_limits`: per-agent max consecutive autonomous turns (default $_DEFAULT_AUTONOMOUS_TURN_LIMIT).
"""
function with_autonomous_mode(builder::HandoffBuilder;
        agents::Union{Nothing, AbstractVector} = nothing,
        prompts::Union{Nothing, AbstractDict} = nothing,
        turn_limits::Union{Nothing, AbstractDict} = nothing,
    )::HandoffBuilder
    builder.autonomous_mode = true
    builder.autonomous_prompts = prompts === nothing ? Dict{String, String}() :
        Dict{String, String}(String(_resolve_handoff_id(builder, k)) => String(v) for (k, v) in prompts)
    builder.autonomous_turn_limits = turn_limits === nothing ? Dict{String, Int}() :
        Dict{String, Int}(String(_resolve_handoff_id(builder, k)) => Int(v) for (k, v) in turn_limits)
    builder.autonomous_enabled_agents = agents === nothing ? String[] :
        String[_resolve_handoff_id(builder, a) for a in agents]
    return builder
end

"""
    with_termination(builder, predicate::Function) -> HandoffBuilder

Set a `predicate(conversation::Vector{Message}) -> Bool` that, when returning
`true`, stops the workflow and yields the conversation as its output.
Checked both before and after each agent turn.
"""
function with_termination(builder::HandoffBuilder, predicate::Function)::HandoffBuilder
    builder.termination_condition = predicate
    return builder
end

"""
    with_checkpointing(builder, storage) -> HandoffBuilder

Enable workflow state persistence using the given checkpoint storage backend.
"""
function with_checkpointing(builder::HandoffBuilder, storage::AbstractCheckpointStorage)::HandoffBuilder
    builder.checkpoint_storage = storage
    return builder
end

# ── Build ──────────────────────────────────────────────────────────────────

function _resolve_handoff_id(builder::HandoffBuilder, agent)::String
    pid = agent isa AbstractString ? String(agent) : _participant_id(agent)
    haskey(builder.participants_dict, pid) ||
        throw(ArgumentError("Agent '$pid' is not in the participants list"))
    return pid
end

function _validate_no_handoff_tool_collisions(participant::Agent)
    for tool in participant.tools
        nm = nothing
        if tool isa FunctionTool
            nm = tool.name
        elseif hasproperty(tool, :name)
            nm = getproperty(tool, :name)
        end
        nm === nothing && continue
        startswith(String(nm), _HANDOFF_TOOL_PREFIX) && throw(ArgumentError(
            "Participant '$(participant.name)' already has a tool named '$nm'. " *
            "Tool names with prefix '$_HANDOFF_TOOL_PREFIX' are reserved for handoff routing."
        ))
    end
end

function _build_handoff_tool(target_id::String, target_agent::Agent,
                             override_description::Union{Nothing, String})::FunctionTool
    desc = override_description !== nothing ? override_description :
        (isempty(target_agent.description) ? "Hand off the conversation to $target_id." : target_agent.description)
    FunctionTool(
        name = get_handoff_tool_name(target_id),
        description = String(desc),
        func = nothing,  # declaration-only; intercepted by custom run loop
        parameters = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}()),
    )
end

function _autonomous_enabled(builder::HandoffBuilder, participant_id::String)::Bool
    builder.autonomous_mode || return false
    isempty(builder.autonomous_enabled_agents) && return true
    return participant_id in builder.autonomous_enabled_agents
end

function _autonomous_prompt(builder::HandoffBuilder, participant_id::String)::String
    get(builder.autonomous_prompts, participant_id, _AUTONOMOUS_MODE_DEFAULT_PROMPT)
end

function _autonomous_turn_limit(builder::HandoffBuilder, participant_id::String)::Int
    get(builder.autonomous_turn_limits, participant_id, _DEFAULT_AUTONOMOUS_TURN_LIMIT)
end

# Per-participant session keys (re-using orchestration session helpers)
_handoff_session_key(participant_id::String)::String = "handoff:" * participant_id

# Detect a handoff request by scanning the messages produced by an LLM round.
# Returns the FIRST matching target id, or `nothing`.
function _detect_handoff(messages::Vector{Message}, allowed_targets::Set{String})::Union{Nothing, String}
    for msg in messages
        for c in msg.contents
            if is_function_call(c)
                nm = c.name
                if nm isa AbstractString && startswith(nm, _HANDOFF_TOOL_PREFIX)
                    target = String(SubString(nm, length(_HANDOFF_TOOL_PREFIX) + 1))
                    target in allowed_targets && return target
                end
            end
        end
    end
    return nothing
end

# Custom run loop that mirrors `_execute_agent_run` but stops the moment a
# handoff tool call appears in an LLM response, before that tool would be
# executed. Returns (assistant messages emitted this run, optional handoff target).
function _run_handoff_aware!(
    agent::Agent,
    conversation::Vector{Message},
    session::AgentSession,
    allowed_targets::Set{String},
)::Tuple{Vector{Message}, Union{Nothing, String}}

    ctx = SessionContext(
        session_id = session.id,
        service_session_id = session.thread_id,
        input_messages = conversation,
        options = Dict{String, Any}(),
    )

    _run_before_context_providers!(agent, session, ctx)

    all_messages = _build_message_list(agent, ctx)
    chat_options, all_tools = _prepare_chat_options(agent, ctx, nothing)

    emitted = Message[]
    iteration = 0
    handoff_target::Union{Nothing, String} = nothing

    while iteration < agent.max_tool_iterations
        iteration += 1

        chat_response = get_response(agent.client, all_messages, chat_options)
        _sync_service_session!(session, ctx, chat_response.conversation_id)
        chat_options = _refresh_chat_options_thread_id(chat_options, ctx)

        # First scan for a handoff tool invocation — wins over normal tools.
        handoff_target = _detect_handoff(chat_response.messages, allowed_targets)
        if handoff_target !== nothing
            append!(emitted, chat_response.messages)
            break
        end

        # Otherwise reuse the standard tool loop semantics: collect tool calls,
        # run them, append results, continue.
        tool_calls = Content[]
        for msg in chat_response.messages
            for c in msg.contents
                is_function_call(c) && push!(tool_calls, c)
            end
        end

        append!(emitted, chat_response.messages)

        if isempty(tool_calls)
            break
        end

        append!(all_messages, chat_response.messages)
        tool_results = _execute_tool_calls(agent, all_tools, tool_calls)
        push!(emitted, Message(role = :tool, contents = tool_results))
        push!(all_messages, Message(role = :tool, contents = tool_results))
    end

    if iteration >= agent.max_tool_iterations && handoff_target === nothing
        @warn "Handoff agent reached max tool iterations" agent=_participant_id(agent) iter=iteration
    end

    # Best-effort after-hooks (response metadata is not all carried through here).
    ctx.response = AgentResponse(messages = emitted)
    _run_after_context_providers!(agent, session, ctx)

    return emitted, handoff_target
end

# ExecutorSpec for one handoff participant. Owns the cloned agent (with
# handoff tools pre-injected at build time) and routes via send_message.
function _handoff_participant_executor(
    builder::HandoffBuilder,
    participant_id::String,
    cloned_agent::Agent,
    handoff_targets::Vector{String},
    output_id::String,
)::ExecutorSpec
    allowed = Set{String}(handoff_targets)
    session_key = _handoff_session_key(participant_id)

    handler = (conversation, ctx) -> begin
        conv = conversation isa Vector{Message} ? deepcopy(conversation) :
            throw(WorkflowError("Handoff participant '$participant_id' expected Vector{Message}, got $(typeof(conversation))"))

        # Pre-run termination check.
        if builder.termination_condition !== nothing && builder.termination_condition(conv)
            send_message(ctx, conv; target_id = output_id)
            return nothing
        end

        session = _load_stateful_session!(ctx._state, session_key,
            _default_participant_session(cloned_agent, participant_id))

        emitted, target = _run_handoff_aware!(cloned_agent, conv, session, allowed)
        _save_stateful_session!(ctx._state, session_key, session)

        full_conv = vcat(conv, deepcopy(emitted))

        # Post-run termination check.
        if builder.termination_condition !== nothing && builder.termination_condition(full_conv)
            send_message(ctx, full_conv; target_id = output_id)
            return nothing
        end

        if target !== nothing
            # Reset autonomous turn counter for this agent on successful handoff.
            counters = get!(ctx._state, _HANDOFF_AUTONOMOUS_KEY, Dict{String, Int}())
            counters[participant_id] = 0
            yield_output(ctx, HandoffSentEvent(participant_id, target))
            send_message(ctx, full_conv; target_id = target)
            return nothing
        end

        # No handoff. Decide between autonomous loop and final output.
        if _autonomous_enabled(builder, participant_id)
            counters = get!(ctx._state, _HANDOFF_AUTONOMOUS_KEY, Dict{String, Int}())
            count = get(counters, participant_id, 0) + 1
            limit = _autonomous_turn_limit(builder, participant_id)
            if count <= limit
                counters[participant_id] = count
                continue_msg = Message(role = :user,
                    contents = [text_content(_autonomous_prompt(builder, participant_id))])
                push!(full_conv, continue_msg)
                send_message(ctx, full_conv; target_id = participant_id)
                return nothing
            else
                # Limit reached — reset and emit output.
                counters[participant_id] = 0
            end
        end

        send_message(ctx, full_conv; target_id = output_id)
        return nothing
    end

    return ExecutorSpec(
        id = participant_id,
        description = "Handoff participant: $participant_id",
        input_types = DataType[Vector{Message}],
        output_types = DataType[Vector{Message}, HandoffSentEvent],
        yield_types = DataType[HandoffSentEvent],
        handler = handler,
    )
end

function _handoff_output_executor()::ExecutorSpec
    ExecutorSpec(
        id = _HANDOFF_OUTPUT_ID,
        description = "Handoff orchestration output",
        input_types = DataType[Vector{Message}],
        output_types = DataType[Vector{Message}],
        yield_types = DataType[Vector{Message}],
        handler = (conversation, ctx) -> begin
            yield_output(ctx, deepcopy(conversation))
            return nothing
        end,
    )
end

"""
    build(builder::HandoffBuilder; validate_types=true) -> Workflow

Compile the builder into a runnable [`Workflow`](@ref). If no `add_handoff`
calls were made, every participant is wired to every other participant
(complete mesh).
"""
function build(builder::HandoffBuilder; validate_types::Bool = true)::Workflow
    isempty(builder.participants_dict) &&
        throw(ArgumentError("Call participants!(...) before build(...)"))

    # Default mesh topology.
    if isempty(builder.handoff_config)
        for src in builder.participants_order
            others = filter(!=(src), builder.participants_order)
            if !isempty(others)
                add_handoff(builder, src, others)
            end
        end
    end

    start_id = builder.start_id !== nothing ? builder.start_id : first(builder.participants_order)

    # Validate no participant already has a tool named handoff_to_*.
    for participant in values(builder.participants_dict)
        _validate_no_handoff_tool_collisions(participant)
    end

    # Clone each participant once and inject its handoff tools.
    cloned = Dict{String, Agent}()
    targets_for = Dict{String, Vector{String}}()
    for participant_id in builder.participants_order
        original = builder.participants_dict[participant_id]
        configs = get(builder.handoff_config, participant_id, HandoffConfiguration[])
        targets = String[c.target for c in configs]
        new_tools = collect(original.tools)
        for cfg in configs
            target_agent = builder.participants_dict[cfg.target]
            push!(new_tools, _build_handoff_tool(cfg.target, target_agent, cfg.description))
        end
        clone = deepcopy(original)
        clone.tools = new_tools
        cloned[participant_id] = clone
        targets_for[participant_id] = targets
    end

    input = _conversation_input_executor(_HANDOFF_INPUT_ID, "Normalize input for handoff orchestration")
    output = _handoff_output_executor()

    wb = WorkflowBuilder(
        name = builder.name,
        start = input,
        checkpoint_storage = builder.checkpoint_storage,
    )
    add_executor(wb, output)
    add_output(wb, output.id)

    # Add participant executors.
    for participant_id in builder.participants_order
        spec = _handoff_participant_executor(builder, participant_id,
            cloned[participant_id], targets_for[participant_id], output.id)
        add_executor(wb, spec)
    end

    # Wire input → start participant.
    add_edge(wb, input.id, start_id)

    # Wire each participant → each of its handoff targets, plus → output.
    for participant_id in builder.participants_order
        for tgt in targets_for[participant_id]
            add_edge(wb, participant_id, tgt)
        end
        # Self-edge for autonomous mode.
        if _autonomous_enabled(builder, participant_id)
            add_edge(wb, participant_id, participant_id)
        end
        add_edge(wb, participant_id, output.id)
    end

    return build(wb; validate_types = validate_types)
end
