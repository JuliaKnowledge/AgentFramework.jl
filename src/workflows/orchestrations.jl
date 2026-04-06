# High-level orchestration builders layered on the generic workflow engine.

Base.@kwdef struct ConcurrentParticipantResult
    participant_id::String
    conversation::Vector{Message}
    response_messages::Vector{Message} = Message[]
end

Base.@kwdef struct GroupChatState
    conversation::Vector{Message}
    participant_ids::Vector{String}
    round::Int = 0
    last_speaker::Union{Nothing, String} = nothing
    metadata::Dict{String, Any} = Dict{String, Any}()
end

Base.@kwdef struct GroupChatTurnResult
    participant_id::String
    conversation::Vector{Message}
    response_messages::Vector{Message} = Message[]
end

Base.@kwdef struct MagenticTaskLedger
    facts::Vector{String} = String[]
    plan::Vector{String} = String[]
    current_step::Int = 1
end

Base.@kwdef struct MagenticProgressLedgerItem
    participant_id::String
    summary::String
    round::Int
    progressed::Bool = true
end

Base.@kwdef struct MagenticProgressLedger
    items::Vector{MagenticProgressLedgerItem} = MagenticProgressLedgerItem[]
    stall_count::Int = 0
end

Base.@kwdef struct MagenticContext
    conversation::Vector{Message}
    participant_ids::Vector{String}
    round::Int = 0
    last_speaker::Union{Nothing, String} = nothing
    task_ledger::MagenticTaskLedger = MagenticTaskLedger()
    progress_ledger::MagenticProgressLedger = MagenticProgressLedger()
end

Base.@kwdef struct MagenticPlanReviewRequest
    task_ledger::MagenticTaskLedger
    context::MagenticContext
end

Base.@kwdef struct MagenticPlanReviewResponse
    approved::Bool = true
    feedback::Union{Nothing, String} = nothing
end

abstract type AbstractMagenticManager end

function magentic_plan end
function magentic_select end
function magentic_finalize end

function _default_magentic_plan(context::MagenticContext)::MagenticTaskLedger
    user_prompts = [strip(get_text(message)) for message in context.conversation if message.role == ROLE_USER]
    facts = isempty(user_prompts) ? String[] : [last(user_prompts)]
    plan = ["Consult $(participant_id)" for participant_id in context.participant_ids]
    return MagenticTaskLedger(
        facts = facts,
        plan = plan,
        current_step = isempty(plan) ? 0 : 1,
    )
end

function _default_magentic_select(context::MagenticContext)::Union{Nothing, String}
    isempty(context.participant_ids) && return nothing
    limit = isempty(context.task_ledger.plan) ? length(context.participant_ids) : length(context.task_ledger.plan)
    context.round >= limit && return nothing
    if context.last_speaker === nothing
        return context.participant_ids[1]
    end
    idx = findfirst(==(context.last_speaker), context.participant_ids)
    idx === nothing && return context.participant_ids[1]
    return context.participant_ids[mod1(idx + 1, length(context.participant_ids))]
end

_default_magentic_finalize(context::MagenticContext) = deepcopy(context.conversation)

Base.@kwdef mutable struct StandardMagenticManager <: AbstractMagenticManager
    planner::Function = _default_magentic_plan
    selector::Function = _default_magentic_select
    finalizer::Function = _default_magentic_finalize
end

magentic_plan(manager::StandardMagenticManager, context::MagenticContext) = manager.planner(context)
magentic_select(manager::StandardMagenticManager, context::MagenticContext) = manager.selector(context)
magentic_finalize(manager::StandardMagenticManager, context::MagenticContext) = manager.finalizer(context)

for T in (
    ConcurrentParticipantResult,
    GroupChatState,
    GroupChatTurnResult,
    MagenticTaskLedger,
    MagenticProgressLedgerItem,
    MagenticProgressLedger,
    MagenticContext,
    MagenticPlanReviewRequest,
    MagenticPlanReviewResponse,
)
    register_state_type!(T)
end

const _SEQUENTIAL_INPUT_ID = "__sequential_input__"
const _SEQUENTIAL_END_ID = "__sequential_end__"
const _CONCURRENT_DISPATCH_ID = "__concurrent_dispatch__"
const _CONCURRENT_END_ID = "__concurrent_end__"
const _GROUP_CHAT_INPUT_ID = "__group_chat_input__"
const _GROUP_CHAT_ORCHESTRATOR_ID = "__group_chat_orchestrator__"
const _MAGENTIC_INPUT_ID = "__magentic_input__"
const _MAGENTIC_ORCHESTRATOR_ID = "__magentic_orchestrator__"
const _ORCHESTRATION_SESSION_KEY = "__orchestration_sessions__"

function _conversation_input_executor(id::String, description::String)::ExecutorSpec
    ExecutorSpec(
        id = id,
        description = description,
        input_types = DataType[Any],
        output_types = DataType[Vector{Message}],
        yield_types = DataType[],
        handler = (input, ctx) -> begin
            send_message(ctx, normalize_messages(input))
            return nothing
        end,
    )
end

function _conversation_output_executor(id::String = _SEQUENTIAL_END_ID)::ExecutorSpec
    ExecutorSpec(
        id = id,
        description = "Yield the final orchestration conversation",
        input_types = DataType[Vector{Message}],
        output_types = DataType[],
        yield_types = DataType[Vector{Message}],
        handler = (conversation, ctx) -> begin
            yield_output(ctx, deepcopy(conversation))
            return nothing
        end,
    )
end

function _supports_declared_type(types::Vector{DataType}, target::DataType)::Bool
    Any in types && return true
    target in types && return true
    return false
end

function _reserved_ids(kind::Symbol)
    if kind == :sequential
        return Set((_SEQUENTIAL_INPUT_ID, _SEQUENTIAL_END_ID))
    elseif kind == :concurrent
        return Set((_CONCURRENT_DISPATCH_ID, _CONCURRENT_END_ID))
    elseif kind == :group_chat
        return Set((_GROUP_CHAT_INPUT_ID, _GROUP_CHAT_ORCHESTRATOR_ID))
    elseif kind == :magentic
        return Set((_MAGENTIC_INPUT_ID, _MAGENTIC_ORCHESTRATOR_ID))
    end
    return Set{String}()
end

function _participant_id(participant)::String
    if participant isa Agent
        isempty(strip(participant.name)) && throw(ArgumentError("Agent participants must have a non-empty name"))
        return participant.name
    elseif participant isa AgentExecutor
        isempty(strip(participant.id)) && throw(ArgumentError("AgentExecutor participants must have a non-empty id"))
        return participant.id
    elseif participant isa ExecutorSpec
        isempty(strip(participant.id)) && throw(ArgumentError("ExecutorSpec participants must have a non-empty id"))
        return participant.id
    end
    throw(ArgumentError("Unsupported orchestration participant type: $(typeof(participant))"))
end

function _collect_orchestration_participants(participants::AbstractVector, kind::Symbol)::Vector{Any}
    isempty(participants) && throw(ArgumentError("participants cannot be empty"))
    reserved = _reserved_ids(kind)
    seen = Set{String}()
    collected = Any[]
    for participant in participants
        if !(participant isa Agent || participant isa AgentExecutor || participant isa ExecutorSpec)
            throw(ArgumentError("participants must be Agent, AgentExecutor, or ExecutorSpec"))
        end
        participant_id = _participant_id(participant)
        participant_id in reserved && throw(ArgumentError("participant id '$participant_id' is reserved by $(kind) orchestration"))
        participant_id in seen && throw(ArgumentError("Duplicate participant id '$participant_id'"))
        push!(seen, participant_id)
        push!(collected, participant)
    end
    return collected
end

function _load_stateful_session!(state::Dict{String, Any}, key::String, default_session::AgentSession)::AgentSession
    store = get!(state, _ORCHESTRATION_SESSION_KEY, Dict{String, Any}())
    if !(store isa AbstractDict)
        throw(WorkflowError("Workflow orchestration session store must be a dictionary"))
    end
    session_store = store isa Dict{String, Any} ? store : Dict{String, Any}(string(k) => v for (k, v) in pairs(store))
    state[_ORCHESTRATION_SESSION_KEY] = session_store

    if haskey(session_store, key)
        raw = session_store[key]
        if raw isa AgentSession
            return raw
        elseif raw isa AbstractDict
            return session_from_dict(Dict{String, Any}(string(k) => _deserialize_any_value(v) for (k, v) in pairs(raw)))
        end
    end

    session = deepcopy(default_session)
    session_store[key] = session_to_dict(session)
    return session
end

function _save_stateful_session!(state::Dict{String, Any}, key::String, session::AgentSession)
    store = get!(state, _ORCHESTRATION_SESSION_KEY, Dict{String, Any}())
    if !(store isa AbstractDict)
        throw(WorkflowError("Workflow orchestration session store must be a dictionary"))
    end
    session_store = store isa Dict{String, Any} ? store : Dict{String, Any}(string(k) => v for (k, v) in pairs(store))
    session_store[key] = session_to_dict(session)
    state[_ORCHESTRATION_SESSION_KEY] = session_store
    return session
end

function _default_participant_session(participant, participant_id::String)::AgentSession
    if participant isa AgentExecutor
        return deepcopy(participant.session)
    elseif participant isa Agent
        return create_session(participant; session_id = "orchestration:" * participant_id)
    end
    throw(ArgumentError("Only agent participants have sessions"))
end

function _run_orchestration_agent!(
    participant::Union{Agent, AgentExecutor},
    conversation::Vector{Message},
    state::Dict{String, Any},
    state_key::String,
)
    participant_id = _participant_id(participant)
    session = _load_stateful_session!(state, state_key, _default_participant_session(participant, participant_id))
    agent = participant isa AgentExecutor ? participant.agent : participant
    response = run_agent(agent, conversation; session = session)
    _save_stateful_session!(state, state_key, session)
    full_conversation = vcat(deepcopy(conversation), deepcopy(response.messages))
    if participant isa AgentExecutor
        participant.session = session
        participant._full_conversation = deepcopy(full_conversation)
        empty!(participant._message_cache)
    end
    return response, full_conversation
end

function _assistant_messages(messages::Vector{Message})::Vector{Message}
    [deepcopy(message) for message in messages if message.role == ROLE_ASSISTANT]
end

function _message_text(message::Message)::String
    strip(get_text(message))
end

function _last_message_with_role(messages::Vector{Message}, role::Symbol)::Union{Nothing, Message}
    for message in Iterators.reverse(messages)
        if message.role == role
            return message
        end
    end
    return nothing
end

function _coerce_concurrent_result(value)::ConcurrentParticipantResult
    if value isa ConcurrentParticipantResult
        return value
    end
    restored = _deserialize_any_value(value)
    restored isa ConcurrentParticipantResult || throw(WorkflowError("Expected ConcurrentParticipantResult, got $(typeof(restored))"))
    return restored
end

function _coerce_group_chat_state(value)::GroupChatState
    if value isa GroupChatState
        return value
    end
    restored = _deserialize_any_value(value)
    restored isa GroupChatState || throw(WorkflowError("Expected GroupChatState, got $(typeof(restored))"))
    return restored
end

function _coerce_magentic_context(value)::MagenticContext
    if value isa MagenticContext
        return value
    end
    restored = _deserialize_any_value(value)
    restored isa MagenticContext || throw(WorkflowError("Expected MagenticContext, got $(typeof(restored))"))
    return restored
end

function _coerce_plan_review_response(value)::MagenticPlanReviewResponse
    if value isa MagenticPlanReviewResponse
        return value
    elseif value isa AbstractString
        response = strip(value)
        isempty(response) && return MagenticPlanReviewResponse()
        lowercase(response) in ("approve", "approved", "ok", "yes", "continue") && return MagenticPlanReviewResponse()
        return MagenticPlanReviewResponse(approved = false, feedback = response)
    elseif value isa AbstractDict
        dict = Dict{String, Any}(string(k) => _deserialize_any_value(v) for (k, v) in pairs(value))
        approved = Bool(get(dict, "approved", true))
        feedback = get(dict, "feedback", nothing)
        return MagenticPlanReviewResponse(approved = approved, feedback = feedback === nothing ? nothing : string(feedback))
    end
    throw(WorkflowError("Unsupported Magentic plan review response type: $(typeof(value))"))
end

function _default_concurrent_output(results::Vector{ConcurrentParticipantResult})::Vector{Message}
    prompt_message = nothing
    replies = Message[]
    for result in results
        if prompt_message === nothing
            for message in result.conversation
                if message.role == ROLE_USER
                    prompt_message = deepcopy(message)
                    break
                end
            end
        end

        candidate = _last_message_with_role(result.response_messages, ROLE_ASSISTANT)
        candidate === nothing && (candidate = _last_message_with_role(result.conversation, ROLE_ASSISTANT))
        candidate === nothing && !isempty(result.conversation) && (candidate = deepcopy(result.conversation[end]))
        candidate === nothing || push!(replies, deepcopy(candidate))
    end

    output = Message[]
    prompt_message === nothing || push!(output, prompt_message)
    append!(output, replies)
    return output
end

function _invoke_concurrent_aggregator(aggregator, results::Vector{ConcurrentParticipantResult}, ctx::WorkflowContext)
    if aggregator === nothing
        yield_output(ctx, _default_concurrent_output(results))
        return nothing
    elseif aggregator isa Function
        if applicable(aggregator, results, ctx)
            output = aggregator(results, ctx)
        elseif applicable(aggregator, results)
            output = aggregator(results)
        else
            throw(WorkflowError("Concurrent aggregator callback must accept results or results with workflow context"))
        end
        output === nothing || yield_output(ctx, output)
        return nothing
    elseif aggregator isa ExecutorSpec
        inner = execute_handler(aggregator, results, [ctx.executor_id], ctx._state)
        if !isempty(inner._yielded_outputs)
            for output in inner._yielded_outputs
                yield_output(ctx, output)
            end
        elseif !isempty(inner._sent_messages)
            for message in inner._sent_messages
                yield_output(ctx, message.data)
            end
        end
        return nothing
    end

    throw(WorkflowError("Unsupported concurrent aggregator type: $(typeof(aggregator))"))
end

function _round_robin_choice(participant_ids::Vector{String}, last_speaker::Union{Nothing, String})::Union{Nothing, String}
    isempty(participant_ids) && return nothing
    last_speaker === nothing && return participant_ids[1]
    idx = findfirst(==(last_speaker), participant_ids)
    idx === nothing && return participant_ids[1]
    return participant_ids[mod1(idx + 1, length(participant_ids))]
end

function _parse_participant_choice(choice, participant_ids::Vector{String})::Union{Nothing, String}
    choice === nothing && return nothing
    raw = strip(string(choice))
    isempty(raw) && return nothing
    uppercase(raw) in ("DONE", "STOP", "FINISH", "END", "NONE") && return nothing
    lowered = lowercase(raw)
    for participant_id in participant_ids
        lowercase(participant_id) == lowered && return participant_id
    end
    for participant_id in participant_ids
        occursin(lowercase(participant_id), lowered) && return participant_id
    end
    throw(WorkflowError("Orchestrator selected unknown participant '$raw'"))
end

function _group_chat_choice_with_agent!(
    participant::Union{Agent, AgentExecutor},
    state::GroupChatState,
    workflow_state::Dict{String, Any},
    session_key::String,
)::Union{Nothing, String}
    transcript = join(["$(message.role): $(_message_text(message))" for message in state.conversation], "\n")
    prompt = join([
        "Choose the next participant for this group chat.",
        "Participants: " * join(state.participant_ids, ", "),
        "Last speaker: " * something(state.last_speaker, "none"),
        "Round: $(state.round)",
        "Reply with exactly one participant name or DONE.",
        "Conversation:",
        transcript,
    ], "\n")
    response, _ = _run_orchestration_agent!(participant, [Message(ROLE_USER, prompt)], workflow_state, session_key)
    return _parse_participant_choice(response.text, state.participant_ids)
end

function _sequential_agent_participant(
    participant::Union{Agent, AgentExecutor},
    session_key::String;
    chain_only_agent_responses::Bool = false,
    intermediate_output::Bool = false,
)::ExecutorSpec
    participant_id = _participant_id(participant)
    ExecutorSpec(
        id = participant_id,
        description = "Sequential orchestration participant: $participant_id",
        input_types = DataType[Vector{Message}],
        output_types = DataType[Vector{Message}],
        yield_types = intermediate_output ? DataType[Vector{Message}] : DataType[],
        handler = (conversation, ctx) -> begin
            response, full_conversation = _run_orchestration_agent!(participant, deepcopy(conversation), ctx._state, session_key)
            next_conversation = if chain_only_agent_responses
                vcat(_assistant_messages(conversation), deepcopy(response.messages))
            else
                full_conversation
            end
            send_message(ctx, next_conversation)
            intermediate_output && yield_output(ctx, deepcopy(next_conversation))
            return nothing
        end,
    )
end

function _concurrent_agent_participant(
    participant::Union{Agent, AgentExecutor},
    session_key::String;
    intermediate_output::Bool = false,
)::ExecutorSpec
    participant_id = _participant_id(participant)
    ExecutorSpec(
        id = participant_id,
        description = "Concurrent orchestration participant: $participant_id",
        input_types = DataType[Vector{Message}],
        output_types = DataType[ConcurrentParticipantResult],
        yield_types = intermediate_output ? DataType[ConcurrentParticipantResult] : DataType[],
        handler = (conversation, ctx) -> begin
            response, full_conversation = _run_orchestration_agent!(participant, deepcopy(conversation), ctx._state, session_key)
            result = ConcurrentParticipantResult(
                participant_id = participant_id,
                conversation = full_conversation,
                response_messages = deepcopy(response.messages),
            )
            send_message(ctx, result)
            intermediate_output && yield_output(ctx, result)
            return nothing
        end,
    )
end

function _concurrent_result_adapter(participant_id::String; intermediate_output::Bool = false)::ExecutorSpec
    adapter_id = participant_id * "__concurrent_result__"
    ExecutorSpec(
        id = adapter_id,
        description = "Convert participant conversation to concurrent aggregation result",
        input_types = DataType[Vector{Message}],
        output_types = DataType[ConcurrentParticipantResult],
        yield_types = intermediate_output ? DataType[ConcurrentParticipantResult] : DataType[],
        handler = (conversation, ctx) -> begin
            result = ConcurrentParticipantResult(
                participant_id = participant_id,
                conversation = deepcopy(conversation),
                response_messages = Message[],
            )
            send_message(ctx, result)
            intermediate_output && yield_output(ctx, result)
            return nothing
        end,
    )
end

function _group_chat_agent_participant(
    participant::Union{Agent, AgentExecutor},
    session_key::String,
)::ExecutorSpec
    participant_id = _participant_id(participant)
    ExecutorSpec(
        id = participant_id,
        description = "Group chat participant: $participant_id",
        input_types = DataType[Vector{Message}],
        output_types = DataType[GroupChatTurnResult],
        yield_types = DataType[],
        handler = (conversation, ctx) -> begin
            response, full_conversation = _run_orchestration_agent!(participant, deepcopy(conversation), ctx._state, session_key)
            send_message(ctx, GroupChatTurnResult(
                participant_id = participant_id,
                conversation = full_conversation,
                response_messages = deepcopy(response.messages),
            ))
            return nothing
        end,
    )
end

function _group_chat_result_adapter(participant_id::String)::ExecutorSpec
    adapter_id = participant_id * "__group_chat_result__"
    ExecutorSpec(
        id = adapter_id,
        description = "Convert participant conversation into a group chat turn result",
        input_types = DataType[Vector{Message}],
        output_types = DataType[GroupChatTurnResult],
        yield_types = DataType[],
        handler = (conversation, ctx) -> begin
            send_message(ctx, GroupChatTurnResult(
                participant_id = participant_id,
                conversation = deepcopy(conversation),
                response_messages = Message[],
            ))
            return nothing
        end,
    )
end

function _group_chat_state_key(orchestrator_id::String)
    "__group_chat_state__:" * orchestrator_id
end

function _magentic_state_key(orchestrator_id::String)
    "__magentic_context__:" * orchestrator_id
end

function _plan_review_state_key(orchestrator_id::String)
    "__magentic_plan_review__:" * orchestrator_id
end

function _selection_session_key(orchestrator_id::String)
    "selection:" * orchestrator_id
end

function _conversation_orchestrator_input(kind::Symbol)::ExecutorSpec
    if kind == :group_chat
        return _conversation_input_executor(_GROUP_CHAT_INPUT_ID, "Normalize input for group chat orchestration")
    elseif kind == :magentic
        return _conversation_input_executor(_MAGENTIC_INPUT_ID, "Normalize input for Magentic orchestration")
    end
    throw(ArgumentError("Unknown orchestrator input kind: $kind"))
end

function _update_group_chat_state(current::GroupChatState, turn::GroupChatTurnResult)::GroupChatState
    GroupChatState(
        conversation = deepcopy(turn.conversation),
        participant_ids = copy(current.participant_ids),
        round = current.round + 1,
        last_speaker = turn.participant_id,
        metadata = deepcopy(current.metadata),
    )
end

function _update_magentic_context(current::MagenticContext, turn::GroupChatTurnResult)::MagenticContext
    summary_message = _last_message_with_role(turn.response_messages, ROLE_ASSISTANT)
    summary_message === nothing && (summary_message = _last_message_with_role(turn.conversation, ROLE_ASSISTANT))
    summary = summary_message === nothing ? "" : _message_text(summary_message)

    previous_summary = isempty(current.progress_ledger.items) ? nothing : current.progress_ledger.items[end].summary
    progressed = !isempty(summary) && summary != previous_summary
    stall_count = progressed ? 0 : current.progress_ledger.stall_count + 1

    items = copy(current.progress_ledger.items)
    push!(items, MagenticProgressLedgerItem(
        participant_id = turn.participant_id,
        summary = summary,
        round = current.round + 1,
        progressed = progressed,
    ))

    ledger = MagenticTaskLedger(
        facts = copy(current.task_ledger.facts),
        plan = copy(current.task_ledger.plan),
        current_step = current.task_ledger.current_step + (progressed ? 1 : 0),
    )

    return MagenticContext(
        conversation = deepcopy(turn.conversation),
        participant_ids = copy(current.participant_ids),
        round = current.round + 1,
        last_speaker = turn.participant_id,
        task_ledger = ledger,
        progress_ledger = MagenticProgressLedger(items = items, stall_count = stall_count),
    )
end

mutable struct SequentialBuilder
    participants::Vector{Any}
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage}
    chain_only_agent_responses::Bool
    intermediate_outputs::Bool
    name::String
end

function SequentialBuilder(;
    participants::AbstractVector,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    chain_only_agent_responses::Bool = false,
    intermediate_outputs::Bool = false,
    name::String = "SequentialWorkflow",
)
    SequentialBuilder(
        _collect_orchestration_participants(participants, :sequential),
        checkpoint_storage,
        chain_only_agent_responses,
        intermediate_outputs,
        name,
    )
end

function build(builder::SequentialBuilder; validate_types::Bool = true)::Workflow
    input = _conversation_input_executor(_SEQUENTIAL_INPUT_ID, "Normalize input for sequential orchestration")
    output = _conversation_output_executor()
    workflow_builder = WorkflowBuilder(
        name = builder.name,
        start = input,
        checkpoint_storage = builder.checkpoint_storage,
    )

    previous_id = input.id
    total = length(builder.participants)
    for (index, participant) in enumerate(builder.participants)
        if participant isa ExecutorSpec
            _supports_declared_type(participant.input_types, Vector{Message}) ||
                throw(ArgumentError("Sequential custom executor '$(participant.id)' must accept Vector{Message}"))
            index < total && !_supports_declared_type(participant.output_types, Vector{Message}) &&
                throw(ArgumentError("Sequential custom executor '$(participant.id)' must output Vector{Message}"))
            add_executor(workflow_builder, participant)
            add_edge(workflow_builder, previous_id, participant.id)
            previous_id = participant.id
        else
            wrapped = _sequential_agent_participant(
                participant,
                "sequential:" * _participant_id(participant);
                chain_only_agent_responses = builder.chain_only_agent_responses,
                intermediate_output = builder.intermediate_outputs && index < total,
            )
            add_executor(workflow_builder, wrapped)
            add_edge(workflow_builder, previous_id, wrapped.id)
            previous_id = wrapped.id
        end
    end

    add_executor(workflow_builder, output)
    add_edge(workflow_builder, previous_id, output.id)
    add_output(workflow_builder, output.id)
    return build(workflow_builder; validate_types = validate_types)
end

mutable struct ConcurrentBuilder
    participants::Vector{Any}
    aggregator::Union{Nothing, Function, ExecutorSpec}
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage}
    intermediate_outputs::Bool
    name::String
end

function ConcurrentBuilder(;
    participants::AbstractVector,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    aggregator::Union{Nothing, Function, ExecutorSpec} = nothing,
    intermediate_outputs::Bool = false,
    name::String = "ConcurrentWorkflow",
)
    ConcurrentBuilder(
        _collect_orchestration_participants(participants, :concurrent),
        aggregator,
        checkpoint_storage,
        intermediate_outputs,
        name,
    )
end

function with_aggregator(builder::ConcurrentBuilder, aggregator::Union{Function, ExecutorSpec})::ConcurrentBuilder
    builder.aggregator = aggregator
    return builder
end

function build(builder::ConcurrentBuilder; validate_types::Bool = true)::Workflow
    dispatcher = _conversation_input_executor(_CONCURRENT_DISPATCH_ID, "Broadcast a normalized conversation to concurrent participants")
    participant_ids = String[_participant_id(participant) for participant in builder.participants]
    aggregator = ExecutorSpec(
        id = _CONCURRENT_END_ID,
        description = "Aggregate concurrent participant results",
        input_types = DataType[ConcurrentParticipantResult],
        output_types = DataType[],
        yield_types = builder.aggregator === nothing ? DataType[Vector{Message}] : DataType[Any],
        handler = (result, ctx) -> begin
            buffer_key = "__concurrent_results__:" * ctx.executor_id
            raw = get!(ctx._state, buffer_key, Dict{String, Any}())
            if !(raw isa AbstractDict)
                throw(WorkflowError("Concurrent aggregation buffer must be a dictionary"))
            end
            buffer = raw isa Dict{String, Any} ? raw : Dict{String, Any}(string(k) => v for (k, v) in pairs(raw))
            ctx._state[buffer_key] = buffer
            current = _coerce_concurrent_result(result)
            buffer[current.participant_id] = current

            length(buffer) < length(participant_ids) && return nothing

            ordered = ConcurrentParticipantResult[_coerce_concurrent_result(buffer[participant_id]) for participant_id in participant_ids]
            Base.delete!(ctx._state, buffer_key)
            _invoke_concurrent_aggregator(builder.aggregator, ordered, ctx)
            return nothing
        end,
    )

    workflow_builder = WorkflowBuilder(
        name = builder.name,
        start = dispatcher,
        checkpoint_storage = builder.checkpoint_storage,
    )
    add_executor(workflow_builder, aggregator)

    fan_in_sources = String[]
    for participant in builder.participants
        if participant isa ExecutorSpec
            _supports_declared_type(participant.input_types, Vector{Message}) ||
                throw(ArgumentError("Concurrent custom executor '$(participant.id)' must accept Vector{Message}"))
            _supports_declared_type(participant.output_types, Vector{Message}) ||
                throw(ArgumentError("Concurrent custom executor '$(participant.id)' must output Vector{Message}"))
            adapter = _concurrent_result_adapter(participant.id; intermediate_output = builder.intermediate_outputs)
            add_executor(workflow_builder, participant)
            add_executor(workflow_builder, adapter)
            add_edge(workflow_builder, participant.id, adapter.id)
            push!(fan_in_sources, adapter.id)
            builder.intermediate_outputs && add_output(workflow_builder, adapter.id)
        else
            wrapped = _concurrent_agent_participant(
                participant,
                "concurrent:" * _participant_id(participant);
                intermediate_output = builder.intermediate_outputs,
            )
            add_executor(workflow_builder, wrapped)
            push!(fan_in_sources, wrapped.id)
            builder.intermediate_outputs && add_output(workflow_builder, wrapped.id)
        end
    end

    add_fan_out(workflow_builder, dispatcher.id, participant_ids)
    add_fan_in(workflow_builder, fan_in_sources, aggregator.id)
    add_output(workflow_builder, aggregator.id)
    return build(workflow_builder; validate_types = validate_types)
end

mutable struct GroupChatBuilder
    participants::Vector{Any}
    selection_func::Union{Nothing, Function}
    orchestrator_agent::Union{Nothing, Agent, AgentExecutor}
    max_rounds::Union{Nothing, Int}
    termination_condition::Union{Nothing, Function}
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage}
    intermediate_outputs::Bool
    name::String
end

function GroupChatBuilder(;
    participants::AbstractVector,
    selection_func::Union{Nothing, Function} = nothing,
    orchestrator_agent::Union{Nothing, Agent, AgentExecutor} = nothing,
    max_rounds::Union{Nothing, Int} = nothing,
    termination_condition::Union{Nothing, Function} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    intermediate_outputs::Bool = false,
    name::String = "GroupChatWorkflow",
)
    selection_func !== nothing && orchestrator_agent !== nothing &&
        throw(ArgumentError("Provide either selection_func or orchestrator_agent, not both"))
    GroupChatBuilder(
        _collect_orchestration_participants(participants, :group_chat),
        selection_func,
        orchestrator_agent,
        max_rounds,
        termination_condition,
        checkpoint_storage,
        intermediate_outputs,
        name,
    )
end

function with_selection_func(builder::GroupChatBuilder, selection_func::Function)::GroupChatBuilder
    builder.selection_func = selection_func
    builder.orchestrator_agent = nothing
    return builder
end

function with_termination(builder::GroupChatBuilder, termination_condition::Function)::GroupChatBuilder
    builder.termination_condition = termination_condition
    return builder
end

function _group_chat_orchestrator(builder::GroupChatBuilder, participant_ids::Vector{String})::ExecutorSpec
    ExecutorSpec(
        id = _GROUP_CHAT_ORCHESTRATOR_ID,
        description = "Coordinate group chat turns",
        input_types = DataType[Any],
        output_types = DataType[Vector{Message}],
        yield_types = DataType[Vector{Message}],
        handler = (payload, ctx) -> begin
            state_key = _group_chat_state_key(ctx.executor_id)

            current = if payload isa GroupChatState
                payload
            elseif payload isa Vector{Message}
                GroupChatState(
                    conversation = deepcopy(payload),
                    participant_ids = copy(participant_ids),
                )
            elseif payload isa GroupChatTurnResult
                prior = haskey(ctx._state, state_key) ? _coerce_group_chat_state(ctx._state[state_key]) :
                    GroupChatState(conversation = Message[], participant_ids = copy(participant_ids))
                updated = _update_group_chat_state(prior, payload)
                builder.intermediate_outputs && yield_output(ctx, deepcopy(updated.conversation))
                updated
            else
                throw(WorkflowError("Group chat orchestrator received unsupported payload $(typeof(payload))"))
            end

            ctx._state[state_key] = current

            if builder.termination_condition !== nothing && builder.termination_condition(current.conversation)
                yield_output(ctx, deepcopy(current.conversation))
                return nothing
            end

            if builder.max_rounds !== nothing && current.round >= builder.max_rounds
                yield_output(ctx, deepcopy(current.conversation))
                return nothing
            end

            next_participant = if builder.selection_func !== nothing
                _parse_participant_choice(builder.selection_func(current), participant_ids)
            elseif builder.orchestrator_agent !== nothing
                _group_chat_choice_with_agent!(
                    builder.orchestrator_agent,
                    current,
                    ctx._state,
                    _selection_session_key(ctx.executor_id),
                )
            else
                _round_robin_choice(participant_ids, current.last_speaker)
            end

            if next_participant === nothing
                yield_output(ctx, deepcopy(current.conversation))
                return nothing
            end

            send_message(ctx, deepcopy(current.conversation); target_id = next_participant)
            return nothing
        end,
    )
end

function build(builder::GroupChatBuilder; validate_types::Bool = true)::Workflow
    input = _conversation_orchestrator_input(:group_chat)
    participant_ids = String[_participant_id(participant) for participant in builder.participants]
    orchestrator = _group_chat_orchestrator(builder, participant_ids)
    workflow_builder = WorkflowBuilder(
        name = builder.name,
        start = input,
        checkpoint_storage = builder.checkpoint_storage,
    )
    add_executor(workflow_builder, orchestrator)
    add_edge(workflow_builder, input.id, orchestrator.id)
    add_output(workflow_builder, orchestrator.id)

    for participant in builder.participants
        participant_id = _participant_id(participant)
        if participant isa ExecutorSpec
            _supports_declared_type(participant.input_types, Vector{Message}) ||
                throw(ArgumentError("Group chat custom executor '$(participant.id)' must accept Vector{Message}"))
            _supports_declared_type(participant.output_types, Vector{Message}) ||
                throw(ArgumentError("Group chat custom executor '$(participant.id)' must output Vector{Message}"))
            adapter = _group_chat_result_adapter(participant_id)
            add_executor(workflow_builder, participant)
            add_executor(workflow_builder, adapter)
            add_edge(workflow_builder, orchestrator.id, participant_id)
            add_edge(workflow_builder, participant_id, adapter.id)
            add_edge(workflow_builder, adapter.id, orchestrator.id)
        else
            wrapped = _group_chat_agent_participant(participant, "group_chat:" * participant_id)
            add_executor(workflow_builder, wrapped)
            add_edge(workflow_builder, orchestrator.id, wrapped.id)
            add_edge(workflow_builder, wrapped.id, orchestrator.id)
        end
    end

    return build(workflow_builder; validate_types = validate_types)
end

mutable struct MagenticBuilder
    participants::Vector{Any}
    manager::AbstractMagenticManager
    enable_plan_review::Bool
    max_stall_count::Int
    max_round_count::Union{Nothing, Int}
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage}
    intermediate_outputs::Bool
    name::String
end

function MagenticBuilder(;
    participants::AbstractVector,
    manager::AbstractMagenticManager = StandardMagenticManager(),
    enable_plan_review::Bool = false,
    max_stall_count::Int = 3,
    max_round_count::Union{Nothing, Int} = nothing,
    checkpoint_storage::Union{Nothing, AbstractCheckpointStorage} = nothing,
    intermediate_outputs::Bool = false,
    name::String = "MagenticWorkflow",
)
    max_stall_count > 0 || throw(ArgumentError("max_stall_count must be positive"))
    MagenticBuilder(
        _collect_orchestration_participants(participants, :magentic),
        manager,
        enable_plan_review,
        max_stall_count,
        max_round_count,
        checkpoint_storage,
        intermediate_outputs,
        name,
    )
end

function with_plan_review(builder::MagenticBuilder; enable::Bool = true)::MagenticBuilder
    builder.enable_plan_review = enable
    return builder
end

function _magentic_orchestrator(builder::MagenticBuilder, participant_ids::Vector{String})::ExecutorSpec
    ExecutorSpec(
        id = _MAGENTIC_ORCHESTRATOR_ID,
        description = "Coordinate planning-oriented Magentic orchestration",
        input_types = DataType[Any],
        output_types = DataType[Vector{Message}],
        yield_types = DataType[Vector{Message}],
        handler = (payload, ctx) -> begin
            context_key = _magentic_state_key(ctx.executor_id)
            review_key = _plan_review_state_key(ctx.executor_id)

            context = if payload isa Vector{Message}
                initial = MagenticContext(
                    conversation = deepcopy(payload),
                    participant_ids = copy(participant_ids),
                )
                planned = MagenticContext(
                    conversation = initial.conversation,
                    participant_ids = initial.participant_ids,
                    round = initial.round,
                    last_speaker = initial.last_speaker,
                    task_ledger = magentic_plan(builder.manager, initial),
                    progress_ledger = initial.progress_ledger,
                )
                ctx._state[context_key] = planned
                if builder.enable_plan_review && !Bool(get(ctx._state, review_key, false))
                    request_info(ctx, MagenticPlanReviewRequest(task_ledger = planned.task_ledger, context = planned))
                    return nothing
                end
                planned
            elseif payload isa GroupChatTurnResult
                prior = haskey(ctx._state, context_key) ? _coerce_magentic_context(ctx._state[context_key]) :
                    MagenticContext(conversation = Message[], participant_ids = copy(participant_ids))
                updated = _update_magentic_context(prior, payload)
                ctx._state[context_key] = updated
                builder.intermediate_outputs && yield_output(ctx, deepcopy(updated.conversation))
                updated
            else
                prior = haskey(ctx._state, context_key) ? _coerce_magentic_context(ctx._state[context_key]) :
                    throw(WorkflowError("Magentic plan review response received before planning"))
                review = _coerce_plan_review_response(payload)
                reviewed = if review.approved
                    prior
                else
                    feedback = review.feedback === nothing ? "" : review.feedback
                    revised = MagenticContext(
                        conversation = prior.conversation,
                        participant_ids = prior.participant_ids,
                        round = prior.round,
                        last_speaker = prior.last_speaker,
                        task_ledger = MagenticTaskLedger(
                            facts = vcat(copy(prior.task_ledger.facts), isempty(feedback) ? String[] : ["Reviewer feedback: " * feedback]),
                            plan = copy(prior.task_ledger.plan),
                            current_step = prior.task_ledger.current_step,
                        ),
                        progress_ledger = prior.progress_ledger,
                    )
                    MagenticContext(
                        conversation = revised.conversation,
                        participant_ids = revised.participant_ids,
                        round = revised.round,
                        last_speaker = revised.last_speaker,
                        task_ledger = magentic_plan(builder.manager, revised),
                        progress_ledger = revised.progress_ledger,
                    )
                end
                ctx._state[review_key] = true
                ctx._state[context_key] = reviewed
                reviewed
            end

            if builder.max_round_count !== nothing && context.round >= builder.max_round_count
                yield_output(ctx, deepcopy(magentic_finalize(builder.manager, context)))
                return nothing
            end

            if context.progress_ledger.stall_count >= builder.max_stall_count
                yield_output(ctx, deepcopy(magentic_finalize(builder.manager, context)))
                return nothing
            end

            next_participant = _parse_participant_choice(magentic_select(builder.manager, context), participant_ids)
            if next_participant === nothing
                yield_output(ctx, deepcopy(magentic_finalize(builder.manager, context)))
                return nothing
            end

            send_message(ctx, deepcopy(context.conversation); target_id = next_participant)
            return nothing
        end,
    )
end

function build(builder::MagenticBuilder; validate_types::Bool = true)::Workflow
    input = _conversation_orchestrator_input(:magentic)
    participant_ids = String[_participant_id(participant) for participant in builder.participants]
    orchestrator = _magentic_orchestrator(builder, participant_ids)
    workflow_builder = WorkflowBuilder(
        name = builder.name,
        start = input,
        checkpoint_storage = builder.checkpoint_storage,
    )
    add_executor(workflow_builder, orchestrator)
    add_edge(workflow_builder, input.id, orchestrator.id)
    add_output(workflow_builder, orchestrator.id)

    for participant in builder.participants
        participant_id = _participant_id(participant)
        if participant isa ExecutorSpec
            _supports_declared_type(participant.input_types, Vector{Message}) ||
                throw(ArgumentError("Magentic custom executor '$(participant.id)' must accept Vector{Message}"))
            _supports_declared_type(participant.output_types, Vector{Message}) ||
                throw(ArgumentError("Magentic custom executor '$(participant.id)' must output Vector{Message}"))
            adapter = _group_chat_result_adapter(participant_id)
            add_executor(workflow_builder, participant)
            add_executor(workflow_builder, adapter)
            add_edge(workflow_builder, orchestrator.id, participant_id)
            add_edge(workflow_builder, participant_id, adapter.id)
            add_edge(workflow_builder, adapter.id, orchestrator.id)
        else
            wrapped = _group_chat_agent_participant(participant, "magentic:" * participant_id)
            add_executor(workflow_builder, wrapped)
            add_edge(workflow_builder, orchestrator.id, wrapped.id)
            add_edge(workflow_builder, wrapped.id, orchestrator.id)
        end
    end

    return build(workflow_builder; validate_types = validate_types)
end
