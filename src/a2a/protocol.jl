@enum A2ATaskState begin
    A2A_TASK_SUBMITTED
    A2A_TASK_WORKING
    A2A_TASK_INPUT_REQUIRED
    A2A_TASK_AUTH_REQUIRED
    A2A_TASK_COMPLETED
    A2A_TASK_FAILED
    A2A_TASK_CANCELED
    A2A_TASK_REJECTED
    A2A_TASK_UNKNOWN
end

const A2A_TASK_STATE_STRINGS = Dict{A2ATaskState, String}(
    A2A_TASK_SUBMITTED => "submitted",
    A2A_TASK_WORKING => "working",
    A2A_TASK_INPUT_REQUIRED => "input_required",
    A2A_TASK_AUTH_REQUIRED => "auth_required",
    A2A_TASK_COMPLETED => "completed",
    A2A_TASK_FAILED => "failed",
    A2A_TASK_CANCELED => "canceled",
    A2A_TASK_REJECTED => "rejected",
    A2A_TASK_UNKNOWN => "unknown",
)

const STRING_TO_A2A_TASK_STATE = Dict{String, A2ATaskState}(value => key for (key, value) in A2A_TASK_STATE_STRINGS)
const TERMINAL_TASK_STATES = Set([A2A_TASK_COMPLETED, A2A_TASK_FAILED, A2A_TASK_CANCELED, A2A_TASK_REJECTED])
const IN_PROGRESS_TASK_STATES = Set([A2A_TASK_SUBMITTED, A2A_TASK_WORKING, A2A_TASK_INPUT_REQUIRED, A2A_TASK_AUTH_REQUIRED])

function task_state_string(state::A2ATaskState)::String
    A2A_TASK_STATE_STRINGS[state]
end

function parse_task_state(value)::A2ATaskState
    get(STRING_TO_A2A_TASK_STATE, lowercase(string(value)), A2A_TASK_UNKNOWN)
end

is_terminal_task_state(state::A2ATaskState)::Bool = state in TERMINAL_TASK_STATES
is_in_progress_task_state(state::A2ATaskState)::Bool = state in IN_PROGRESS_TASK_STATES

Base.@kwdef mutable struct A2AContinuationToken
    task_id::String
    context_id::Union{Nothing, String} = nothing
end

function Base.show(io::IO, token::A2AContinuationToken)
    print(io, "A2AContinuationToken(task_id=\"", token.task_id, "\")")
end

function continuation_token_to_dict(token::A2AContinuationToken)::Dict{String, Any}
    payload = Dict{String, Any}("task_id" => token.task_id)
    token.context_id !== nothing && (payload["context_id"] = token.context_id)
    return payload
end

function continuation_token_from_dict(data::AbstractDict)::A2AContinuationToken
    values = Dict{String, Any}(string(key) => value for (key, value) in pairs(data))
    haskey(values, "task_id") || throw(A2AError("Continuation token is missing `task_id`"))
    return A2AContinuationToken(
        task_id = string(values["task_id"]),
        context_id = get(values, "context_id", nothing) === nothing ? nothing : string(values["context_id"]),
    )
end

Base.@kwdef mutable struct A2AAgentCard
    name::Union{Nothing, String} = nothing
    description::Union{Nothing, String} = nothing
    url::String
    version::Union{Nothing, String} = nothing
    default_input_modes::Vector{String} = String[]
    default_output_modes::Vector{String} = String[]
    capabilities::Dict{String, Any} = Dict{String, Any}()
    skills::Vector{Dict{String, Any}} = Dict{String, Any}[]
    additional_properties::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end

function Base.show(io::IO, card::A2AAgentCard)
    name = card.name === nothing ? card.url : card.name
    print(io, "A2AAgentCard(\"", name, "\")")
end

Base.@kwdef mutable struct A2ATaskStatus
    state::A2ATaskState = A2A_TASK_UNKNOWN
    message::Union{Nothing, Message} = nothing
    timestamp::Union{Nothing, String} = nothing
    raw_representation::Any = nothing
end

Base.@kwdef mutable struct A2AArtifact
    artifact_id::Union{Nothing, String} = nothing
    name::Union{Nothing, String} = nothing
    description::Union{Nothing, String} = nothing
    contents::Vector{Content} = Content[]
    additional_properties::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end

Base.@kwdef mutable struct A2ATask
    id::String
    context_id::Union{Nothing, String} = nothing
    status::A2ATaskStatus = A2ATaskStatus()
    artifacts::Vector{A2AArtifact} = A2AArtifact[]
    history::Vector{Message} = Message[]
    metadata::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end

function Base.show(io::IO, task::A2ATask)
    print(io, "A2ATask(\"", task.id, "\", state=", task_state_string(task.status.state), ")")
end
