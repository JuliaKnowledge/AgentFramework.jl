# Chat client abstraction for AgentFramework.jl
# Mirrors Python BaseChatClient, ChatResponse, ChatResponseUpdate, ResponseStream.

"""
    FinishReason

Reason a chat response completed.
"""
@enum FinishReason begin
    STOP
    LENGTH
    TOOL_CALLS
    CONTENT_FILTER
    FINISH_ERROR
end

const FINISH_REASON_STRINGS = Dict{FinishReason, String}(
    STOP => "stop",
    LENGTH => "length",
    TOOL_CALLS => "tool_calls",
    CONTENT_FILTER => "content_filter",
    FINISH_ERROR => "error",
)

const STRING_TO_FINISH_REASON = Dict{String, FinishReason}(v => k for (k, v) in FINISH_REASON_STRINGS)

function parse_finish_reason(s::AbstractString)::FinishReason
    get(STRING_TO_FINISH_REASON, s, STOP)
end

"""
    ChatOptions

Options for a chat client request.

# Fields
- `model::Union{Nothing, String}`: Model identifier override.
- `temperature::Union{Nothing, Float64}`: Sampling temperature.
- `top_p::Union{Nothing, Float64}`: Nucleus sampling parameter.
- `max_tokens::Union{Nothing, Int}`: Maximum tokens in response.
- `stop::Union{Nothing, Vector{String}}`: Stop sequences.
- `tools::Vector{FunctionTool}`: Tools available for this request.
- `tool_choice::Union{Nothing, String}`: Tool selection mode ("auto", "none", "required", or function name).
- `response_format::Union{Nothing, Dict{String, Any}}`: Structured output format.
- `additional::Dict{String, Any}`: Provider-specific options.
"""
Base.@kwdef mutable struct ChatOptions
    model::Union{Nothing, String} = nothing
    temperature::Union{Nothing, Float64} = nothing
    top_p::Union{Nothing, Float64} = nothing
    max_tokens::Union{Nothing, Int} = nothing
    stop::Union{Nothing, Vector{String}} = nothing
    tools::Vector{FunctionTool} = FunctionTool[]
    tool_choice::Union{Nothing, String} = nothing
    response_format::Union{Nothing, Dict{String, Any}} = nothing
    additional::Dict{String, Any} = Dict{String, Any}()
end

"""
    merge_chat_options(base, override) -> ChatOptions

Merge two ChatOptions, with override values taking precedence.
"""
function merge_chat_options(base::ChatOptions, override::ChatOptions)::ChatOptions
    ChatOptions(
        model = override.model !== nothing ? override.model : base.model,
        temperature = override.temperature !== nothing ? override.temperature : base.temperature,
        top_p = override.top_p !== nothing ? override.top_p : base.top_p,
        max_tokens = override.max_tokens !== nothing ? override.max_tokens : base.max_tokens,
        stop = override.stop !== nothing ? override.stop : base.stop,
        tools = !isempty(override.tools) ? override.tools : base.tools,
        tool_choice = override.tool_choice !== nothing ? override.tool_choice : base.tool_choice,
        response_format = override.response_format !== nothing ? override.response_format : base.response_format,
        additional = merge(base.additional, override.additional),
    )
end

# ── Chat Response Types ──────────────────────────────────────────────────────

"""
    ChatResponseUpdate

A streaming update from a chat client.

# Fields
- `role::Union{Nothing, Symbol}`: Message role (usually set on first update).
- `contents::Vector{Content}`: Content items in this update.
- `finish_reason::Union{Nothing, FinishReason}`: Set on the final update.
- `model_id::Union{Nothing, String}`: Model that produced this update.
- `usage_details::Union{Nothing, UsageDetails}`: Token usage (usually on final update).
- `response_id::Union{Nothing, String}`: Response identifier.
"""
Base.@kwdef mutable struct ChatResponseUpdate
    role::Union{Nothing, Symbol} = nothing
    contents::Vector{Content} = Content[]
    finish_reason::Union{Nothing, FinishReason} = nothing
    model_id::Union{Nothing, String} = nothing
    usage_details::Union{Nothing, UsageDetails} = nothing
    response_id::Union{Nothing, String} = nothing
    raw_representation::Any = nothing
end

"""
    get_text(update::ChatResponseUpdate) -> String

Extract text from a streaming update.
"""
function get_text(update::ChatResponseUpdate)::String
    join((get_text(c) for c in update.contents if is_text(c)), "")
end

"""
    ChatResponse

Complete response from a chat client.

# Fields
- `messages::Vector{Message}`: Response messages.
- `response_id::Union{Nothing, String}`: Response identifier.
- `conversation_id::Union{Nothing, String}`: Conversation state identifier.
- `model_id::Union{Nothing, String}`: Model used.
- `finish_reason::Union{Nothing, FinishReason}`: Completion reason.
- `usage_details::Union{Nothing, UsageDetails}`: Token usage.
"""
Base.@kwdef mutable struct ChatResponse
    messages::Vector{Message} = Message[]
    response_id::Union{Nothing, String} = nothing
    conversation_id::Union{Nothing, String} = nothing
    model_id::Union{Nothing, String} = nothing
    created_at::Union{Nothing, String} = nothing
    finish_reason::Union{Nothing, FinishReason} = nothing
    usage_details::Union{Nothing, UsageDetails} = nothing
    additional_properties::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end

"""
    get_text(response::ChatResponse) -> String

Extract concatenated text from all response messages.
"""
function get_text(response::ChatResponse)::String
    join((get_text(m) for m in response.messages), " ")
end

# Property-style .text access
function Base.getproperty(r::ChatResponse, name::Symbol)
    name === :text && return get_text(r)
    return getfield(r, name)
end

function Base.propertynames(::ChatResponse, private::Bool=false)
    return (:messages, :response_id, :conversation_id, :model_id, :created_at,
            :finish_reason, :usage_details, :additional_properties, :raw_representation, :text)
end

"""
    ChatResponse(updates::Vector{ChatResponseUpdate}) -> ChatResponse

Build a ChatResponse by joining streaming updates.
"""
function ChatResponse(updates::Vector{ChatResponseUpdate})
    isempty(updates) && return ChatResponse()

    # Accumulate content
    role = :assistant
    contents = Content[]
    tool_accumulator = StreamingToolAccumulator()
    finish_reason = nothing
    model_id = nothing
    usage = nothing
    response_id = nothing

    for update in updates
        if update.role !== nothing
            role = update.role
        end
        fragments = _extract_streaming_tool_fragments(update)
        if !isempty(fragments)
            for fragment in fragments
                accumulate_tool_call!(
                    tool_accumulator,
                    fragment["index"];
                    call_id = get(fragment, "call_id", nothing),
                    name = get(fragment, "name", nothing),
                    arguments_fragment = get(fragment, "arguments_fragment", nothing),
                )
            end
            append!(contents, [content for content in update.contents if !is_function_call(content)])
        else
            append!(contents, update.contents)
        end
        if update.finish_reason !== nothing
            finish_reason = update.finish_reason
        end
        if update.model_id !== nothing
            model_id = update.model_id
        end
        usage = add_usage_details(usage, update.usage_details)
        if update.response_id !== nothing
            response_id = update.response_id
        end
    end

    if has_tool_calls(tool_accumulator)
        append!(contents, get_accumulated_tool_calls(tool_accumulator))
    end

    # Coalesce adjacent text contents
    coalesced = _coalesce_text_contents(contents)

    msg = Message(role=role, contents=coalesced)
    ChatResponse(
        messages = [msg],
        finish_reason = finish_reason,
        model_id = model_id,
        usage_details = usage,
        response_id = response_id,
    )
end

function _coalesce_text_contents(contents::Vector{Content})::Vector{Content}
    result = Content[]
    for c in contents
        if is_text(c) && !isempty(result) && is_text(last(result))
            # Merge text
            prev = last(result)
            prev.text = string(something(prev.text, ""), something(c.text, ""))
        else
            push!(result, c)
        end
    end
    return result
end

function _normalize_tool_fragment_index(value, fallback::Int)::Int
    if value isa Integer
        return Int(value)
    elseif value isa AbstractString
        parsed = tryparse(Int, value)
        return parsed === nothing ? fallback : parsed
    else
        return fallback
    end
end

function _extract_openai_tool_fragments(raw::Dict{String, Any})::Vector{Dict{String, Any}}
    choices = get(raw, "choices", Any[])
    isempty(choices) && return Dict{String, Any}[]

    choice = choices[1]
    choice_dict = choice isa AbstractDict ? Dict{String, Any}(string(k) => v for (k, v) in pairs(choice)) : Dict{String, Any}()
    delta = get(choice_dict, "delta", Dict{String, Any}())
    delta_dict = delta isa AbstractDict ? Dict{String, Any}(string(k) => v for (k, v) in pairs(delta)) : Dict{String, Any}()
    tool_calls = get(delta_dict, "tool_calls", nothing)
    tool_calls isa AbstractVector || return Dict{String, Any}[]

    fragments = Dict{String, Any}[]
    for (fallback_index, tool_call) in enumerate(tool_calls)
        tc_dict = tool_call isa AbstractDict ? Dict{String, Any}(string(k) => v for (k, v) in pairs(tool_call)) : Dict{String, Any}()
        func_data = get(tc_dict, "function", Dict{String, Any}())
        func_dict = func_data isa AbstractDict ? Dict{String, Any}(string(k) => v for (k, v) in pairs(func_data)) : Dict{String, Any}()
        push!(fragments, Dict{String, Any}(
            "index" => _normalize_tool_fragment_index(get(tc_dict, "index", fallback_index - 1), fallback_index - 1),
            "call_id" => let value = get(tc_dict, "id", nothing); value === nothing ? nothing : string(value) end,
            "name" => let value = get(func_dict, "name", nothing); value === nothing ? nothing : string(value) end,
            "arguments_fragment" => let value = get(func_dict, "arguments", nothing); value === nothing ? nothing : string(value) end,
        ))
    end
    return fragments
end

function _extract_streaming_tool_fragments(update::ChatResponseUpdate)::Vector{Dict{String, Any}}
    raw = update.raw_representation
    raw isa AbstractDict || return Dict{String, Any}[]
    raw_dict = Dict{String, Any}(string(k) => v for (k, v) in pairs(raw))

    if haskey(raw_dict, "__streaming_tool_fragments__")
        fragments = get(raw_dict, "__streaming_tool_fragments__", Any[])
        return [
            Dict{String, Any}(
                "index" => _normalize_tool_fragment_index(get(fragment, "index", idx - 1), idx - 1),
                "call_id" => let value = get(fragment, "call_id", nothing); value === nothing ? nothing : string(value) end,
                "name" => let value = get(fragment, "name", nothing); value === nothing ? nothing : string(value) end,
                "arguments_fragment" => let value = get(fragment, "arguments_fragment", nothing); value === nothing ? nothing : string(value) end,
            )
            for (idx, fragment) in enumerate(fragments)
            if fragment isa AbstractDict
        ]
    end

    return _extract_openai_tool_fragments(raw_dict)
end

# ── Response Stream ──────────────────────────────────────────────────────────

"""
    ResponseStream{T}

Wrapper around a `Channel{T}` that provides async iteration for streaming responses.

# Usage
```julia
stream = run_agent_streaming(agent, "Hello")
for update in stream
    print(get_text(update))
end
response = get_final_response(stream)
```
"""
mutable struct ResponseStream{T}
    channel::Channel{T}
    final_response::Any
    task::Union{Nothing, Task}
    error::Any
    _lock::ReentrantLock
end

function ResponseStream{T}(channel::Channel{T}) where T
    ResponseStream{T}(channel, nothing, nothing, nothing, ReentrantLock())
end

function _await_stream_completion!(stream::ResponseStream)
    task = lock(stream._lock) do
        stream.task
    end
    task !== nothing && wait(task)

    err = lock(stream._lock) do
        stream.error
    end
    err === nothing || throw(err)
    return nothing
end

function _iterate_stream(stream::ResponseStream, result)
    if result === nothing
        _await_stream_completion!(stream)
        return nothing
    end
    return result
end

Base.iterate(s::ResponseStream) = _iterate_stream(s, iterate(s.channel))
Base.iterate(s::ResponseStream, state) = _iterate_stream(s, iterate(s.channel, state))
Base.eltype(::Type{ResponseStream{T}}) where T = T

"""
    get_final_response(stream::ResponseStream)

Get the final aggregated response after the stream completes.
Must be called after iterating the stream.
"""
function get_final_response(stream::ResponseStream)
    _await_stream_completion!(stream)
    lock(stream._lock) do
        return stream.final_response
    end
end

# ── Agent Response Types ─────────────────────────────────────────────────────

"""
    AgentResponse

Response from an agent's `run_agent()` call.

# Fields
- `messages::Vector{Message}`: Response messages.
- `finish_reason::Union{Nothing, FinishReason}`: Completion reason.
- `usage_details::Union{Nothing, UsageDetails}`: Aggregated token usage.
- `model_id::Union{Nothing, String}`: Model used.
"""
Base.@kwdef mutable struct AgentResponse
    messages::Vector{Message} = Message[]
    finish_reason::Union{Nothing, FinishReason} = nothing
    usage_details::Union{Nothing, UsageDetails} = nothing
    model_id::Union{Nothing, String} = nothing
end

"""
    get_text(response::AgentResponse) -> String
"""
function get_text(response::AgentResponse)::String
    join((get_text(m) for m in response.messages), " ")
end

function Base.getproperty(r::AgentResponse, name::Symbol)
    name === :text && return get_text(r)
    return getfield(r, name)
end

function Base.propertynames(::AgentResponse, private::Bool=false)
    return (:messages, :finish_reason, :usage_details, :model_id, :text)
end

"""
    AgentResponseUpdate

A streaming update from an agent.
"""
Base.@kwdef mutable struct AgentResponseUpdate
    role::Union{Nothing, Symbol} = nothing
    contents::Vector{Content} = Content[]
    finish_reason::Union{Nothing, FinishReason} = nothing
    model_id::Union{Nothing, String} = nothing
    usage_details::Union{Nothing, UsageDetails} = nothing
end

function get_text(update::AgentResponseUpdate)::String
    join((get_text(c) for c in update.contents if is_text(c)), "")
end

# ── Abstract Chat Client Interface ───────────────────────────────────────────

"""
    get_response(client::AbstractChatClient, messages, options) -> ChatResponse

Send messages to the LLM and get a complete response.
Must be implemented by concrete chat client types.
"""
function get_response end

"""
    get_response_streaming(client::AbstractChatClient, messages, options) -> Channel{ChatResponseUpdate}

Send messages to the LLM and get a streaming response channel.
Must be implemented by concrete chat client types.
"""
function get_response_streaming end

# ── Streaming Tool Call Accumulation ─────────────────────────────────────────

"""
    StreamingToolAccumulator

Accumulates partial tool call data across streaming chunks.
Tool calls arrive in pieces: first the function name, then argument fragments.
"""
mutable struct StreamingToolAccumulator
    tool_calls::Dict{Int, Dict{String, Any}}  # index → {name, arguments, call_id}
    lock::ReentrantLock
end

StreamingToolAccumulator() = StreamingToolAccumulator(Dict{Int, Dict{String, Any}}(), ReentrantLock())

"""Process a streaming chunk that may contain partial tool call data."""
function accumulate_tool_call!(acc::StreamingToolAccumulator, index::Int;
    call_id::Union{Nothing, String} = nothing,
    name::Union{Nothing, String} = nothing,
    arguments_fragment::Union{Nothing, String} = nothing,
)
    lock(acc.lock) do
        if !haskey(acc.tool_calls, index)
            acc.tool_calls[index] = Dict{String, Any}("name" => "", "arguments" => "", "call_id" => "")
        end
        tc = acc.tool_calls[index]
        call_id !== nothing && (tc["call_id"] = call_id)
        name !== nothing && (tc["name"] = tc["name"] * name)
        arguments_fragment !== nothing && (tc["arguments"] = tc["arguments"] * arguments_fragment)
    end
end

"""Get all completed tool calls as Content items."""
function get_accumulated_tool_calls(acc::StreamingToolAccumulator)::Vector{Content}
    lock(acc.lock) do
        contents = Content[]
        for idx in sort(collect(keys(acc.tool_calls)))
            tc = acc.tool_calls[idx]
            name = tc["name"]
            isempty(name) && continue
            push!(contents, function_call_content(tc["call_id"], name, tc["arguments"]))
        end
        return contents
    end
end

"""Reset the accumulator."""
function reset_accumulator!(acc::StreamingToolAccumulator)
    lock(acc.lock) do
        empty!(acc.tool_calls)
    end
end

"""Check if there are any accumulated tool calls."""
function has_tool_calls(acc::StreamingToolAccumulator)::Bool
    lock(acc.lock) do
        !isempty(acc.tool_calls)
    end
end
