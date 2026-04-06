"""
AG-UI protocol helpers for request parsing and SSE event formatting.
"""

const AGUI_ROLE_USER = "user"
const AGUI_ROLE_ASSISTANT = "assistant"
const AGUI_ROLE_SYSTEM = "system"
const AGUI_ROLE_DEVELOPER = "developer"
const AGUI_ROLE_TOOL = "tool"

const AGUI_EVENT_RUN_STARTED = "RUN_STARTED"
const AGUI_EVENT_RUN_FINISHED = "RUN_FINISHED"
const AGUI_EVENT_RUN_ERROR = "RUN_ERROR"
const AGUI_EVENT_TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
const AGUI_EVENT_TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
const AGUI_EVENT_TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
const AGUI_EVENT_TOOL_CALL_START = "TOOL_CALL_START"
const AGUI_EVENT_TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
const AGUI_EVENT_TOOL_CALL_END = "TOOL_CALL_END"
const AGUI_EVENT_TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
const AGUI_EVENT_STATE_SNAPSHOT = "STATE_SNAPSHOT"
const AGUI_EVENT_STATE_DELTA = "STATE_DELTA"

function _agui_generated_id(prefix::AbstractString)
    return string(prefix, "_", replace(string(UUIDs.uuid4()), "-" => ""))
end

function _agui_get(dict::AbstractDict, keys::AbstractString...; default=nothing)
    for key in keys
        if haskey(dict, key)
            return dict[key]
        end
    end
    return default
end

function _agui_string(value)
    value === nothing && return ""
    return value isa AbstractString ? String(value) : string(value)
end

function _agui_json_string(value)
    value === nothing && return "{}"
    return value isa AbstractString ? String(value) : JSON3.write(value)
end

function parse_agui_request(body::AbstractString)
    parsed = JSON3.read(String(body), Dict{String, Any})
    parsed isa Dict || error("expected AG-UI request object")

    messages = get(parsed, "messages", nothing)
    messages isa AbstractVector || error("missing required field: messages")

    return Dict{String, Any}(
        "messages" => collect(messages),
        "thread_id" => _agui_get(parsed, "threadId", "thread_id"; default=nothing),
        "run_id" => something(_agui_get(parsed, "runId", "run_id"; default=nothing), _agui_generated_id("run")),
        "state" => get(parsed, "state", nothing),
        "context" => get(parsed, "context", nothing),
        "forwarded_props" => _agui_get(parsed, "forwardedProps", "forwarded_props"; default=nothing),
    )
end

function _agui_framework_role(role::AbstractString)
    normalized = lowercase(String(role))
    normalized == AGUI_ROLE_DEVELOPER && return :system
    normalized in (AGUI_ROLE_USER, AGUI_ROLE_ASSISTANT, AGUI_ROLE_SYSTEM, AGUI_ROLE_TOOL) ||
        error("unsupported AG-UI role: $role")
    return Symbol(normalized)
end

function _agui_tool_call(tool_call::AbstractDict)
    function_payload = get(tool_call, "function", nothing)
    if function_payload !== nothing && !(function_payload isa AbstractDict)
        error("assistant toolCalls.function must be an object")
    end

    payload = function_payload isa AbstractDict ? function_payload : tool_call
    name = _agui_string(_agui_get(payload, "name", "toolCallName"; default=""))
    isempty(name) && error("assistant tool call missing function name")

    call_id = _agui_string(_agui_get(tool_call, "id", "toolCallId", "tool_call_id"; default=""))
    isempty(call_id) && (call_id = _agui_generated_id("toolcall"))

    return Dict{String, Any}(
        "call_id" => call_id,
        "name" => name,
        "arguments" => _agui_json_string(_agui_get(payload, "arguments"; default=Dict{String, Any}())),
    )
end

function _agui_message_to_framework(message::AbstractDict)
    role = lowercase(_agui_string(get(message, "role", "")))
    isempty(role) && error("AG-UI message missing role")

    content = _agui_string(get(message, "content", ""))
    if role == AGUI_ROLE_ASSISTANT
        contents = AgentFramework.Content[]
        !isempty(content) && push!(contents, AgentFramework.text_content(content))

        tool_calls = _agui_get(message, "toolCalls", "tool_calls"; default=Any[])
        tool_calls isa AbstractVector || error("assistant toolCalls must be an array")
        for raw_tool_call in tool_calls
            raw_tool_call isa AbstractDict || error("assistant toolCalls must contain objects")
            tool_call = _agui_tool_call(raw_tool_call)
            push!(contents, AgentFramework.function_call_content(
                tool_call["call_id"],
                tool_call["name"],
                tool_call["arguments"],
            ))
        end

        isempty(contents) && return nothing
        return AgentFramework.Message(:assistant, contents)
    elseif role == AGUI_ROLE_TOOL
        tool_call_id = _agui_string(_agui_get(message, "toolCallId", "tool_call_id"; default=""))
        isempty(tool_call_id) && error("tool message missing toolCallId")
        return AgentFramework.Message(:tool, [AgentFramework.function_result_content(tool_call_id, content)])
    end

    return AgentFramework.Message(_agui_framework_role(role), content)
end

function agui_messages_to_framework(messages)::Vector{AgentFramework.Message}
    converted = AgentFramework.Message[]
    for raw_message in messages
        raw_message isa AbstractDict || error("AG-UI messages must be objects")
        message = _agui_message_to_framework(raw_message)
        message === nothing || push!(converted, message)
    end
    return converted
end

function split_agui_messages(messages::Vector{AgentFramework.Message})
    isempty(messages) && error("AG-UI request must include at least one message")

    current_message = last(messages)
    current_message.role == :user || error("AG-UI messages must end with a user message")

    history_messages = length(messages) > 1 ? messages[1:(end - 1)] : AgentFramework.Message[]
    return history_messages, current_message
end

function format_agui_sse(data)
    return "data: $(JSON3.write(data))\n\n"
end

function write_agui_sse(writer, data)
    write(writer.io, format_agui_sse(data))
    flush(writer.io)
end

function write_agui_event(writer, event_type::AbstractString, payload::AbstractDict=Dict{String, Any}())
    event = Dict{String, Any}("type" => String(event_type))
    for (key, value) in pairs(payload)
        event[String(key)] = value
    end
    write_agui_sse(writer, event)
end

function write_agui_error(writer, message::AbstractString; code::Union{Nothing, AbstractString}=nothing)
    payload = Dict{String, Any}("message" => String(message))
    code !== nothing && (payload["code"] = String(code))
    write_agui_event(writer, AGUI_EVENT_RUN_ERROR, payload)
end
