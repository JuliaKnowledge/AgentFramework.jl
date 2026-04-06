"""
SSE (Server-Sent Events) streaming support for the DevUI.
"""

"""
    SSEWriter

Wraps an IO stream for writing SSE-formatted events.
"""
struct SSEWriter
    io::IO
end

"""
    write_sse(writer, event, data)

Write an SSE event: `event: {event}\\ndata: {json}\\n\\n`
"""
function write_sse(writer::SSEWriter, event::String, data)
    json_str = JSON3.write(data)
    write(writer.io, "event: $event\ndata: $json_str\n\n")
    flush(writer.io)
end

"""
    write_sse_done(writer)

Write the SSE stream terminator: `data: [DONE]\\n\\n`
"""
function write_sse_done(writer::SSEWriter)
    write(writer.io, "data: [DONE]\n\n")
    flush(writer.io)
end

"""
    format_sse(event, data) → String

Format an SSE event as a string (for testing or buffered writes).
"""
function format_sse(event::String, data)
    json_str = JSON3.write(data)
    return "event: $event\ndata: $json_str\n\n"
end

"""
    format_sse_done() → String

Format the SSE stream terminator as a string.
"""
function format_sse_done()
    return "data: [DONE]\n\n"
end

function _save_history_messages!(agent::AgentFramework.Agent,
                                 session::AgentFramework.AgentSession,
                                 messages::Vector{AgentFramework.Message})
    isempty(messages) && return
    for provider in agent.context_providers
        if provider isa AgentFramework.AbstractHistoryProvider
            AgentFramework.save_messages!(provider, session.id, messages)
        end
    end
end

function _serialize_tool_result(result)::String
    if result isa AbstractString
        return String(result)
    elseif result !== nothing
        return JSON3.write(result)
    else
        return ""
    end
end

function _framework_message_parts(message::AgentFramework.Message)
    tool_calls = Dict{String, Any}[]
    tool_results = Dict{String, Any}[]
    for content in message.contents
        if AgentFramework.is_function_call(content)
            push!(tool_calls, Dict{String, Any}(
                "call_id" => something(content.call_id, ""),
                "name" => something(content.name, ""),
                "arguments" => something(content.arguments, ""),
            ))
        elseif AgentFramework.is_function_result(content)
            push!(tool_results, Dict{String, Any}(
                "call_id" => something(content.call_id, ""),
                "result" => _serialize_tool_result(content.result),
            ))
        end
    end
    return AgentFramework.get_text(message), tool_calls, tool_results
end

function _response_tool_parts(response::AgentFramework.AgentResponse)
    tool_calls = Dict{String, Any}[]
    tool_results = Dict{String, Any}[]
    for message in response.messages
        _, message_tool_calls, message_tool_results = _framework_message_parts(message)
        append!(tool_calls, message_tool_calls)
        append!(tool_results, message_tool_results)
    end
    return tool_calls, tool_results
end

"""
    stream_agent_response(writer, agent, message, session)

Run agent in streaming mode and convert updates to SSE events.

Events emitted:
- `message.start` — when the response begins
- `message.delta` — text chunks
- `tool_call.start` — tool invocation started
- `tool_call.result` — tool result received
- `message.complete` — when the response is finished
- `error` — on error
"""
function stream_agent_response(writer::SSEWriter,
                               agent::AgentFramework.Agent,
                               message::String,
                               session::Union{Nothing, AgentFramework.AgentSession})
    accumulated_text = IOBuffer()
    try
        write_sse(writer, "message.start", Dict("role" => "assistant"))

        stream = AgentFramework.run_agent_streaming(agent, message; session=session)

        for update in stream
            _process_streaming_update(writer, update, accumulated_text)
        end

        final_resp = AgentFramework.get_final_response(stream)

        # Save conversation history to the agent's history providers
        # (run_agent_streaming doesn't call after_run! so we do it manually)
        if session !== nothing && final_resp !== nothing
            _save_streaming_history(agent, session, message, final_resp)
        end

        complete_data = Dict{String, Any}(
            "role" => "assistant",
            "finish_reason" => final_resp !== nothing && final_resp.finish_reason !== nothing ?
                string(final_resp.finish_reason) : "stop",
        )
        if final_resp !== nothing && final_resp.usage_details !== nothing
            usage = final_resp.usage_details
            complete_data["usage"] = Dict{String, Any}(
                "input_tokens" => something(usage.input_tokens, 0),
                "output_tokens" => something(usage.output_tokens, 0),
                "total_tokens" => something(usage.total_tokens, 0),
            )
        end
        write_sse(writer, "message.complete", complete_data)
        write_sse_done(writer)
    catch e
        @error "Streaming error" exception=(e, catch_backtrace())
        write_sse(writer, "error", Dict("error" => sprint(showerror, e)))
        write_sse_done(writer)
    end
    return String(take!(accumulated_text))
end

"""
    _save_streaming_history(agent, session, user_message, response)

Manually save conversation turn to the agent's history providers after streaming.
This is needed because `run_agent_streaming` doesn't call `after_run!` on context providers.
"""
function _save_streaming_history(agent::AgentFramework.Agent,
                                 session::AgentFramework.AgentSession,
                                 user_message::String,
                                 response::AgentFramework.AgentResponse)
    to_save = AgentFramework.Message[]
    push!(to_save, AgentFramework.Message(role=:user,
        contents=[AgentFramework.text_content(user_message)]))
    append!(to_save, response.messages)
    _save_history_messages!(agent, session, to_save)
end

"""
Process a single streaming update and emit SSE events.
"""
function _process_streaming_update(writer::SSEWriter, update::AgentFramework.AgentResponseUpdate,
                                   accumulated_text::IOBuffer)
    for content in update.contents
        if AgentFramework.is_text(content)
            text = AgentFramework.get_text(content)
            if !isempty(text)
                write(accumulated_text, text)
                write_sse(writer, "message.delta", Dict(
                    "role" => "assistant",
                    "content" => text,
                ))
            end
        elseif content.type == AgentFramework.TEXT_REASONING
            text = AgentFramework.get_text(content)
            if !isempty(text)
                write_sse(writer, "message.thinking", Dict(
                    "role" => "assistant",
                    "content" => text,
                ))
            end
        elseif AgentFramework.is_function_call(content)
            write_sse(writer, "tool_call.start", Dict(
                "call_id" => something(content.call_id, ""),
                "name" => something(content.name, ""),
                "arguments" => something(content.arguments, ""),
            ))
        elseif AgentFramework.is_function_result(content)
            write_sse(writer, "tool_call.result", Dict(
                "call_id" => something(content.call_id, ""),
                "result" => _serialize_tool_result(content.result),
            ))
        end
    end
end

function _agui_parse_json_payload(data::AbstractString)
    stripped = strip(String(data))
    isempty(stripped) && return nothing
    try
        return JSON3.read(stripped)
    catch
        return nothing
    end
end

function _process_streaming_update_agui(writer::SSEWriter,
                                        update::AgentFramework.AgentResponseUpdate,
                                        accumulated_text::IOBuffer,
                                        message_id::AbstractString,
                                        text_started::Bool)
    started = text_started
    role = update.role === nothing ? AGUI_ROLE_ASSISTANT : string(update.role)

    for content in update.contents
        if AgentFramework.is_text(content)
            text = AgentFramework.get_text(content)
            if !isempty(text)
                if !started
                    write_agui_event(writer, AGUI_EVENT_TEXT_MESSAGE_START, Dict(
                        "messageId" => String(message_id),
                        "role" => role,
                    ))
                    started = true
                end

                write(accumulated_text, text)
                write_agui_event(writer, AGUI_EVENT_TEXT_MESSAGE_CONTENT, Dict(
                    "messageId" => String(message_id),
                    "delta" => text,
                ))
            end
        elseif content.type == AgentFramework.DATA
            payload = _agui_parse_json_payload(something(content.text, ""))
            if payload !== nothing
                media_type = lowercase(something(content.media_type, ""))
                if media_type == "application/json"
                    write_agui_event(writer, AGUI_EVENT_STATE_SNAPSHOT, Dict("snapshot" => payload))
                elseif media_type == "application/json-patch+json" || media_type == "application/json-patch"
                    write_agui_event(writer, AGUI_EVENT_STATE_DELTA, Dict("delta" => payload))
                end
            end
        elseif AgentFramework.is_function_call(content)
            tool_call_id = something(content.call_id, _agui_generated_id("toolcall"))
            write_agui_event(writer, AGUI_EVENT_TOOL_CALL_START, Dict(
                "toolCallId" => tool_call_id,
                "toolCallName" => something(content.name, ""),
                "parentMessageId" => String(message_id),
            ))
            write_agui_event(writer, AGUI_EVENT_TOOL_CALL_ARGS, Dict(
                "toolCallId" => tool_call_id,
                "delta" => something(content.arguments, "{}"),
            ))
            write_agui_event(writer, AGUI_EVENT_TOOL_CALL_END, Dict("toolCallId" => tool_call_id))
        elseif AgentFramework.is_function_result(content)
            write_agui_event(writer, AGUI_EVENT_TOOL_CALL_RESULT, Dict(
                "messageId" => String(message_id),
                "toolCallId" => something(content.call_id, ""),
                "content" => _serialize_tool_result(content.result),
                "role" => AGUI_ROLE_TOOL,
            ))
        end
    end

    return started
end

function stream_agent_response_agui(writer::SSEWriter,
                                    agent::AgentFramework.Agent,
                                    message::String,
                                    session::Union{Nothing, AgentFramework.AgentSession};
                                    thread_id::AbstractString,
                                    run_id::AbstractString)
    accumulated_text = IOBuffer()
    message_id = _agui_generated_id("msg")
    text_started = false
    final_response = nothing

    try
        write_agui_event(writer, AGUI_EVENT_RUN_STARTED, Dict(
            "threadId" => String(thread_id),
            "runId" => String(run_id),
        ))

        stream = AgentFramework.run_agent_streaming(agent, message; session=session)
        for update in stream
            text_started = _process_streaming_update_agui(writer, update, accumulated_text, message_id, text_started)
        end

        final_response = AgentFramework.get_final_response(stream)
        if session !== nothing && final_response !== nothing
            _save_streaming_history(agent, session, message, final_response)
        end

        if text_started
            write_agui_event(writer, AGUI_EVENT_TEXT_MESSAGE_END, Dict("messageId" => message_id))
        end

        write_agui_event(writer, AGUI_EVENT_RUN_FINISHED, Dict(
            "threadId" => String(thread_id),
            "runId" => String(run_id),
        ))
    catch e
        @error "AG-UI streaming error" exception=(e, catch_backtrace())
        if text_started
            write_agui_event(writer, AGUI_EVENT_TEXT_MESSAGE_END, Dict("messageId" => message_id))
        end
        write_agui_error(writer, "Agent execution error: $(sprint(showerror, e))";
                         code=string(nameof(typeof(e))))
    end

    return (text=String(take!(accumulated_text)), response=final_response)
end

function stream_workflow_response_agui(writer::SSEWriter,
                                       workflow::AgentFramework.Workflow,
                                       message::String;
                                       thread_id::AbstractString,
                                       run_id::AbstractString)
    response_text = ""

    try
        write_agui_event(writer, AGUI_EVENT_RUN_STARTED, Dict(
            "threadId" => String(thread_id),
            "runId" => String(run_id),
        ))

        result = AgentFramework.run_workflow(workflow, message)
        outputs = AgentFramework.get_outputs(result)
        response_text = isempty(outputs) ? "Workflow completed." : string(first(outputs))

        if !isempty(response_text)
            message_id = _agui_generated_id("msg")
            write_agui_event(writer, AGUI_EVENT_TEXT_MESSAGE_START, Dict(
                "messageId" => message_id,
                "role" => AGUI_ROLE_ASSISTANT,
            ))
            write_agui_event(writer, AGUI_EVENT_TEXT_MESSAGE_CONTENT, Dict(
                "messageId" => message_id,
                "delta" => response_text,
            ))
            write_agui_event(writer, AGUI_EVENT_TEXT_MESSAGE_END, Dict("messageId" => message_id))
        end

        write_agui_event(writer, AGUI_EVENT_RUN_FINISHED, Dict(
            "threadId" => String(thread_id),
            "runId" => String(run_id),
        ))
    catch e
        @error "AG-UI workflow error" exception=(e, catch_backtrace())
        write_agui_error(writer, "Workflow execution error: $(sprint(showerror, e))";
                         code=string(nameof(typeof(e))))
    end

    return response_text
end
