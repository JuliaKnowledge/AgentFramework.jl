Base.@kwdef mutable struct ClaudeCodeChatClient <: AbstractChatClient
    model::String = get(ENV, "CLAUDE_AGENT_MODEL", "")
    cli_path::String = get(ENV, "CLAUDE_AGENT_CLI_PATH", "claude")
    cwd::Union{Nothing, String} = nothing
    agent::Union{Nothing, String} = nothing
    permission_mode::Union{Nothing, String} = nothing
    max_turns::Union{Nothing, Int} = nothing
    max_budget_usd::Union{Nothing, Float64} = nothing
    effort::Union{Nothing, String} = nothing
    add_dirs::Vector{String} = String[]
    available_tools::Vector{String} = String[]
    allowed_tools::Vector{String} = String[]
    disallowed_tools::Vector{String} = String[]
    mcp_config::Vector{String} = String[]
    settings::Union{Nothing, String} = nothing
    append_system_prompt::Union{Nothing, String} = nothing
    cli_args::Vector{String} = String[]
    env::Dict{String, String} = Dict{String, String}()
    capture_runner::Function = _default_capture_command
    stream_runner::Function = _default_stream_command
end

function Base.show(io::IO, client::ClaudeCodeChatClient)
    model = isempty(client.model) ? "default" : client.model
    print(io, "ClaudeCodeChatClient(\"", model, "\")")
end

AgentFramework.streaming_capability(::Type{ClaudeCodeChatClient}) = HasStreaming()
AgentFramework.code_interpreter_capability(::Type{ClaudeCodeChatClient}) = HasCodeInterpreter()
AgentFramework.file_search_capability(::Type{ClaudeCodeChatClient}) = HasFileSearch()
AgentFramework.web_search_capability(::Type{ClaudeCodeChatClient}) = HasWebSearch()
AgentFramework.structured_output_capability(::Type{ClaudeCodeChatClient}) = HasStructuredOutput()

function _claude_effective_values(client::ClaudeCodeChatClient, options::ChatOptions)
    overrides = _provider_overrides(options, "claude_code")
    env = merge(client.env, _coerce_string_dict(get(overrides, "env", nothing), "claude_code.env"))
    return (
        overrides = overrides,
        model = options.model !== nothing ? options.model : String(get(overrides, "model", client.model)),
        cwd = get(overrides, "cwd", client.cwd),
        agent = get(overrides, "agent", client.agent),
        permission_mode = get(overrides, "permission_mode", client.permission_mode),
        max_turns = get(overrides, "max_turns", client.max_turns),
        max_budget_usd = get(overrides, "max_budget_usd", client.max_budget_usd),
        effort = get(overrides, "effort", client.effort),
        add_dirs = haskey(overrides, "add_dirs") ? _coerce_string_vector(overrides["add_dirs"], "claude_code.add_dirs") : client.add_dirs,
        available_tools = haskey(overrides, "available_tools") ? _coerce_string_vector(overrides["available_tools"], "claude_code.available_tools") : client.available_tools,
        allowed_tools = haskey(overrides, "allowed_tools") ? _coerce_string_vector(overrides["allowed_tools"], "claude_code.allowed_tools") : client.allowed_tools,
        disallowed_tools = haskey(overrides, "disallowed_tools") ? _coerce_string_vector(overrides["disallowed_tools"], "claude_code.disallowed_tools") : client.disallowed_tools,
        mcp_config = haskey(overrides, "mcp_config") ? _coerce_string_vector(overrides["mcp_config"], "claude_code.mcp_config") : client.mcp_config,
        settings = get(overrides, "settings", client.settings),
        append_system_prompt = get(overrides, "append_system_prompt", client.append_system_prompt),
        cli_args = vcat(client.cli_args, _coerce_string_vector(get(overrides, "cli_args", nothing), "claude_code.cli_args")),
        env = env,
    )
end

function _claude_finish_reason(value)
    value === nothing && return STOP
    stop_reason = String(value)
    if stop_reason in ("stop", "end_turn", "stop_sequence")
        return STOP
    elseif stop_reason in ("max_tokens", "model_context_window_exceeded")
        return LENGTH
    elseif stop_reason in ("tool_use", "pause_turn")
        return TOOL_CALLS
    end
    return FINISH_ERROR
end

function _claude_command(client::ClaudeCodeChatClient, messages::Vector{Message}, options::ChatOptions; stream::Bool = false)
    _ensure_supported_chat_options("ClaudeCodeChatClient", options; supports_response_format = true)
    values = _claude_effective_values(client, options)
    prompt = _format_prompt(_select_prompt_messages(messages, options))
    isempty(strip(prompt)) && throw(ChatClientInvalidRequestError("ClaudeCodeChatClient received an empty prompt."))

    args = String[client.cli_path, "-p", prompt, "--output-format", stream ? "stream-json" : "json"]
    stream && push!(args, "--verbose")
    stream && push!(args, "--include-partial-messages")
    !isempty(values.model) && append!(args, ["--model", values.model])
    values.agent !== nothing && append!(args, ["--agent", String(values.agent)])
    values.permission_mode !== nothing && append!(args, ["--permission-mode", String(values.permission_mode)])
    values.max_turns !== nothing && append!(args, ["--max-turns", string(values.max_turns)])
    values.max_budget_usd !== nothing && append!(args, ["--max-budget-usd", string(values.max_budget_usd)])
    values.effort !== nothing && append!(args, ["--effort", String(values.effort)])
    values.settings !== nothing && append!(args, ["--settings", String(values.settings)])
    values.append_system_prompt !== nothing && append!(args, ["--append-system-prompt", String(values.append_system_prompt)])

    for directory in values.add_dirs
        append!(args, ["--add-dir", directory])
    end
    !isempty(values.available_tools) && append!(args, ["--tools", join(values.available_tools, ",")])
    !isempty(values.allowed_tools) && append!(args, ["--allowed-tools", join(values.allowed_tools, ",")])
    !isempty(values.disallowed_tools) && append!(args, ["--disallowed-tools", join(values.disallowed_tools, ",")])
    for cfg in values.mcp_config
        append!(args, ["--mcp-config", cfg])
    end

    if options.response_format !== nothing
        options.response_format isa AbstractDict || throw(ChatClientInvalidRequestError("ClaudeCodeChatClient expects response_format to be a JSON schema dictionary."))
        append!(args, ["--json-schema", _json_string(options.response_format)])
    end

    thread_id = _runtime_thread_id(options)
    if thread_id !== nothing
        append!(args, ["--resume", String(thread_id)])
    end
    append!(args, values.cli_args)

    cwd = values.cwd === nothing ? client.cwd : String(values.cwd)
    return _make_cmd(args, cwd, values.env)
end

function _claude_result_model_id(payload::Dict{String, Any}, fallback::String)
    model_usage = get(payload, "modelUsage", nothing)
    if model_usage isa AbstractDict && !isempty(model_usage)
        return String(first(keys(model_usage)))
    end
    return isempty(fallback) ? nothing : fallback
end

function _claude_response_from_payload(payload::Dict{String, Any}, model_hint::String)
    if get(payload, "is_error", false)
        throw(ChatClientError("Claude Code CLI returned an error result."))
    end

    result_value = get(payload, "result", "")
    rendered = result_value isa AbstractString ? String(result_value) : JSON3.write(_materialize_json(result_value))
    contents = isempty(rendered) ? Content[] : [text_content(rendered)]
    isempty(contents) && push!(contents, text_content(""))

    additional_properties = Dict{String, Any}("result" => _materialize_json(result_value))
    for key in ("permission_denials", "terminal_reason", "fast_mode_state")
        haskey(payload, key) && (additional_properties[key] = _materialize_json(payload[key]))
    end

    return ChatResponse(
        messages = [Message(:assistant, contents)],
        response_id = get(payload, "uuid", nothing),
        conversation_id = get(payload, "session_id", nothing),
        model_id = _claude_result_model_id(payload, model_hint),
        finish_reason = _claude_finish_reason(get(payload, "stop_reason", nothing)),
        usage_details = _usage_details_from_claude(get(payload, "usage", nothing)),
        additional_properties = additional_properties,
        raw_representation = payload,
    )
end

function get_response(client::ClaudeCodeChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    cmd = _claude_command(client, messages, options; stream = false)
    result = client.capture_runner(cmd)
    result.exitcode == 0 || throw(ChatClientError("Claude Code CLI exited with code $(result.exitcode): $(strip(String(result.stderr)))"))
    payload = _parse_json_line(result.stdout)
    payload === nothing && throw(ChatClientInvalidResponseError("Claude Code CLI returned no JSON payload."))
    values = _claude_effective_values(client, options)
    return _claude_response_from_payload(payload, isempty(values.model) ? client.model : values.model)
end

function get_response_streaming(client::ClaudeCodeChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    cmd = _claude_command(client, messages, options; stream = true)
    channel = Channel{ChatResponseUpdate}(32)

    Threads.@spawn begin
        saw_text_delta = false
        saw_reasoning_delta = false
        last_response_id = nothing

        try
            result = client.stream_runner(cmd, line -> begin
                payload = _parse_json_line(line)
                payload === nothing && return
                payload_type = String(get(payload, "type", ""))

                if payload_type == "stream_event"
                    event = get(payload, "event", Dict{String, Any}())
                    event isa AbstractDict || return
                    event_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(event))
                    event_type = String(get(event_dict, "type", ""))

                    if event_type == "message_start"
                        message = get(event_dict, "message", Dict{String, Any}())
                        if message isa AbstractDict
                            message_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(message))
                            last_response_id = get(message_dict, "id", last_response_id)
                        end
                    elseif event_type == "content_block_delta"
                        delta = get(event_dict, "delta", Dict{String, Any}())
                        delta isa AbstractDict || return
                        delta_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(delta))
                        delta_type = String(get(delta_dict, "type", ""))
                        if delta_type == "text_delta"
                            saw_text_delta = true
                            text = get(delta_dict, "text", nothing)
                            if text isa AbstractString && !isempty(text)
                                put!(channel, ChatResponseUpdate(
                                    role = :assistant,
                                    contents = [text_content(text)],
                                    response_id = last_response_id === nothing ? nothing : String(last_response_id),
                                    conversation_id = get(payload, "session_id", nothing),
                                    raw_representation = payload,
                                ))
                            end
                        elseif delta_type == "thinking_delta"
                            saw_reasoning_delta = true
                            thinking = get(delta_dict, "thinking", nothing)
                            if thinking isa AbstractString && !isempty(thinking)
                                put!(channel, ChatResponseUpdate(
                                    role = :assistant,
                                    contents = [reasoning_content(thinking)],
                                    response_id = last_response_id === nothing ? nothing : String(last_response_id),
                                    conversation_id = get(payload, "session_id", nothing),
                                    raw_representation = payload,
                                ))
                            end
                        end
                    end
                elseif payload_type == "assistant"
                    message = get(payload, "message", Dict{String, Any}())
                    message isa AbstractDict || return
                    message_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(message))
                    last_response_id = get(message_dict, "id", last_response_id)
                    blocks = get(message_dict, "content", Any[])
                    contents = Content[]
                    if blocks isa AbstractVector
                        for block in blocks
                            block isa AbstractDict || continue
                            block_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(block))
                            block_type = String(get(block_dict, "type", ""))
                            if block_type == "text" && !saw_text_delta
                                text = get(block_dict, "text", nothing)
                                text isa AbstractString && !isempty(text) && push!(contents, text_content(text))
                            elseif block_type == "thinking" && !saw_reasoning_delta
                                thinking = get(block_dict, "thinking", nothing)
                                thinking isa AbstractString && !isempty(thinking) && push!(contents, reasoning_content(thinking))
                            end
                        end
                    end
                    if !isempty(contents)
                        put!(channel, ChatResponseUpdate(
                            role = :assistant,
                            contents = contents,
                            response_id = last_response_id === nothing ? nothing : String(last_response_id),
                            conversation_id = get(payload, "session_id", nothing),
                            raw_representation = payload,
                        ))
                    end
                elseif payload_type == "result"
                    if get(payload, "is_error", false)
                        put!(channel, _error_update(
                            "Claude Code CLI returned an error result.";
                            conversation_id = get(payload, "session_id", nothing),
                            raw = payload,
                        ))
                    else
                        put!(channel, ChatResponseUpdate(
                            finish_reason = _claude_finish_reason(get(payload, "stop_reason", nothing)),
                            response_id = last_response_id === nothing ? get(payload, "uuid", nothing) : String(last_response_id),
                            conversation_id = get(payload, "session_id", nothing),
                            usage_details = _usage_details_from_claude(get(payload, "usage", nothing)),
                            model_id = _claude_result_model_id(payload, client.model),
                            raw_representation = payload,
                        ))
                    end
                end
            end)

            if result.exitcode != 0
                put!(channel, _error_update("Claude Code CLI exited with code $(result.exitcode): $(strip(String(result.stderr)))"))
            end
        catch err
            @error "Claude Code streaming error" exception = (err, catch_backtrace())
            put!(channel, _error_update("Claude Code streaming failed: $(err)"; raw = sprint(showerror, err)))
        finally
            close(channel)
        end
    end

    return channel
end
