const AF_SESSION_ID_OPTION_KEY = "_agentframework_session_id"
const AF_THREAD_ID_OPTION_KEY = "_agentframework_thread_id"
const AF_INPUT_MESSAGES_OPTION_KEY = "_agentframework_input_messages"

function _materialize_json(value)
    if value isa AbstractDict
        return Dict{String, Any}(String(k) => _materialize_json(v) for (k, v) in pairs(value))
    elseif value isa AbstractVector
        return Any[_materialize_json(v) for v in value]
    end
    return value
end

function _parse_json_line(line::AbstractString)
    stripped = strip(String(line))
    isempty(stripped) && return nothing
    try
        raw = JSON3.read(stripped)
        raw_dict = _materialize_json(raw)
        raw_dict isa AbstractDict || throw(ChatClientInvalidResponseError("Expected JSON object output, got $(typeof(raw_dict))."))
        return Dict{String, Any}(String(k) => v for (k, v) in pairs(raw_dict))
    catch err
        throw(ChatClientInvalidResponseError("Failed to parse CLI JSON output: $(stripped)"))
    end
end

function _parse_json_lines(stdout::AbstractString)
    events = Dict{String, Any}[]
    for line in split(String(stdout), '\n')
        parsed = _parse_json_line(line)
        parsed === nothing || push!(events, parsed)
    end
    return events
end

function _provider_overrides(options::ChatOptions, key::AbstractString)
    raw = get(options.additional, String(key), nothing)
    raw === nothing && return Dict{String, Any}()
    raw isa AbstractDict || throw(ChatClientInvalidRequestError("ChatOptions.additional[\"$(key)\"] must be a dictionary."))
    return Dict{String, Any}(String(k) => _materialize_json(v) for (k, v) in pairs(raw))
end

_runtime_session_id(options::ChatOptions) = get(options.additional, AF_SESSION_ID_OPTION_KEY, nothing)
_runtime_thread_id(options::ChatOptions) = get(options.additional, AF_THREAD_ID_OPTION_KEY, nothing)

function _runtime_input_messages(options::ChatOptions)
    raw = get(options.additional, AF_INPUT_MESSAGES_OPTION_KEY, nothing)
    raw === nothing && return nothing
    raw isa Vector{Message} || throw(ChatClientInvalidRequestError("ChatOptions.additional[\"$(AF_INPUT_MESSAGES_OPTION_KEY)\"] must be a Vector{Message}."))
    return raw
end

function _coerce_string_vector(value, field_name::AbstractString)
    value === nothing && return String[]
    value isa AbstractString && return [String(value)]
    value isa AbstractVector || throw(ChatClientInvalidRequestError("$(field_name) must be a string or vector of strings."))
    return String[String(item) for item in value]
end

function _coerce_string_dict(value, field_name::AbstractString)
    value === nothing && return Dict{String, String}()
    value isa AbstractDict || throw(ChatClientInvalidRequestError("$(field_name) must be a dictionary of strings."))
    return Dict{String, String}(String(k) => String(v) for (k, v) in pairs(value))
end

function _coerce_bool(value, field_name::AbstractString)
    value isa Bool || throw(ChatClientInvalidRequestError("$(field_name) must be a boolean."))
    return value
end

function _json_string(value)
    return JSON3.write(_materialize_json(value))
end

function _make_cmd(args::Vector{String}, cwd::Union{Nothing, String}, env::Dict{String, String})
    cmd = cwd === nothing ? Cmd(args) : Cmd(args; dir = cwd)
    isempty(env) && return cmd
    merged_env = Dict{String, String}(String(k) => String(v) for (k, v) in ENV)
    merge!(merged_env, env)
    return setenv(cmd, merged_env)
end

function _default_capture_command(cmd::Cmd)
    stdout_buffer = IOBuffer()
    stderr_buffer = IOBuffer()
    process = run(pipeline(ignorestatus(cmd), stdout = stdout_buffer, stderr = stderr_buffer))
    return (
        stdout = String(take!(stdout_buffer)),
        stderr = String(take!(stderr_buffer)),
        exitcode = process.exitcode,
    )
end

function _default_stream_command(cmd::Cmd, on_line::Function)
    stdout_pipe = Pipe()
    stderr_pipe = Pipe()
    stderr_buffer = IOBuffer()
    stderr_task = nothing
    process = nothing

    try
        process = run(pipeline(ignorestatus(cmd), stdout = stdout_pipe, stderr = stderr_pipe), wait = false)
        close(stdout_pipe.in)
        close(stderr_pipe.in)

        stderr_task = @async begin
            for line in eachline(stderr_pipe)
                println(stderr_buffer, line)
            end
        end

        for line in eachline(stdout_pipe)
            on_line(line)
        end

        wait(process)
        wait(stderr_task)
        return (
            stderr = String(take!(stderr_buffer)),
            exitcode = process.exitcode,
        )
    finally
        try
            close(stdout_pipe)
        catch
        end
        try
            close(stderr_pipe)
        catch
        end
    end
end

function _ensure_supported_chat_options(provider_name::AbstractString, options::ChatOptions; supports_response_format::Bool = false)
    options.temperature === nothing || throw(ChatClientInvalidRequestError("$(provider_name) does not support temperature overrides."))
    options.top_p === nothing || throw(ChatClientInvalidRequestError("$(provider_name) does not support top_p overrides."))
    options.max_tokens === nothing || throw(ChatClientInvalidRequestError("$(provider_name) does not support max_tokens overrides."))
    options.stop === nothing || throw(ChatClientInvalidRequestError("$(provider_name) does not support stop sequence overrides."))
    options.tool_choice === nothing || throw(ChatClientInvalidRequestError("$(provider_name) does not support tool_choice overrides."))
    isempty(options.tools) || throw(ChatClientInvalidRequestError("$(provider_name) does not yet bridge Julia FunctionTools into provider-native tools."))
    if !supports_response_format && options.response_format !== nothing
        throw(ChatClientInvalidRequestError("$(provider_name) does not support structured output requests."))
    end
    return nothing
end

function _content_prompt_fragments(content::Content)
    if is_text(content)
        text = get_text(content)
        return isempty(text) ? String[] : [text]
    elseif is_reasoning(content)
        text = get_text(content)
        return isempty(text) ? String[] : ["[reasoning] " * text]
    elseif is_function_call(content)
        name = something(content.name, "tool")
        arguments = something(content.arguments, "")
        return ["[tool call $(name)] $(arguments)"]
    elseif is_function_result(content)
        rendered = if content.result isa AbstractString
            content.result
        elseif content.result !== nothing
            JSON3.write(_materialize_json(content.result))
        else
            ""
        end
        name = something(content.name, "tool")
        return ["[tool result $(name)] $(rendered)"]
    elseif content.message !== nothing
        return ["[message] " * String(content.message)]
    elseif content.uri !== nothing
        return ["[uri] " * String(content.uri)]
    end

    text = get_text(content)
    return isempty(text) ? String[] : [text]
end

function _format_prompt_message(message::Message)
    fragments = String[]
    for content in message.contents
        append!(fragments, _content_prompt_fragments(content))
    end
    filtered = [fragment for fragment in fragments if !isempty(strip(fragment))]
    isempty(filtered) && return ""
    return uppercasefirst(String(message.role)) * ": " * join(filtered, "\n")
end

function _format_prompt(messages::Vector{Message})
    rendered = String[]
    for message in messages
        line = _format_prompt_message(message)
        isempty(strip(line)) || push!(rendered, line)
    end
    return join(rendered, "\n\n")
end

function _select_prompt_messages(messages::Vector{Message}, options::ChatOptions)
    thread_id = _runtime_thread_id(options)
    input_messages = _runtime_input_messages(options)
    if thread_id !== nothing && input_messages !== nothing && !isempty(input_messages)
        return input_messages
    end
    return messages
end

function _error_update(message::AbstractString; conversation_id = nothing, raw = nothing)
    return ChatResponseUpdate(
        role = :assistant,
        contents = [error_content(message)],
        finish_reason = FINISH_ERROR,
        conversation_id = conversation_id === nothing ? nothing : String(conversation_id),
        raw_representation = raw,
    )
end

function _usage_details_from_claude(raw_usage)
    raw_usage === nothing && return nothing
    usage_dict = raw_usage isa AbstractDict ? Dict{String, Any}(String(k) => v for (k, v) in pairs(raw_usage)) : Dict{String, Any}()
    additional = Dict{String, Int}()
    for key in ("cache_creation_input_tokens", "cache_read_input_tokens")
        value = get(usage_dict, key, nothing)
        value isa Integer && (additional[key] = Int(value))
    end

    server_tool_use = get(usage_dict, "server_tool_use", nothing)
    if server_tool_use isa AbstractDict
        for (key, value) in pairs(server_tool_use)
            value isa Integer && (additional["server_tool_use." * String(key)] = Int(value))
        end
    end

    input_tokens = get(usage_dict, "input_tokens", nothing)
    output_tokens = get(usage_dict, "output_tokens", nothing)
    total_tokens = if input_tokens isa Integer || output_tokens isa Integer
        something(input_tokens, 0) + something(output_tokens, 0)
    else
        nothing
    end

    return UsageDetails(
        input_tokens = input_tokens isa Integer ? Int(input_tokens) : nothing,
        output_tokens = output_tokens isa Integer ? Int(output_tokens) : nothing,
        total_tokens = total_tokens,
        additional = additional,
    )
end

function _usage_details_from_copilot(output_tokens, raw_usage)
    additional = Dict{String, Int}()
    if raw_usage isa AbstractDict
        usage_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(raw_usage))
        premium = get(usage_dict, "premiumRequests", nothing)
        premium isa Integer && (additional["premium_requests"] = Int(premium))
        total_duration = get(usage_dict, "totalApiDurationMs", nothing)
        total_duration isa Integer && (additional["total_api_duration_ms"] = Int(total_duration))
        session_duration = get(usage_dict, "sessionDurationMs", nothing)
        session_duration isa Integer && (additional["session_duration_ms"] = Int(session_duration))
    end

    output_tokens isa Integer || return isempty(additional) ? nothing : UsageDetails(additional = additional)
    return UsageDetails(output_tokens = Int(output_tokens), additional = additional)
end
