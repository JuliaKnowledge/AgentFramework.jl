Base.@kwdef mutable struct GitHubCopilotChatClient <: AbstractChatClient
    model::String = get(ENV, "GITHUB_COPILOT_MODEL", "")
    cli_path::String = get(ENV, "GITHUB_COPILOT_CLI_PATH", "copilot")
    cwd::Union{Nothing, String} = nothing
    agent::Union{Nothing, String} = nothing
    reasoning_effort::Union{Nothing, String} = nothing
    allow_all_tools::Bool = true
    allow_all_paths::Bool = false
    allow_all_urls::Bool = false
    no_ask_user::Bool = true
    autopilot::Bool = false
    add_dirs::Vector{String} = String[]
    available_tools::Vector{String} = String[]
    excluded_tools::Vector{String} = String[]
    allow_tools::Vector{String} = String[]
    deny_tools::Vector{String} = String[]
    allow_urls::Vector{String} = String[]
    deny_urls::Vector{String} = String[]
    additional_mcp_config::Vector{String} = String[]
    github_mcp_tools::Vector{String} = String[]
    github_mcp_toolsets::Vector{String} = String[]
    enable_all_github_mcp_tools::Bool = false
    config_dir::Union{Nothing, String} = nothing
    log_dir::Union{Nothing, String} = nothing
    log_level::Union{Nothing, String} = nothing
    cli_args::Vector{String} = String[]
    env::Dict{String, String} = Dict{String, String}()
    capture_runner::Function = _default_capture_command
    stream_runner::Function = _default_stream_command
end

function Base.show(io::IO, client::GitHubCopilotChatClient)
    model = isempty(client.model) ? "default" : client.model
    print(io, "GitHubCopilotChatClient(\"", model, "\")")
end

AgentFramework.streaming_capability(::Type{GitHubCopilotChatClient}) = HasStreaming()
AgentFramework.code_interpreter_capability(::Type{GitHubCopilotChatClient}) = HasCodeInterpreter()
AgentFramework.file_search_capability(::Type{GitHubCopilotChatClient}) = HasFileSearch()
AgentFramework.web_search_capability(::Type{GitHubCopilotChatClient}) = HasWebSearch()

function _copilot_effective_values(client::GitHubCopilotChatClient, options::ChatOptions)
    overrides = _provider_overrides(options, "github_copilot")
    env = merge(client.env, _coerce_string_dict(get(overrides, "env", nothing), "github_copilot.env"))
    return (
        overrides = overrides,
        model = options.model !== nothing ? options.model : String(get(overrides, "model", client.model)),
        cwd = get(overrides, "cwd", client.cwd),
        agent = get(overrides, "agent", client.agent),
        reasoning_effort = get(overrides, "reasoning_effort", client.reasoning_effort),
        allow_all_tools = haskey(overrides, "allow_all_tools") ? _coerce_bool(overrides["allow_all_tools"], "github_copilot.allow_all_tools") : client.allow_all_tools,
        allow_all_paths = haskey(overrides, "allow_all_paths") ? _coerce_bool(overrides["allow_all_paths"], "github_copilot.allow_all_paths") : client.allow_all_paths,
        allow_all_urls = haskey(overrides, "allow_all_urls") ? _coerce_bool(overrides["allow_all_urls"], "github_copilot.allow_all_urls") : client.allow_all_urls,
        no_ask_user = haskey(overrides, "no_ask_user") ? _coerce_bool(overrides["no_ask_user"], "github_copilot.no_ask_user") : client.no_ask_user,
        autopilot = haskey(overrides, "autopilot") ? _coerce_bool(overrides["autopilot"], "github_copilot.autopilot") : client.autopilot,
        add_dirs = haskey(overrides, "add_dirs") ? _coerce_string_vector(overrides["add_dirs"], "github_copilot.add_dirs") : client.add_dirs,
        available_tools = haskey(overrides, "available_tools") ? _coerce_string_vector(overrides["available_tools"], "github_copilot.available_tools") : client.available_tools,
        excluded_tools = haskey(overrides, "excluded_tools") ? _coerce_string_vector(overrides["excluded_tools"], "github_copilot.excluded_tools") : client.excluded_tools,
        allow_tools = haskey(overrides, "allow_tools") ? _coerce_string_vector(overrides["allow_tools"], "github_copilot.allow_tools") : client.allow_tools,
        deny_tools = haskey(overrides, "deny_tools") ? _coerce_string_vector(overrides["deny_tools"], "github_copilot.deny_tools") : client.deny_tools,
        allow_urls = haskey(overrides, "allow_urls") ? _coerce_string_vector(overrides["allow_urls"], "github_copilot.allow_urls") : client.allow_urls,
        deny_urls = haskey(overrides, "deny_urls") ? _coerce_string_vector(overrides["deny_urls"], "github_copilot.deny_urls") : client.deny_urls,
        additional_mcp_config = haskey(overrides, "additional_mcp_config") ? _coerce_string_vector(overrides["additional_mcp_config"], "github_copilot.additional_mcp_config") : client.additional_mcp_config,
        github_mcp_tools = haskey(overrides, "github_mcp_tools") ? _coerce_string_vector(overrides["github_mcp_tools"], "github_copilot.github_mcp_tools") : client.github_mcp_tools,
        github_mcp_toolsets = haskey(overrides, "github_mcp_toolsets") ? _coerce_string_vector(overrides["github_mcp_toolsets"], "github_copilot.github_mcp_toolsets") : client.github_mcp_toolsets,
        enable_all_github_mcp_tools = haskey(overrides, "enable_all_github_mcp_tools") ? _coerce_bool(overrides["enable_all_github_mcp_tools"], "github_copilot.enable_all_github_mcp_tools") : client.enable_all_github_mcp_tools,
        config_dir = get(overrides, "config_dir", client.config_dir),
        log_dir = get(overrides, "log_dir", client.log_dir),
        log_level = get(overrides, "log_level", client.log_level),
        cli_args = vcat(client.cli_args, _coerce_string_vector(get(overrides, "cli_args", nothing), "github_copilot.cli_args")),
        env = env,
    )
end

function _copilot_command(client::GitHubCopilotChatClient, messages::Vector{Message}, options::ChatOptions)
    _ensure_supported_chat_options("GitHubCopilotChatClient", options)
    values = _copilot_effective_values(client, options)
    prompt = _format_prompt(_select_prompt_messages(messages, options))
    isempty(strip(prompt)) && throw(ChatClientInvalidRequestError("GitHubCopilotChatClient received an empty prompt."))

    args = String[client.cli_path, "-p", prompt, "--output-format", "json"]
    !isempty(values.model) && append!(args, ["--model", values.model])
    values.reasoning_effort !== nothing && append!(args, ["--reasoning-effort", String(values.reasoning_effort)])
    values.agent !== nothing && append!(args, ["--agent", String(values.agent)])
    values.allow_all_tools && push!(args, "--allow-all-tools")
    values.allow_all_paths && push!(args, "--allow-all-paths")
    values.allow_all_urls && push!(args, "--allow-all-urls")
    values.no_ask_user && push!(args, "--no-ask-user")
    values.autopilot && push!(args, "--autopilot")
    values.config_dir !== nothing && append!(args, ["--config-dir", String(values.config_dir)])
    values.log_dir !== nothing && append!(args, ["--log-dir", String(values.log_dir)])
    values.log_level !== nothing && append!(args, ["--log-level", String(values.log_level)])

    for directory in values.add_dirs
        append!(args, ["--add-dir", directory])
    end
    !isempty(values.available_tools) && push!(args, "--available-tools=$(join(values.available_tools, ','))")
    !isempty(values.excluded_tools) && push!(args, "--excluded-tools=$(join(values.excluded_tools, ','))")
    !isempty(values.allow_tools) && push!(args, "--allow-tool=$(join(values.allow_tools, ','))")
    !isempty(values.deny_tools) && push!(args, "--deny-tool=$(join(values.deny_tools, ','))")
    !isempty(values.allow_urls) && push!(args, "--allow-url=$(join(values.allow_urls, ','))")
    !isempty(values.deny_urls) && push!(args, "--deny-url=$(join(values.deny_urls, ','))")

    for cfg in values.additional_mcp_config
        append!(args, ["--additional-mcp-config", cfg])
    end
    if values.enable_all_github_mcp_tools
        push!(args, "--enable-all-github-mcp-tools")
    else
        for tool in values.github_mcp_tools
            append!(args, ["--add-github-mcp-tool", tool])
        end
        for toolset in values.github_mcp_toolsets
            append!(args, ["--add-github-mcp-toolset", toolset])
        end
    end

    thread_id = _runtime_thread_id(options)
    thread_id !== nothing && push!(args, "--resume=$(thread_id)")
    append!(args, values.cli_args)

    cwd = values.cwd === nothing ? client.cwd : String(values.cwd)
    return _make_cmd(args, cwd, values.env)
end

function _copilot_response_from_events(events::Vector{Dict{String, Any}}, model_hint::String)
    isempty(events) && throw(ChatClientInvalidResponseError("GitHub Copilot returned no JSON events."))

    response_id = nothing
    session_id = nothing
    saw_message_delta = false
    saw_reasoning_delta = false
    output_tokens = nothing
    raw_usage = nothing
    text_buffer = IOBuffer()
    reasoning_buffer = IOBuffer()

    for event in events
        event_type = String(get(event, "type", ""))
        data = get(event, "data", nothing)
        data_dict = data isa AbstractDict ? Dict{String, Any}(String(k) => v for (k, v) in pairs(data)) : Dict{String, Any}()

        if event_type == "assistant.message_delta"
            saw_message_delta = true
            delta = get(data_dict, "deltaContent", nothing)
            delta isa AbstractString && print(text_buffer, delta)
            response_id = get(data_dict, "messageId", response_id)
        elseif event_type == "assistant.reasoning_delta"
            saw_reasoning_delta = true
            delta = get(data_dict, "deltaContent", nothing)
            delta isa AbstractString && print(reasoning_buffer, delta)
        elseif event_type == "assistant.message"
            response_id = get(data_dict, "messageId", response_id)
            if !saw_message_delta
                content = get(data_dict, "content", nothing)
                content isa AbstractString && print(text_buffer, content)
            end
            if !saw_reasoning_delta
                reasoning = get(data_dict, "reasoningText", nothing)
                reasoning isa AbstractString && print(reasoning_buffer, reasoning)
            end
            token_count = get(data_dict, "outputTokens", nothing)
            token_count isa Integer && (output_tokens = Int(token_count))
        elseif event_type == "result"
            exit_code = get(event, "exitCode", 0)
            exit_code == 0 || throw(ChatClientError("GitHub Copilot CLI exited with code $(exit_code)."))
            session_id = get(event, "sessionId", session_id)
            raw_usage = get(event, "usage", raw_usage)
        elseif occursin("error", event_type)
            throw(ChatClientError("GitHub Copilot returned error event $(event_type)."))
        end
    end

    text = String(take!(text_buffer))
    reasoning = String(take!(reasoning_buffer))
    contents = Content[]
    !isempty(reasoning) && push!(contents, reasoning_content(reasoning))
    !isempty(text) && push!(contents, text_content(text))
    isempty(contents) && push!(contents, text_content(""))

    return ChatResponse(
        messages = [Message(:assistant, contents)],
        response_id = response_id === nothing ? nothing : String(response_id),
        conversation_id = session_id === nothing ? nothing : String(session_id),
        model_id = isempty(model_hint) ? nothing : model_hint,
        finish_reason = STOP,
        usage_details = _usage_details_from_copilot(output_tokens, raw_usage),
        additional_properties = Dict{String, Any}(
            "events" => events,
            "usage" => raw_usage === nothing ? nothing : _materialize_json(raw_usage),
        ),
        raw_representation = events,
    )
end

function get_response(client::GitHubCopilotChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    cmd = _copilot_command(client, messages, options)
    result = client.capture_runner(cmd)
    result.exitcode == 0 || throw(ChatClientError("GitHub Copilot CLI exited with code $(result.exitcode): $(strip(String(result.stderr)))"))
    events = _parse_json_lines(result.stdout)
    values = _copilot_effective_values(client, options)
    return _copilot_response_from_events(events, isempty(values.model) ? client.model : values.model)
end

function get_response_streaming(client::GitHubCopilotChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    cmd = _copilot_command(client, messages, options)
    channel = Channel{ChatResponseUpdate}(32)

    Threads.@spawn begin
        saw_message_delta = false
        saw_reasoning_delta = false
        last_response_id = nothing

        try
            result = client.stream_runner(cmd, line -> begin
                event = _parse_json_line(line)
                event === nothing && return

                event_type = String(get(event, "type", ""))
                data = get(event, "data", nothing)
                data_dict = data isa AbstractDict ? Dict{String, Any}(String(k) => v for (k, v) in pairs(data)) : Dict{String, Any}()

                if event_type == "assistant.reasoning_delta"
                    saw_reasoning_delta = true
                    delta = get(data_dict, "deltaContent", nothing)
                    if delta isa AbstractString && !isempty(delta)
                        put!(channel, ChatResponseUpdate(
                            role = :assistant,
                            contents = [reasoning_content(delta)],
                            raw_representation = event,
                        ))
                    end
                elseif event_type == "assistant.message_delta"
                    saw_message_delta = true
                    last_response_id = get(data_dict, "messageId", last_response_id)
                    delta = get(data_dict, "deltaContent", nothing)
                    if delta isa AbstractString && !isempty(delta)
                        put!(channel, ChatResponseUpdate(
                            role = :assistant,
                            contents = [text_content(delta)],
                            response_id = last_response_id === nothing ? nothing : String(last_response_id),
                            raw_representation = event,
                        ))
                    end
                elseif event_type == "assistant.message"
                    last_response_id = get(data_dict, "messageId", last_response_id)
                    contents = Content[]
                    if !saw_reasoning_delta
                        reasoning = get(data_dict, "reasoningText", nothing)
                        reasoning isa AbstractString && !isempty(reasoning) && push!(contents, reasoning_content(reasoning))
                    end
                    if !saw_message_delta
                        content = get(data_dict, "content", nothing)
                        content isa AbstractString && !isempty(content) && push!(contents, text_content(content))
                    end
                    if !isempty(contents)
                        put!(channel, ChatResponseUpdate(
                            role = :assistant,
                            contents = contents,
                            response_id = last_response_id === nothing ? nothing : String(last_response_id),
                            raw_representation = event,
                        ))
                    end
                elseif event_type == "result"
                    exit_code = get(event, "exitCode", 0)
                    session_id = get(event, "sessionId", nothing)
                    if exit_code == 0
                        put!(channel, ChatResponseUpdate(
                            finish_reason = STOP,
                            response_id = last_response_id === nothing ? nothing : String(last_response_id),
                            conversation_id = session_id === nothing ? nothing : String(session_id),
                            usage_details = _usage_details_from_copilot(nothing, get(event, "usage", nothing)),
                            raw_representation = event,
                        ))
                    else
                        put!(channel, _error_update(
                            "GitHub Copilot CLI exited with code $(exit_code).";
                            conversation_id = session_id,
                            raw = event,
                        ))
                    end
                elseif occursin("error", event_type)
                    put!(channel, _error_update("GitHub Copilot returned error event $(event_type)."; raw = event))
                end
            end)

            if result.exitcode != 0
                put!(channel, _error_update("GitHub Copilot CLI exited with code $(result.exitcode): $(strip(String(result.stderr)))"))
            end
        catch err
            @error "GitHub Copilot streaming error" exception = (err, catch_backtrace())
            put!(channel, _error_update("GitHub Copilot streaming failed: $(err)"; raw = sprint(showerror, err)))
        finally
            close(channel)
        end
    end

    return channel
end
