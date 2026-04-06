function _json_response(status::Integer, payload)
    return HTTP.Response(
        Int(status),
        ["Content-Type" => "application/json"],
        JSON3.write(_jsonish(payload)),
    )
end

function _error_response(status::Integer, message::AbstractString)
    return _json_response(status, Dict{String, Any}("error" => String(message)))
end

function _request_path(request::HTTP.Request)
    return HTTP.URI(String(request.target)).path
end

function _request_segments(request::HTTP.Request)
    path = _request_path(request)
    return filter(segment -> !isempty(segment), split(path, '/'))
end

function _request_body_string(request::HTTP.Request)
    body = request.body
    body === nothing && return ""
    body isa AbstractVector{UInt8} && return String(body)
    body isa String && return body
    return String(read(body))
end

function _request_json(request::HTTP.Request)
    body = strip(_request_body_string(request))
    isempty(body) && return Dict{String, Any}()
    return Dict{String, Any}(_materialize_json(JSON3.read(body)))
end

function _chat_options_from_dict(raw)
    raw === nothing && return nothing
    options = Dict{String, Any}(_materialize_json(raw))
    return AgentFramework.ChatOptions(
        model = get(options, "model", nothing),
        temperature = get(options, "temperature", nothing),
        top_p = get(options, "top_p", nothing),
        max_tokens = get(options, "max_tokens", nothing),
        stop = get(options, "stop", nothing),
        tool_choice = get(options, "tool_choice", nothing),
        response_format = get(options, "response_format", nothing),
        additional = Dict{String, Any}(_materialize_json(get(options, "additional", Dict{String, Any}()))),
    )
end

function _session_to_dict(session::AgentFramework.AgentSession)
    return Dict{String, Any}(
        "id" => session.id,
        "state" => _jsonish(session.state),
        "user_id" => session.user_id,
        "thread_id" => session.thread_id,
        "metadata" => _jsonish(session.metadata),
    )
end

function _message_to_dict(message::AgentFramework.Message)
    return AgentFramework.message_to_dict(message)
end

function _finish_reason_string(reason)
    reason === nothing && return nothing
    return lowercase(string(reason))
end

function _response_usage_dict(usage)
    usage === nothing && return nothing
    return Dict{String, Any}(
        "input_tokens" => usage.input_tokens,
        "output_tokens" => usage.output_tokens,
        "total_tokens" => usage.total_tokens,
    )
end

function _agent_run_to_dict(result)
    return Dict{String, Any}(
        "session" => _session_to_dict(result.session),
        "history" => Any[_message_to_dict(message) for message in result.history],
        "response" => Dict{String, Any}(
            "text" => AgentFramework.get_text(result.response),
            "finish_reason" => _finish_reason_string(result.response.finish_reason),
            "messages" => Any[_message_to_dict(message) for message in result.response.messages],
            "usage" => _response_usage_dict(result.response.usage_details),
            "model_id" => result.response.model_id,
        ),
    )
end

function _agent_summary(runtime::HostedRuntime, name::AbstractString)
    agent = runtime.agents[String(name)]
    return Dict{String, Any}(
        "name" => String(name),
        "instructions" => agent.instructions,
        "tools" => [tool.name for tool in agent.tools],
        "context_provider_count" => length(agent.context_providers),
    )
end

function _workflow_summary(runtime::HostedRuntime, name::AbstractString)
    workflow = runtime.workflows[String(name)]
    return Dict{String, Any}(
        "name" => String(name),
        "start" => workflow.start_executor_id,
        "executor_count" => length(workflow.executors),
        "output_count" => length(workflow.output_executor_ids),
    )
end

function handle_request(runtime::HostedRuntime, request::HTTP.Request)
    method = String(request.method)
    segments = _request_segments(request)

    try
        if method == "GET" && segments == ["health"]
            return _json_response(200, Dict("status" => "ok"))
        elseif method == "GET" && segments == ["agents"]
            return _json_response(200, Dict("agents" => [_agent_summary(runtime, name) for name in list_registered_agents(runtime)]))
        elseif method == "GET" && segments == ["workflows"]
            return _json_response(200, Dict("workflows" => [_workflow_summary(runtime, name) for name in list_registered_workflows(runtime)]))
        elseif length(segments) == 3 && segments[1] == "agents" && method == "POST" && segments[3] == "run"
            body = _request_json(request)
            haskey(body, "message") || return _error_response(400, "Request body must include `message`.")
            result = run_agent!(
                runtime,
                segments[2],
                body["message"];
                session_id = get(body, "session_id", nothing),
                options = _chat_options_from_dict(get(body, "options", nothing)),
            )
            return _json_response(200, _agent_run_to_dict(result))
        elseif length(segments) == 3 && segments[1] == "agents" && method == "GET" && segments[3] == "sessions"
            sessions = list_agent_sessions(runtime, segments[2])
            return _json_response(200, Dict("sessions" => [_session_to_dict(session) for session in sessions]))
        elseif length(segments) == 4 && segments[1] == "agents" && segments[3] == "sessions" && method == "GET"
            session_info = get_agent_session(runtime, segments[2], segments[4])
            return _json_response(200, Dict(
                "session" => _session_to_dict(session_info.session),
                "history" => [_message_to_dict(message) for message in session_info.history],
            ))
        elseif length(segments) == 4 && segments[1] == "agents" && segments[3] == "sessions" && method == "DELETE"
            deleted = delete_agent_session!(runtime, segments[2], segments[4])
            return _json_response(200, Dict("deleted" => deleted))
        elseif length(segments) == 3 && segments[1] == "workflows" && method == "POST" && segments[3] == "runs"
            body = _request_json(request)
            input = get(body, "input", nothing)
            run = start_workflow_run!(
                runtime,
                segments[2],
                input;
                run_id = get(body, "run_id", nothing),
                metadata = get(body, "metadata", Dict{String, Any}()),
            )
            return _json_response(200, hosted_workflow_run_to_dict(run))
        elseif length(segments) == 3 && segments[1] == "workflows" && method == "GET" && segments[3] == "runs"
            runs = list_workflow_runs(runtime, segments[2])
            return _json_response(200, Dict("runs" => [hosted_workflow_run_to_dict(run) for run in runs]))
        elseif length(segments) == 4 && segments[1] == "workflows" && segments[3] == "runs" && method == "GET"
            run = get_workflow_run(runtime, segments[2], segments[4])
            return _json_response(200, hosted_workflow_run_to_dict(run))
        elseif length(segments) == 5 && segments[1] == "workflows" && segments[3] == "runs" && segments[5] == "resume" && method == "POST"
            body = _request_json(request)
            responses = get(body, "responses", nothing)
            responses isa AbstractDict || return _error_response(400, "Request body must include a `responses` object.")
            run = resume_workflow_run!(runtime, segments[2], segments[4], responses)
            return _json_response(200, hosted_workflow_run_to_dict(run))
        end

        return _error_response(404, "Route not found.")
    catch err
        if err isa KeyError
            return _error_response(404, sprint(showerror, err))
        elseif err isa ArgumentError
            return _error_response(400, err.msg)
        end
        rethrow()
    end
end

function serve(runtime::HostedRuntime; host::AbstractString = "127.0.0.1", port::Integer = 8080, kwargs...)
    return HTTP.serve!(request -> handle_request(runtime, request), String(host), Int(port); kwargs...)
end
