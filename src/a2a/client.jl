mutable struct A2AClient
    base_url::String
    timeout::Float64
    headers::Dict{String, String}
    poll_interval::Float64
    max_polls::Union{Nothing, Int}
end

function A2AClient(;
    base_url::AbstractString,
    timeout::Real = 60.0,
    headers::Union{Nothing, AbstractDict} = nothing,
    poll_interval::Real = 1.0,
    max_polls::Union{Nothing, Integer} = 300,
)
    return A2AClient(
        _normalize_base_url(base_url),
        Float64(timeout),
        headers === nothing ? Dict{String, String}() : Dict{String, String}(string(key) => string(value) for (key, value) in pairs(headers)),
        Float64(poll_interval),
        max_polls === nothing ? nothing : Int(max_polls),
    )
end

function Base.show(io::IO, client::A2AClient)
    print(io, "A2AClient(\"", client.base_url, "\")")
end

function _resolve_poll_interval(client::A2AClient, poll_interval)::Float64
    poll_interval === nothing && return client.poll_interval
    return Float64(poll_interval)
end

function _resolve_max_polls(client::A2AClient, max_polls)::Union{Nothing, Int}
    max_polls === nothing && return client.max_polls
    return Int(max_polls)
end

function _request_headers(client::A2AClient; json::Bool=false)::Vector{Pair{String, String}}
    headers = Dict{String, String}(
        "Accept" => "application/json",
        "User-Agent" => "AgentFramework.jl/A2A/0.1.0",
    )
    json && (headers["Content-Type"] = "application/json")
    merge!(headers, client.headers)
    return collect(headers)
end

function _http_readtimeout(client::A2AClient)::Int
    return max(1, ceil(Int, client.timeout))
end

function _jsonrpc_request(client::A2AClient, method::String, params::Dict{String, Any})::Dict{String, Any}
    payload = Dict{String, Any}(
        "jsonrpc" => "2.0",
        "id" => string(UUIDs.uuid4()),
        "method" => method,
        "params" => params,
    )

    response = try
        HTTP.request(
            "POST",
            client.base_url,
            _request_headers(client; json = true),
            JSON3.write(payload);
            status_exception = false,
            readtimeout = _http_readtimeout(client),
        )
    catch error
        throw(A2AError("A2A request failed for $method", error))
    end

    if response.status < 200 || response.status >= 300
        throw(A2AProtocolError("A2A $method request failed with HTTP $(response.status): $(String(response.body))", method))
    end

    body = _json_body(response)
    if haskey(body, "error")
        err = _dict(body["error"])
        code = get(err, "code", nothing)
        message = _maybe_string(get(err, "message", "Unknown A2A error"))
        detail = code === nothing ? message : "$message (code=$(code))"
        throw(A2AProtocolError("A2A $method request failed: $detail", method))
    end

    haskey(body, "result") || throw(A2AProtocolError("A2A $method response was missing `result`", method))
    return _dict(body["result"])
end

function get_agent_card(client::A2AClient)::A2AAgentCard
    response = try
        HTTP.request(
            "GET",
            _join_url(client.base_url, "/.well-known/agent.json"),
            _request_headers(client);
            status_exception = false,
            readtimeout = _http_readtimeout(client),
        )
    catch error
        throw(A2AError("A2A agent card request failed", error))
    end

    if response.status < 200 || response.status >= 300
        throw(A2AProtocolError("A2A agent card request failed with HTTP $(response.status): $(String(response.body))", "agent/card"))
    end

    return a2a_agent_card_from_dict(_json_body(response))
end

function send_message(
    client::A2AClient,
    message::Message;
    context_id::Union{Nothing, String} = nothing,
    reference_task_ids::Vector{String} = String[],
    background::Bool = false,
    poll_interval = nothing,
    max_polls = nothing,
)::AgentResponse
    response = response_from_a2a_result(
        _jsonrpc_request(
            client,
            "message/send",
            Dict{String, Any}(
                "message" => message_to_a2a_dict(
                    message;
                    context_id = context_id,
                    reference_task_ids = reference_task_ids,
                ),
            ),
        ),
    )

    if !background && response.continuation_token isa A2AContinuationToken
        return wait_for_completion(
            client,
            response.continuation_token;
            poll_interval = _resolve_poll_interval(client, poll_interval),
            max_polls = _resolve_max_polls(client, max_polls),
        )
    end

    return response
end

function get_task(
    client::A2AClient,
    token::A2AContinuationToken;
    background::Bool = true,
    poll_interval = nothing,
    max_polls = nothing,
)::AgentResponse
    response = response_from_a2a_result(
        _jsonrpc_request(client, "tasks/get", Dict{String, Any}("id" => token.task_id)),
    )

    if !background && response.continuation_token isa A2AContinuationToken
        return wait_for_completion(
            client,
            response.continuation_token;
            poll_interval = _resolve_poll_interval(client, poll_interval),
            max_polls = _resolve_max_polls(client, max_polls),
        )
    end

    return response
end

function get_task(
    client::A2AClient,
    task_id::AbstractString;
    kwargs...,
)::AgentResponse
    return get_task(client, A2AContinuationToken(task_id = String(task_id)); kwargs...)
end

function wait_for_completion(
    client::A2AClient,
    token::A2AContinuationToken;
    poll_interval = nothing,
    max_polls = nothing,
)::AgentResponse
    interval = _resolve_poll_interval(client, poll_interval)
    limit = _resolve_max_polls(client, max_polls)
    current = token
    polls = 0

    while true
        polls += 1
        if limit !== nothing && polls > limit
            throw(A2ATimeoutError("A2A task $(current.task_id) did not complete after $(limit) polls"))
        end

        sleep(interval)
        response = get_task(client, current; background = true)
        if !(response.continuation_token isa A2AContinuationToken)
            return response
        end
        current = response.continuation_token
    end
end
