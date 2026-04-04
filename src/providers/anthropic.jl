# Anthropic Claude chat client for AgentFramework.jl
# Implements AbstractChatClient using Claude's Messages API.
# Uses curl subprocess for streaming (same pattern as OpenAI provider).

# ── Anthropic Client ─────────────────────────────────────────────────────────

"""
    AnthropicChatClient <: AbstractChatClient

Chat client for Anthropic's Claude Messages API.

# Fields
- `model::String`: Model name (default: "claude-sonnet-4-20250514").
- `api_key::String`: API key. Falls back to `ENV["ANTHROPIC_API_KEY"]` if empty.
- `base_url::String`: API base URL (default: "https://api.anthropic.com").
- `api_version::String`: Anthropic API version header (default: "2023-06-01").
- `options::Dict{String, Any}`: Default provider-specific options.
- `read_timeout::Int`: Read timeout in seconds (default: 120).

# Examples
```julia
client = AnthropicChatClient(model="claude-sonnet-4-20250514")
response = get_response(client, [Message(:user, "Hello!")], ChatOptions())
println(response.text)
```
"""
Base.@kwdef mutable struct AnthropicChatClient <: AbstractChatClient
    model::String = "claude-sonnet-4-20250514"
    api_key::String = ""
    base_url::String = "https://api.anthropic.com"
    api_version::String = "2023-06-01"
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120
end

function Base.show(io::IO, c::AnthropicChatClient)
    print(io, "AnthropicChatClient(\"", c.model, "\")")
end

# ── API Key Resolution ───────────────────────────────────────────────────────

function _resolve_api_key(client::AnthropicChatClient)::String
    key = client.api_key
    if isempty(key)
        key = get(ENV, "ANTHROPIC_API_KEY", "")
    end
    if isempty(key)
        throw(ChatClientError("Anthropic API key not set. Provide api_key or set ANTHROPIC_API_KEY."))
    end
    return key
end

# ── Message Conversion ───────────────────────────────────────────────────────

"""
    _split_system_anthropic(messages) -> (system_prompt, remaining_messages)

Extract system messages into a top-level system prompt string (Anthropic requires
system as a top-level field, not as a message). Returns the joined system text
and the non-system messages.
"""
function _split_system_anthropic(messages::Vector{Message})
    system_parts = String[]
    remaining = Message[]
    for msg in messages
        if msg.role == :system
            txt = get_text(msg)
            if !isempty(txt)
                push!(system_parts, txt)
            end
        else
            push!(remaining, msg)
        end
    end
    system_prompt = isempty(system_parts) ? nothing : join(system_parts, "\n\n")
    return (system_prompt, remaining)
end

"""
    _messages_to_anthropic(messages) -> Vector{Dict}

Convert AgentFramework Messages to Anthropic's messages format.
Handles:
- Text content → `{"type": "text", "text": "..."}`
- Function calls (in assistant messages) → `{"type": "tool_use", ...}`
- Function results (in tool messages) → `{"type": "tool_result", ...}` in a user message
- Merges adjacent same-role messages (Anthropic requires alternating user/assistant)
"""
function _messages_to_anthropic(messages::Vector{Message})
    raw = Dict{String, Any}[]

    for msg in messages
        role_str = msg.role == :tool ? "user" : String(msg.role)
        content_blocks = Any[]

        for c in msg.contents
            if is_text(c)
                txt = something(c.text, "")
                if !isempty(txt)
                    push!(content_blocks, Dict{String, Any}("type" => "text", "text" => txt))
                end
            elseif is_function_call(c)
                input = Dict{String, Any}()
                if c.arguments !== nothing && !isempty(c.arguments)
                    try
                        input = JSON3.read(c.arguments, Dict{String, Any})
                    catch
                        input = Dict{String, Any}("raw" => c.arguments)
                    end
                end
                push!(content_blocks, Dict{String, Any}(
                    "type" => "tool_use",
                    "id" => something(c.call_id, string(UUIDs.uuid4())),
                    "name" => something(c.name, ""),
                    "input" => input,
                ))
            elseif is_function_result(c)
                result_str = if c.result isa AbstractString
                    c.result
                elseif c.result !== nothing
                    JSON3.write(c.result)
                else
                    ""
                end
                push!(content_blocks, Dict{String, Any}(
                    "type" => "tool_result",
                    "tool_use_id" => something(c.call_id, ""),
                    "content" => result_str,
                ))
            end
        end

        if isempty(content_blocks)
            push!(content_blocks, Dict{String, Any}("type" => "text", "text" => ""))
        end

        push!(raw, Dict{String, Any}("role" => role_str, "content" => content_blocks))
    end

    # Merge adjacent same-role messages (Anthropic requires strict alternation)
    merged = Dict{String, Any}[]
    for entry in raw
        if !isempty(merged) && last(merged)["role"] == entry["role"]
            append!(last(merged)["content"], entry["content"])
        else
            push!(merged, entry)
        end
    end

    return merged
end

"""
    _tools_to_anthropic(tools) -> Union{Nothing, Vector{Dict}}

Convert FunctionTool vector to Anthropic tool format.
Anthropic uses `input_schema` instead of OpenAI's nested `function.parameters`.
"""
function _tools_to_anthropic(tools::Vector{FunctionTool})
    isempty(tools) && return nothing
    return [Dict{String, Any}(
        "name" => t.name,
        "description" => t.description,
        "input_schema" => t.parameters,
    ) for t in tools]
end

# ── Request Body Construction ────────────────────────────────────────────────

function _build_anthropic_request(client::AnthropicChatClient,
                                  api_messages::Vector{Dict{String, Any}},
                                  system_prompt::Union{Nothing, String},
                                  options::ChatOptions;
                                  stream::Bool=false)
    model = options.model !== nothing ? options.model : client.model

    body = Dict{String, Any}(
        "model" => model,
        "messages" => api_messages,
        "max_tokens" => something(options.max_tokens, 4096),
        "stream" => stream,
    )

    if system_prompt !== nothing
        body["system"] = system_prompt
    end

    tools_json = _tools_to_anthropic(options.tools)
    if tools_json !== nothing
        body["tools"] = tools_json
    end
    if options.temperature !== nothing
        body["temperature"] = options.temperature
    end
    if options.top_p !== nothing
        body["top_p"] = options.top_p
    end
    if options.stop !== nothing
        body["stop_sequences"] = options.stop
    end
    if options.tool_choice !== nothing
        # Anthropic tool_choice format: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
        if options.tool_choice isa String
            if options.tool_choice in ("auto", "any", "none")
                body["tool_choice"] = Dict{String, Any}("type" => options.tool_choice)
            else
                body["tool_choice"] = Dict{String, Any}("type" => "tool", "name" => options.tool_choice)
            end
        else
            body["tool_choice"] = options.tool_choice
        end
    end

    for (k, v) in client.options
        body[k] = v
    end

    return body
end

# ── Header Construction ──────────────────────────────────────────────────────

function _build_headers(client::AnthropicChatClient)::Vector{Pair{String, String}}
    key = _resolve_api_key(client)
    return Pair{String, String}[
        "Content-Type" => "application/json",
        "x-api-key" => key,
        "anthropic-version" => client.api_version,
        "Connection" => "close",
    ]
end

# ── Response Parsing ─────────────────────────────────────────────────────────

const _ANTHROPIC_STOP_REASON_MAP = Dict{String, FinishReason}(
    "end_turn" => STOP,
    "stop_sequence" => STOP,
    "max_tokens" => LENGTH,
    "tool_use" => TOOL_CALLS,
)

function _parse_anthropic_stop_reason(reason::Union{Nothing, String})::FinishReason
    reason === nothing && return STOP
    return get(_ANTHROPIC_STOP_REASON_MAP, reason, STOP)
end

function _parse_anthropic_response(resp::HTTP.Response)::ChatResponse
    data = JSON3.read(String(resp.body), Dict{String, Any})
    return _parse_anthropic_response(data)
end

function _parse_anthropic_response(data::Dict{String, Any})::ChatResponse
    role = Symbol(get(data, "role", "assistant"))
    content_blocks = get(data, "content", Any[])

    contents = Content[]
    for block in content_blocks
        block_dict = block isa Dict ? block : Dict{String, Any}(string(k) => v for (k, v) in pairs(block))
        block_type = get(block_dict, "type", "")

        if block_type == "text"
            txt = get(block_dict, "text", "")
            if !isempty(txt)
                push!(contents, text_content(txt))
            end
        elseif block_type == "tool_use"
            input_data = get(block_dict, "input", Dict{String, Any}())
            args_str = input_data isa Dict ? JSON3.write(input_data) : string(input_data)
            push!(contents, function_call_content(
                string(get(block_dict, "id", "")),
                string(get(block_dict, "name", "")),
                args_str,
            ))
        end
    end

    msg = Message(role=role, contents=contents)

    stop_reason = get(data, "stop_reason", nothing)
    finish_reason = _parse_anthropic_stop_reason(stop_reason !== nothing ? string(stop_reason) : nothing)

    usage_data = get(data, "usage", nothing)
    usage = nothing
    if usage_data !== nothing
        usage_dict = usage_data isa Dict ? usage_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(usage_data))
        input_tok = get(usage_dict, "input_tokens", nothing)
        output_tok = get(usage_dict, "output_tokens", nothing)
        total = nothing
        if input_tok !== nothing && output_tok !== nothing
            total = input_tok + output_tok
        end
        usage = UsageDetails(
            input_tokens = input_tok,
            output_tokens = output_tok,
            total_tokens = total,
        )
    end

    ChatResponse(
        messages = [msg],
        finish_reason = finish_reason,
        model_id = get(data, "model", nothing),
        usage_details = usage,
        response_id = get(data, "id", nothing),
        raw_representation = data,
    )
end

# ── Non-Streaming Response ───────────────────────────────────────────────────

function get_response(client::AnthropicChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    _resolve_api_key(client)  # validate key exists early

    system_prompt, remaining = _split_system_anthropic(messages)
    api_messages = _messages_to_anthropic(remaining)
    body = _build_anthropic_request(client, api_messages, system_prompt, options; stream=false)
    headers = _build_headers(client)
    json_body = JSON3.write(body)

    url = rstrip(client.base_url, '/') * "/v1/messages"

    resp = HTTP.post(url, headers, json_body;
        status_exception = false,
        readtimeout = client.read_timeout,
        connect_timeout = 10,
        retry = false,
    )

    if resp.status != 200
        throw(ChatClientError("Anthropic API error ($(resp.status)): $(String(resp.body))"))
    end

    return _parse_anthropic_response(resp)
end

# ── Streaming Response ───────────────────────────────────────────────────────
# Uses curl subprocess for reliable SSE streaming (same pattern as OpenAI provider).

function _build_curl_headers(client::AnthropicChatClient)::Vector{String}
    key = _resolve_api_key(client)
    return ["-H", "Content-Type: application/json",
            "-H", "x-api-key: $key",
            "-H", "anthropic-version: $(client.api_version)"]
end

function get_response_streaming(client::AnthropicChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    _resolve_api_key(client)  # validate key exists early

    system_prompt, remaining = _split_system_anthropic(messages)
    api_messages = _messages_to_anthropic(remaining)
    body = _build_anthropic_request(client, api_messages, system_prompt, options; stream=true)
    json_body = JSON3.write(body)
    curl_headers = _build_curl_headers(client)
    url = rstrip(client.base_url, '/') * "/v1/messages"

    channel = Channel{ChatResponseUpdate}(32)

    Threads.@spawn begin
        proc = nothing
        try
            cmd = `curl -sN --max-time $(client.read_timeout) $curl_headers -d $json_body $url`
            proc = open(cmd, "r")

            current_block_type = nothing
            current_block_id = nothing
            current_block_name = nothing
            current_block_index = nothing
            tool_block_counter = 0

            for line in eachline(proc)
                line = strip(line)
                isempty(line) && continue

                # Parse SSE event type
                if startswith(line, "event: ")
                    # Events are handled implicitly via data payloads
                    continue
                end

                startswith(line, "data: ") || continue
                payload = line[7:end]

                try
                    chunk = JSON3.read(payload, Dict{String, Any})
                    event_type = get(chunk, "type", "")

                    if event_type == "message_start"
                        msg = get(chunk, "message", Dict{String, Any}())
                        model_id = get(msg, "model", nothing)
                        usage_data = get(msg, "usage", nothing)
                        usage = nothing
                        if usage_data !== nothing
                            ud = usage_data isa Dict ? usage_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(usage_data))
                            usage = UsageDetails(input_tokens=get(ud, "input_tokens", nothing))
                        end
                        put!(channel, ChatResponseUpdate(
                            role = :assistant,
                            model_id = model_id !== nothing ? string(model_id) : nothing,
                            usage_details = usage,
                            response_id = get(msg, "id", nothing),
                            raw_representation = chunk,
                        ))

                    elseif event_type == "content_block_start"
                        block = get(chunk, "content_block", Dict{String, Any}())
                        block_dict = block isa Dict ? block : Dict{String, Any}(string(k) => v for (k, v) in pairs(block))
                        current_block_type = get(block_dict, "type", "text")
                        block_index = get(chunk, "index", nothing)
                        if current_block_type == "tool_use"
                            current_block_index = block_index !== nothing ? Int(block_index) : tool_block_counter
                            if block_index === nothing
                                tool_block_counter += 1
                            else
                                tool_block_counter = max(tool_block_counter, Int(block_index) + 1)
                            end
                        else
                            current_block_index = block_index === nothing ? nothing : Int(block_index)
                        end
                        if current_block_type == "tool_use"
                            current_block_id = get(block_dict, "id", "")
                            current_block_name = get(block_dict, "name", "")
                        end

                    elseif event_type == "content_block_delta"
                        delta = get(chunk, "delta", Dict{String, Any}())
                        delta_dict = delta isa Dict ? delta : Dict{String, Any}(string(k) => v for (k, v) in pairs(delta))
                        delta_type = get(delta_dict, "type", "")

                        if delta_type == "text_delta"
                            txt = get(delta_dict, "text", "")
                            if !isempty(txt)
                                put!(channel, ChatResponseUpdate(
                                    contents = [text_content(txt)],
                                    raw_representation = chunk,
                                ))
                            end
                        elseif delta_type == "input_json_delta"
                            # Partial JSON for tool input — emit as function call chunk
                            partial = get(delta_dict, "partial_json", "")
                            if !isempty(partial)
                                put!(channel, ChatResponseUpdate(
                                    contents = [function_call_content(
                                        string(something(current_block_id, "")),
                                        string(something(current_block_name, "")),
                                        partial,
                                    )],
                                    raw_representation = Dict{String, Any}(
                                        "__streaming_tool_fragments__" => [
                                            Dict{String, Any}(
                                                "index" => something(current_block_index, 0),
                                                "call_id" => string(something(current_block_id, "")),
                                                "name" => string(something(current_block_name, "")),
                                                "arguments_fragment" => partial,
                                            ),
                                        ],
                                        "chunk" => chunk,
                                    ),
                                ))
                            end
                        end

                    elseif event_type == "message_delta"
                        delta = get(chunk, "delta", Dict{String, Any}())
                        delta_dict = delta isa Dict ? delta : Dict{String, Any}(string(k) => v for (k, v) in pairs(delta))
                        stop_reason = get(delta_dict, "stop_reason", nothing)
                        finish_reason = stop_reason !== nothing ? _parse_anthropic_stop_reason(string(stop_reason)) : nothing

                        usage_data = get(chunk, "usage", nothing)
                        usage = nothing
                        if usage_data !== nothing
                            ud = usage_data isa Dict ? usage_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(usage_data))
                            usage = UsageDetails(output_tokens=get(ud, "output_tokens", nothing))
                        end

                        put!(channel, ChatResponseUpdate(
                            finish_reason = finish_reason,
                            usage_details = usage,
                            raw_representation = chunk,
                        ))

                    elseif event_type == "message_stop"
                        break
                    end
                catch e
                    @warn "Failed to parse Anthropic streaming chunk" exception=e
                end
            end
        catch e
            if !(e isa InvalidStateException)
                @error "Anthropic streaming error" exception=(e, catch_backtrace())
            end
        finally
            if proc !== nothing
                try; close(proc); catch; end
            end
            close(channel)
        end
    end

    return channel
end

# ── Capability Traits ────────────────────────────────────────────────────────

AgentFramework.streaming_capability(::Type{AnthropicChatClient}) = HasStreaming()
AgentFramework.tool_calling_capability(::Type{AnthropicChatClient}) = HasToolCalling()
AgentFramework.structured_output_capability(::Type{AnthropicChatClient}) = HasStructuredOutput()
