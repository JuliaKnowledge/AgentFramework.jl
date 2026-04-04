# Ollama chat client for AgentFramework.jl
# Implements AbstractChatClient using the Ollama OpenAI-compatible API.
# Uses curl subprocess for streaming (HTTP.jl's HTTP.open has connection pool issues with Ollama).

"""
    OllamaChatClient <: AbstractChatClient

Chat client that connects to a local Ollama instance via its OpenAI-compatible API.

# Fields
- `model::String`: Ollama model name (e.g., "gemma3:latest", "qwen3:8b").
- `base_url::String`: Ollama API base URL (default: "http://localhost:11434").
- `options::Dict{String, Any}`: Default Ollama-specific options (e.g., num_ctx, seed).
- `read_timeout::Int`: Read timeout in seconds for non-streaming requests (default: 300).

# Examples
```julia
client = OllamaChatClient(model="gemma3:latest")
response = get_response(client, [Message(:user, "Hello!")], ChatOptions())
println(response.text)
```
"""
Base.@kwdef mutable struct OllamaChatClient <: AbstractChatClient
    model::String
    base_url::String = "http://localhost:11434"
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 300
end

function Base.show(io::IO, c::OllamaChatClient)
    print(io, "OllamaChatClient(\"", c.model, "\")")
end

# ── Message Conversion ───────────────────────────────────────────────────────

function _messages_to_openai(messages::Vector{Message}, tools::Vector{FunctionTool})
    result = Any[]
    for msg in messages
        role_str = String(msg.role)

        tool_calls_json = Any[]
        text_parts = String[]
        tool_results = Any[]

        for c in msg.contents
            if is_text(c)
                push!(text_parts, something(c.text, ""))
            elseif is_function_call(c)
                push!(tool_calls_json, Dict{String, Any}(
                    "id" => something(c.call_id, ""),
                    "type" => "function",
                    "function" => Dict{String, Any}(
                        "name" => something(c.name, ""),
                        "arguments" => something(c.arguments, "{}"),
                    ),
                ))
            elseif is_function_result(c)
                push!(tool_results, Dict{String, Any}(
                    "role" => "tool",
                    "tool_call_id" => something(c.call_id, ""),
                    "content" => c.result isa AbstractString ? c.result : JSON3.write(something(c.result, "")),
                ))
            end
        end

        if !isempty(tool_results)
            append!(result, tool_results)
        elseif !isempty(tool_calls_json)
            entry = Dict{String, Any}(
                "role" => role_str,
                "content" => isempty(text_parts) ? "" : join(text_parts, " "),
                "tool_calls" => tool_calls_json,
            )
            push!(result, entry)
        else
            entry = Dict{String, Any}(
                "role" => role_str,
                "content" => join(text_parts, " "),
            )
            if msg.author_name !== nothing
                entry["name"] = msg.author_name
            end
            push!(result, entry)
        end
    end
    return result
end

function _tools_to_openai(tools::Vector{FunctionTool})
    isempty(tools) && return nothing
    return [tool_to_schema(t) for t in tools]
end

# ── Request Body Construction ────────────────────────────────────────────────

function _build_request_body(client::OllamaChatClient, messages::Vector{Message}, options::ChatOptions; stream::Bool=false)
    model = options.model !== nothing ? options.model : client.model
    all_tools = options.tools

    body = Dict{String, Any}(
        "model" => model,
        "messages" => _messages_to_openai(messages, all_tools),
        "stream" => stream,
    )

    tools_json = _tools_to_openai(all_tools)
    if tools_json !== nothing
        body["tools"] = tools_json
    end
    if options.temperature !== nothing
        body["temperature"] = options.temperature
    end
    if options.top_p !== nothing
        body["top_p"] = options.top_p
    end
    if options.max_tokens !== nothing
        body["max_tokens"] = options.max_tokens
    end
    if options.stop !== nothing
        body["stop"] = options.stop
    end
    if options.tool_choice !== nothing
        body["tool_choice"] = options.tool_choice
    end
    if options.response_format !== nothing
        body["response_format"] = options.response_format
    end

    for (k, v) in client.options
        body[k] = v
    end

    return body
end

# ── Non-Streaming Response ───────────────────────────────────────────────────

function get_response(client::OllamaChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    body = _build_request_body(client, messages, options; stream=false)
    url = client.base_url * "/v1/chat/completions"
    json_body = JSON3.write(body)

    resp = HTTP.post(url,
        ["Content-Type" => "application/json", "Connection" => "close"],
        json_body;
        status_exception = false,
        readtimeout = client.read_timeout,
        connect_timeout = 10,
        retry = false,
    )

    if resp.status != 200
        throw(ChatClientError("Ollama API error ($(resp.status)): $(String(resp.body))"))
    end

    data = JSON3.read(String(resp.body), Dict{String, Any})
    return _parse_openai_response(data)
end

# ── Streaming Response ───────────────────────────────────────────────────────
# Uses curl subprocess for reliable SSE streaming (HTTP.jl's HTTP.open has
# connection pool issues that cause hangs with long-running Ollama connections).

function get_response_streaming(client::OllamaChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    body = _build_request_body(client, messages, options; stream=true)
    url = client.base_url * "/v1/chat/completions"
    json_body = JSON3.write(body)

    channel = Channel{ChatResponseUpdate}(32)

    Threads.@spawn begin
        proc = nothing
        try
            cmd = `curl -sN --max-time $(client.read_timeout) -H "Content-Type: application/json" -d $json_body $url`
            proc = open(cmd, "r")

            for line in eachline(proc)
                line = strip(line)
                isempty(line) && continue
                startswith(line, "data: ") || continue
                payload = line[7:end]
                payload == "[DONE]" && break

                try
                    chunk = JSON3.read(payload, Dict{String, Any})
                    update = _parse_openai_stream_chunk(chunk)
                    if update !== nothing
                        put!(channel, update)
                    end
                catch e
                    @warn "Failed to parse streaming chunk" exception=e
                end
            end
        catch e
            if !(e isa InvalidStateException)
                @error "Ollama streaming error" exception=(e, catch_backtrace())
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

# ── OpenAI Response Parsing ──────────────────────────────────────────────────

function _parse_openai_response(data::Dict{String, Any})::ChatResponse
    choices = get(data, "choices", Any[])
    isempty(choices) && return ChatResponse()

    choice = choices[1]
    msg_data = get(choice, "message", Dict{String, Any}())
    role = Symbol(get(msg_data, "role", "assistant"))
    content_text = get(msg_data, "content", nothing)
    # Qwen3 thinking mode: reasoning field contains chain-of-thought, content has the answer.
    # Fall back to reasoning if content is empty (e.g., when max_tokens cuts off before answer).
    reasoning_text = get(msg_data, "reasoning", nothing)
    tool_calls_data = get(msg_data, "tool_calls", nothing)

    contents = Content[]
    # Emit reasoning as separate content if present
    if reasoning_text !== nothing && reasoning_text isa AbstractString && !isempty(reasoning_text)
        push!(contents, reasoning_content(reasoning_text))
    end
    # Emit answer content
    if content_text !== nothing && content_text isa AbstractString && !isempty(content_text)
        push!(contents, text_content(content_text))
    end

    if tool_calls_data !== nothing && tool_calls_data isa AbstractVector
        for tc in tool_calls_data
            tc_dict = tc isa Dict ? tc : Dict{String, Any}(string(k) => v for (k, v) in pairs(tc))
            func_data = get(tc_dict, "function", Dict{String, Any}())
            func_dict = func_data isa Dict ? func_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(func_data))
            push!(contents, function_call_content(
                string(get(tc_dict, "id", "")),
                string(get(func_dict, "name", "")),
                string(get(func_dict, "arguments", "{}")),
            ))
        end
    end

    msg = Message(role=role, contents=contents)

    # Parse finish reason
    fr_str = get(choice, "finish_reason", "stop")
    finish_reason = fr_str !== nothing ? parse_finish_reason(string(fr_str)) : STOP

    # Parse usage
    usage_data = get(data, "usage", nothing)
    usage = nothing
    if usage_data !== nothing
        usage_dict = usage_data isa Dict ? usage_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(usage_data))
        usage = UsageDetails(
            input_tokens = get(usage_dict, "prompt_tokens", nothing),
            output_tokens = get(usage_dict, "completion_tokens", nothing),
            total_tokens = get(usage_dict, "total_tokens", nothing),
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

function _parse_openai_stream_chunk(chunk::Dict{String, Any})::Union{Nothing, ChatResponseUpdate}
    choices = get(chunk, "choices", Any[])
    isempty(choices) && return nothing

    choice = choices[1]
    delta = get(choice, "delta", Dict{String, Any}())
    delta_dict = delta isa Dict ? delta : Dict{String, Any}(string(k) => v for (k, v) in pairs(delta))

    role_str = get(delta_dict, "role", nothing)
    role = role_str !== nothing ? Symbol(role_str) : nothing

    content_text = get(delta_dict, "content", nothing)
    reasoning_text = get(delta_dict, "reasoning", nothing)
    tool_calls_data = get(delta_dict, "tool_calls", nothing)

    contents = Content[]
    # Emit reasoning as separate content if present
    if reasoning_text !== nothing && reasoning_text isa AbstractString && !isempty(reasoning_text)
        push!(contents, reasoning_content(reasoning_text))
    end
    # Emit answer content
    if content_text !== nothing && content_text isa AbstractString && !isempty(content_text)
        push!(contents, text_content(content_text))
    end

    if tool_calls_data !== nothing && tool_calls_data isa AbstractVector
        for tc in tool_calls_data
            tc_dict = tc isa Dict ? tc : Dict{String, Any}(string(k) => v for (k, v) in pairs(tc))
            func_data = get(tc_dict, "function", Dict{String, Any}())
            func_dict = func_data isa Dict ? func_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(func_data))
            push!(contents, function_call_content(
                string(get(tc_dict, "id", "")),
                string(get(func_dict, "name", "")),
                string(get(func_dict, "arguments", "{}")),
            ))
        end
    end

    fr_str = get(choice, "finish_reason", nothing)
    finish_reason = fr_str !== nothing ? parse_finish_reason(string(fr_str)) : nothing

    # Usage on final chunk
    usage_data = get(chunk, "usage", nothing)
    usage = nothing
    if usage_data !== nothing
        usage_dict = usage_data isa Dict ? usage_data : Dict{String, Any}(string(k) => v for (k, v) in pairs(usage_data))
        usage = UsageDetails(
            input_tokens = get(usage_dict, "prompt_tokens", nothing),
            output_tokens = get(usage_dict, "completion_tokens", nothing),
            total_tokens = get(usage_dict, "total_tokens", nothing),
        )
    end

    model_id = get(chunk, "model", nothing)
    response_id = get(chunk, "id", nothing)

    # Skip empty updates (no content, no finish)
    if isempty(contents) && finish_reason === nothing && role === nothing
        return nothing
    end

    ChatResponseUpdate(
        role = role,
        contents = contents,
        finish_reason = finish_reason,
        model_id = model_id !== nothing ? string(model_id) : nothing,
        usage_details = usage,
        response_id = response_id !== nothing ? string(response_id) : nothing,
        raw_representation = chunk,
    )
end

# ── Capability Traits ────────────────────────────────────────────────────────

AgentFramework.streaming_capability(::Type{OllamaChatClient}) = HasStreaming()
AgentFramework.tool_calling_capability(::Type{OllamaChatClient}) = HasToolCalling()
AgentFramework.embedding_capability(::Type{OllamaChatClient}) = HasEmbeddings()

# ── Embeddings ───────────────────────────────────────────────────────────────

"""
    get_embeddings(client::OllamaChatClient, texts::Vector{String}; model=nothing) -> Vector{Vector{Float64}}

Get embeddings from the Ollama embeddings API.
"""
function get_embeddings(client::OllamaChatClient, texts::Vector{String}; model::Union{Nothing, String} = nothing)::Vector{Vector{Float64}}
    embed_model = something(model, client.model)
    results = Vector{Float64}[]
    for text in texts
        body = Dict{String, Any}("model" => embed_model, "input" => text)
        resp = HTTP.post("$(client.base_url)/api/embed",
            ["Content-Type" => "application/json"], JSON3.write(body);
            status_exception=false, readtimeout=client.read_timeout)

        resp.status != 200 && throw(ChatClientError("Ollama embeddings error: $(resp.status) — $(String(resp.body))"))
        result = JSON3.read(String(resp.body), Dict{String, Any})
        embeddings = result["embeddings"]
        push!(results, Vector{Float64}(embeddings[1]))
    end
    return results
end
