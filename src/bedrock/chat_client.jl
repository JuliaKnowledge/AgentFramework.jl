const DEFAULT_BEDROCK_MAX_TOKENS = 1024

const _BEDROCK_FINISH_REASON_MAP = Dict{String, FinishReason}(
    "end_turn" => STOP,
    "stop_sequence" => STOP,
    "max_tokens" => LENGTH,
    "length" => LENGTH,
    "content_filtered" => CONTENT_FILTER,
    "tool_use" => TOOL_CALLS,
)

Base.@kwdef mutable struct BedrockChatClient <: AbstractChatClient
    model::String = ""
    region::String = ""
    endpoint::String = ""
    credentials::Union{Nothing, BedrockCredentials} = nothing
    access_key_id::String = ""
    secret_access_key::String = ""
    session_token::Union{Nothing, String} = nothing
    profile::String = ""
    default_headers::Dict{String, String} = Dict{String, String}()
    options::Dict{String, Any} = Dict{String, Any}()
    read_timeout::Int = 120
end

function Base.show(io::IO, client::BedrockChatClient)
    model = isempty(client.model) ? get(ENV, "BEDROCK_CHAT_MODEL", "") : client.model
    print(io, "BedrockChatClient(\"", model, "\")")
end

function _resolve_chat_model(client::BedrockChatClient)::String
    model = isempty(client.model) ? get(ENV, "BEDROCK_CHAT_MODEL", "") : client.model
    model = String(strip(model))
    isempty(model) && throw(
        ChatClientInvalidRequestError(
            "Bedrock chat model not set. Provide model or set BEDROCK_CHAT_MODEL.",
        ),
    )
    return model
end

function _resolve_bedrock_endpoint(client::BedrockChatClient)::String
    endpoint = _nonempty_string(client.endpoint)
    endpoint !== nothing && return rstrip(endpoint, '/')
    return "https://bedrock-runtime.$(_resolve_bedrock_region(client)).amazonaws.com"
end

function _bedrock_converse_url(client::BedrockChatClient, model::AbstractString)::String
    return _resolve_bedrock_endpoint(client) * "/model/" * _aws_percent_encode(model) * "/converse"
end

function _dictify(value)::Dict{String, Any}
    value isa Dict{String, Any} && return value
    value isa AbstractDict && return Dict{String, Any}(string(key) => item for (key, item) in pairs(value))
    return Dict{String, Any}()
end

function _generate_tool_call_id()::String
    return "tool-call-" * string(UUIDs.uuid4())
end

function _convert_prepared_tool_result_to_blocks(value)::Vector{Dict{String, Any}}
    if value isa AbstractVector
        blocks = Dict{String, Any}[]
        for item in value
            append!(blocks, _convert_prepared_tool_result_to_blocks(item))
        end
        return isempty(blocks) ? Dict{String, Any}[Dict("text" => "")] : blocks
    elseif value isa AbstractDict
        return [Dict{String, Any}("json" => Dict{String, Any}(string(key) => item for (key, item) in pairs(value)))]
    elseif value isa NamedTuple
        return [Dict{String, Any}("json" => Dict{String, Any}(string(key) => item for (key, item) in pairs(value)))]
    elseif value isa AbstractString
        return [Dict{String, Any}("text" => String(value))]
    elseif value isa Content && is_text(value)
        return [Dict{String, Any}("text" => something(value.text, ""))]
    elseif value isa Number || value isa Bool || value === nothing
        return [Dict{String, Any}("json" => value)]
    end

    return [Dict{String, Any}("text" => string(value))]
end

function _convert_tool_result_to_blocks(result)::Vector{Dict{String, Any}}
    if result isa AbstractString
        try
            parsed = JSON3.read(result)
            return _convert_prepared_tool_result_to_blocks(parsed)
        catch
            return [Dict{String, Any}("text" => String(result))]
        end
    end

    return _convert_prepared_tool_result_to_blocks(result)
end

function _convert_content_to_bedrock_block(content::Content)::Union{Nothing, Dict{String, Any}}
    if is_text(content) || is_reasoning(content)
        return Dict{String, Any}("text" => something(content.text, ""))
    elseif content.type == AgentFramework.ERROR_CONTENT
        return Dict{String, Any}("text" => something(content.message, ""))
    elseif is_function_call(content)
        arguments = parse_arguments(content)
        arguments === nothing && (arguments = Dict{String, Any}())
        return Dict{String, Any}(
            "toolUse" => Dict{String, Any}(
                "toolUseId" => something(content.call_id, _generate_tool_call_id()),
                "name" => something(content.name, ""),
                "input" => arguments,
            ),
        )
    elseif is_function_result(content)
        tool_result_blocks = if content.items !== nothing
            text_parts = [something(item.text, "") for item in content.items if is_text(item)]
            rich_items = [item for item in content.items if !(is_text(item) || is_reasoning(item))]
            !isempty(rich_items) && @warn "Bedrock does not support rich content in tool results; omitting unsupported items."
            _convert_tool_result_to_blocks(join(text_parts, "\n"))
        else
            _convert_tool_result_to_blocks(content.result)
        end

        tool_result = Dict{String, Any}(
            "toolUseId" => something(content.call_id, _generate_tool_call_id()),
            "content" => tool_result_blocks,
            "status" => content.exception === nothing ? "success" : "error",
        )

        if content.exception !== nothing
            push!(tool_result["content"], Dict{String, Any}("text" => content.exception))
        end

        return Dict{String, Any}("toolResult" => tool_result)
    end

    return nothing
end

function _convert_message_to_bedrock_blocks(message::Message)::Vector{Dict{String, Any}}
    blocks = Dict{String, Any}[]
    for content in message.contents
        block = _convert_content_to_bedrock_block(content)
        block === nothing && continue
        push!(blocks, block)
    end
    return blocks
end

function _align_tool_results_with_pending(
    blocks::Vector{Dict{String, Any}},
    pending_tool_use_ids::Vector{String},
)::Vector{Dict{String, Any}}
    isempty(blocks) && return blocks
    isempty(pending_tool_use_ids) && return [
        block for block in blocks if !(block isa AbstractDict && haskey(block, "toolResult"))
    ]

    pending = copy(pending_tool_use_ids)
    aligned = Dict{String, Any}[]

    for block in blocks
        tool_result = get(block, "toolResult", nothing)
        tool_result === nothing && begin
            push!(aligned, block)
            continue
        end

        if isempty(pending)
            @debug "Dropping extra Bedrock tool result block without a pending tool use ID."
            continue
        end

        tool_result_dict = _dictify(tool_result)
        tool_use_id = _nonempty_string(get(tool_result_dict, "toolUseId", nothing))

        if tool_use_id === nothing
            tool_result_dict["toolUseId"] = popfirst!(pending)
            block["toolResult"] = tool_result_dict
            push!(aligned, block)
            continue
        end

        idx = findfirst(==(tool_use_id), pending)
        if idx === nothing
            @debug "Dropping Bedrock tool result block referencing unknown toolUseId $tool_use_id."
            continue
        end

        deleteat!(pending, idx)
        push!(aligned, block)
    end

    return aligned
end

function _prepare_bedrock_messages(messages::Vector{Message})::Tuple{Vector{Dict{String, String}}, Vector{Dict{String, Any}}}
    system_prompts = Dict{String, String}[]
    conversation = Dict{String, Any}[]
    pending_tool_use_ids = String[]

    for message in messages
        if message.role == :system
            text = strip(get_text(message))
            isempty(text) || push!(system_prompts, Dict{String, String}("text" => text))
            continue
        end

        blocks = _convert_message_to_bedrock_blocks(message)
        isempty(blocks) && continue

        role = message.role == :assistant ? "assistant" : "user"
        if role == "assistant"
            pending_tool_use_ids = [
                _dictify(block["toolUse"])["toolUseId"]
                for block in blocks
                if haskey(block, "toolUse")
            ]
        elseif message.role == :tool
            blocks = _align_tool_results_with_pending(blocks, pending_tool_use_ids)
            pending_tool_use_ids = String[]
            isempty(blocks) && continue
        else
            pending_tool_use_ids = String[]
        end

        push!(conversation, Dict{String, Any}("role" => role, "content" => blocks))
    end

    return system_prompts, conversation
end

function _prepare_tools(tools::Vector{FunctionTool})::Union{Nothing, Dict{String, Any}}
    isempty(tools) && return nothing

    converted = Dict{String, Any}[
        Dict{String, Any}(
            "toolSpec" => Dict{String, Any}(
                "name" => tool.name,
                "description" => tool.description,
                "inputSchema" => Dict{String, Any}("json" => tool.parameters),
            ),
        )
        for tool in tools
    ]

    return Dict{String, Any}("tools" => converted)
end

function _build_converse_request(
    client::BedrockChatClient,
    messages::Vector{Message},
    options::ChatOptions,
)::Dict{String, Any}
    model = options.model !== nothing ? options.model : _resolve_chat_model(client)
    system_prompts, conversation = _prepare_bedrock_messages(messages)
    isempty(conversation) && throw(
        ChatClientInvalidRequestError("Bedrock requests require at least one non-system message."),
    )

    body = Dict{String, Any}(
        "modelId" => model,
        "messages" => conversation,
        "inferenceConfig" => Dict{String, Any}("maxTokens" => something(options.max_tokens, DEFAULT_BEDROCK_MAX_TOKENS)),
    )

    !isempty(system_prompts) && (body["system"] = system_prompts)
    options.temperature !== nothing && (body["inferenceConfig"]["temperature"] = options.temperature)
    options.top_p !== nothing && (body["inferenceConfig"]["topP"] = options.top_p)
    options.stop !== nothing && (body["inferenceConfig"]["stopSequences"] = options.stop)

    if options.tool_choice !== nothing && isempty(options.tools) && options.tool_choice != "none"
        throw(ChatClientInvalidRequestError("Bedrock tool_choice requires at least one tool."))
    end

    tool_config = _prepare_tools(options.tools)
    if options.tool_choice !== nothing
        if options.tool_choice == "none"
            tool_config = nothing
        elseif options.tool_choice == "auto"
            tool_config = something(tool_config, Dict{String, Any}())
            tool_config["toolChoice"] = Dict{String, Any}("auto" => Dict{String, Any}())
        elseif options.tool_choice == "required"
            tool_config = something(tool_config, Dict{String, Any}())
            tool_config["toolChoice"] = Dict{String, Any}("any" => Dict{String, Any}())
        else
            tool_config = something(tool_config, Dict{String, Any}())
            tool_config["toolChoice"] = Dict{String, Any}("tool" => Dict{String, Any}("name" => options.tool_choice))
        end
    end
    tool_config !== nothing && !isempty(tool_config) && (body["toolConfig"] = tool_config)

    for (key, value) in client.options
        body[key] = value
    end
    for (key, value) in options.additional
        body[key] = value
    end

    return body
end

function _parse_usage(usage)::Union{Nothing, UsageDetails}
    usage === nothing && return nothing
    usage_dict = _dictify(usage)
    isempty(usage_dict) && return nothing

    return UsageDetails(
        input_tokens = get(usage_dict, "inputTokens", nothing),
        output_tokens = get(usage_dict, "outputTokens", nothing),
        total_tokens = get(usage_dict, "totalTokens", nothing),
    )
end

function _convert_tool_result_value(content)
    content === nothing && return nothing

    if content isa AbstractVector
        values = Any[]
        for item in content
            item_dict = _dictify(item)
            if !isempty(item_dict)
                if haskey(item_dict, "text")
                    push!(values, item_dict["text"])
                elseif haskey(item_dict, "json")
                    push!(values, item_dict["json"])
                else
                    push!(values, item)
                end
            else
                push!(values, item)
            end
        end
        return length(values) == 1 ? values[1] : values
    end

    item_dict = _dictify(content)
    if !isempty(item_dict)
        haskey(item_dict, "text") && return item_dict["text"]
        haskey(item_dict, "json") && return item_dict["json"]
    end

    return content
end

function _parse_message_contents(content_blocks)::Vector{Content}
    contents = Content[]

    for block in content_blocks
        block_dict = _dictify(block)
        isempty(block_dict) && continue

        if haskey(block_dict, "text")
            content = text_content(String(block_dict["text"]))
            content.raw_representation = block
            push!(contents, content)
            continue
        end

        if haskey(block_dict, "json")
            content = text_content(JSON3.write(block_dict["json"]))
            content.raw_representation = block
            push!(contents, content)
            continue
        end

        tool_use = _dictify(get(block_dict, "toolUse", nothing))
        if !isempty(tool_use)
            tool_name = _nonempty_string(get(tool_use, "name", nothing))
            tool_name === nothing && throw(
                ChatClientInvalidResponseError("Bedrock response was missing a tool name in toolUse."),
            )
            arguments = JSON3.write(get(tool_use, "input", Dict{String, Any}()))
            content = function_call_content(
                something(_nonempty_string(get(tool_use, "toolUseId", nothing)), _generate_tool_call_id()),
                tool_name,
                arguments,
            )
            content.raw_representation = block
            push!(contents, content)
            continue
        end

        tool_result = _dictify(get(block_dict, "toolResult", nothing))
        if !isempty(tool_result)
            status = lowercase(String(get(tool_result, "status", "success")))
            exception = status in ("success", "ok") ? nothing : "Bedrock tool result status: $status"
            content = function_result_content(
                something(_nonempty_string(get(tool_result, "toolUseId", nothing)), _generate_tool_call_id()),
                _convert_tool_result_value(get(tool_result, "content", nothing));
                exception = exception,
            )
            content.raw_representation = block
            push!(contents, content)
            continue
        end
    end

    return contents
end

function _parse_bedrock_finish_reason(reason)::Union{Nothing, FinishReason}
    text = _nonempty_string(reason)
    text === nothing && return nothing
    return get(_BEDROCK_FINISH_REASON_MAP, lowercase(text), nothing)
end

function _parse_converse_response(response::Dict{String, Any}, client::BedrockChatClient)::ChatResponse
    output = _dictify(get(response, "output", nothing))
    message = _dictify(get(output, "message", nothing))
    content_blocks = get(message, "content", Any[])
    contents = _parse_message_contents(content_blocks)

    chat_message = Message(
        role = :assistant,
        contents = contents,
        raw_representation = isempty(message) ? nothing : message,
    )

    additional_properties = Dict{String, Any}()
    for key in ("metrics", "trace", "additionalModelResponseFields")
        haskey(response, key) && (additional_properties[key] = response[key])
    end
    haskey(output, "completionReason") && (additional_properties["completionReason"] = output["completionReason"])

    return ChatResponse(
        messages = [chat_message],
        response_id = let
            value = get(response, "responseId", nothing)
            value === nothing ? _nonempty_string(get(message, "id", nothing)) : String(value)
        end,
        model_id = let
            value = get(response, "modelId", nothing)
            value === nothing ? _resolve_chat_model(client) : String(value)
        end,
        finish_reason = _parse_bedrock_finish_reason(
            haskey(output, "completionReason") ? output["completionReason"] : get(response, "stopReason", nothing),
        ),
        usage_details = _parse_usage(get(response, "usage", nothing)),
        additional_properties = additional_properties,
        raw_representation = response,
    )
end

function get_response(
    client::BedrockChatClient,
    messages::Vector{Message},
    options::ChatOptions,
)::ChatResponse
    body = _build_converse_request(client, messages, options)
    url = _bedrock_converse_url(client, String(body["modelId"]))
    response = _post_json(client, url, body; error_label = "Bedrock API")
    return _parse_converse_response(response, client)
end

function get_response_streaming(
    client::BedrockChatClient,
    messages::Vector{Message},
    options::ChatOptions,
)::Channel{ChatResponseUpdate}
    channel = Channel{ChatResponseUpdate}(1)

    Threads.@spawn begin
        try
            response = get_response(client, messages, options)
            message = isempty(response.messages) ? Message(:assistant, Content[]) : response.messages[1]
            put!(channel, ChatResponseUpdate(
                role = message.role,
                contents = message.contents,
                finish_reason = response.finish_reason,
                model_id = response.model_id,
                usage_details = response.usage_details,
                response_id = response.response_id,
                raw_representation = response.raw_representation,
            ))
        catch exc
            if !(exc isa InvalidStateException)
                @error "Bedrock streaming error" exception = (exc, catch_backtrace())
            end
        finally
            close(channel)
        end
    end

    return channel
end

AgentFramework.streaming_capability(::Type{BedrockChatClient}) = HasStreaming()
AgentFramework.tool_calling_capability(::Type{BedrockChatClient}) = HasToolCalling()
