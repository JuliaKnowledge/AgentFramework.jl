function _message_role_to_a2a(role::Symbol)::String
    role == :assistant && return "agent"
    return "user"
end

function _a2a_role_to_message(role)::Symbol
    lowercase(string(role)) == "agent" && return :assistant
    return :user
end

function _task_finish_reason(task::A2ATask)::Union{Nothing, FinishReason}
    state = task.status.state
    state == A2A_TASK_COMPLETED && return STOP
    state in (A2A_TASK_FAILED, A2A_TASK_CANCELED, A2A_TASK_REJECTED) && return FINISH_ERROR
    return nothing
end

function build_continuation_token(task::A2ATask)::Union{Nothing, A2AContinuationToken}
    is_in_progress_task_state(task.status.state) || return nothing
    return A2AContinuationToken(task_id = task.id, context_id = task.context_id)
end

function content_to_a2a_part(content::Content)::Dict{String, Any}
    metadata = _filter_metadata(content.additional_properties)

    if content.type == AgentFramework.TEXT || content.type == AgentFramework.TEXT_REASONING
        content.text === nothing && throw(A2AError("Text content requires a non-null text payload"))
        part = Dict{String, Any}("kind" => "text", "text" => content.text)
        !isempty(metadata) && (part["metadata"] = metadata)
        return part
    elseif content.type == AgentFramework.ERROR_CONTENT
        text = content.message === nothing ? "An error occurred." : content.message
        part = Dict{String, Any}("kind" => "text", "text" => text)
        !isempty(metadata) && (part["metadata"] = metadata)
        return part
    elseif content.type == AgentFramework.URI
        content.uri === nothing && throw(A2AError("URI content requires a non-null uri payload"))
        file = Dict{String, Any}("uri" => content.uri)
        content.media_type !== nothing && (file["mimeType"] = content.media_type)
        part = Dict{String, Any}("kind" => "file", "file" => file)
        !isempty(metadata) && (part["metadata"] = metadata)
        return part
    elseif content.type == AgentFramework.DATA
        content.text === nothing && throw(A2AError("Data content requires base64 data in the `text` field"))
        file = Dict{String, Any}("bytes" => content.text)
        content.media_type !== nothing && (file["mimeType"] = content.media_type)
        part = Dict{String, Any}("kind" => "file", "file" => file)
        !isempty(metadata) && (part["metadata"] = metadata)
        return part
    elseif content.type == AgentFramework.HOSTED_FILE
        content.file_id === nothing && throw(A2AError("Hosted file content requires `file_id`"))
        part = Dict{String, Any}("kind" => "file", "file" => Dict{String, Any}("uri" => content.file_id))
        !isempty(metadata) && (part["metadata"] = metadata)
        return part
    end

    throw(A2AError("Unsupported content type for A2A conversion: $(content_type_string(content.type))"))
end

function message_to_a2a_dict(
    message::Message;
    context_id::Union{Nothing, String} = nothing,
    reference_task_ids::Vector{String} = String[],
)::Dict{String, Any}
    isempty(message.contents) && throw(A2AError("Message.contents is empty; cannot convert to A2A"))

    effective_context_id = context_id !== nothing ? context_id : _maybe_string(get(message.additional_properties, "context_id", nothing))
    payload = Dict{String, Any}(
        "kind" => "message",
        "role" => _message_role_to_a2a(message.role),
        "messageId" => message.message_id === nothing ? string(UUIDs.uuid4()) : message.message_id,
        "parts" => [content_to_a2a_part(content) for content in message.contents],
    )

    effective_context_id !== nothing && (payload["contextId"] = effective_context_id)

    metadata = _filter_metadata(message.additional_properties)
    !isempty(metadata) && (payload["metadata"] = metadata)
    !isempty(reference_task_ids) && (payload["referenceTaskIds"] = reference_task_ids)

    return payload
end

function a2a_part_to_content(part::AbstractDict)::Content
    values = _dict(part)
    kind = lowercase(string(get(values, "kind", "")))
    metadata = _dict(get(values, "metadata", Dict{String, Any}()))

    if kind == "text"
        return Content(
            type = AgentFramework.TEXT,
            text = string(get(values, "text", "")),
            additional_properties = metadata,
            raw_representation = values,
        )
    elseif kind == "file"
        file = _dict(get(values, "file", Dict{String, Any}()))
        if haskey(file, "uri")
            return Content(
                type = AgentFramework.URI,
                uri = string(file["uri"]),
                media_type = _maybe_string(get(file, "mimeType", nothing)),
                additional_properties = metadata,
                raw_representation = values,
            )
        elseif haskey(file, "bytes")
            return Content(
                type = AgentFramework.DATA,
                text = string(file["bytes"]),
                media_type = _maybe_string(get(file, "mimeType", nothing)),
                additional_properties = metadata,
                raw_representation = values,
            )
        end
        throw(A2AError("A2A file part must include either `uri` or `bytes`"))
    elseif kind == "data"
        return Content(
            type = AgentFramework.TEXT,
            text = JSON3.write(_materialize(get(values, "data", nothing))),
            additional_properties = metadata,
            raw_representation = values,
        )
    end

    throw(A2AError("Unsupported A2A part kind: $(get(values, "kind", nothing))"))
end

function a2a_message_to_message(data::AbstractDict)::Message
    values = _dict(data)
    parts = get(values, "parts", Any[])
    contents = Content[a2a_part_to_content(_dict(part)) for part in parts]
    properties = _dict(get(values, "metadata", Dict{String, Any}()))
    context_id = _maybe_string(get(values, "contextId", nothing))
    context_id !== nothing && (properties["context_id"] = context_id)

    return Message(
        role = _a2a_role_to_message(get(values, "role", "agent")),
        contents = contents,
        message_id = _maybe_string(get(values, "messageId", nothing)),
        additional_properties = properties,
        raw_representation = values,
    )
end

function a2a_agent_card_from_dict(data::AbstractDict)::A2AAgentCard
    values = _dict(data)
    haskey(values, "url") || throw(A2AError("A2A agent card is missing `url`"))

    known = Set([
        "name",
        "description",
        "url",
        "version",
        "defaultInputModes",
        "defaultOutputModes",
        "capabilities",
        "skills",
    ])

    additional = Dict{String, Any}(key => value for (key, value) in values if !(key in known))

    return A2AAgentCard(
        name = _maybe_string(get(values, "name", nothing)),
        description = _maybe_string(get(values, "description", nothing)),
        url = string(values["url"]),
        version = _maybe_string(get(values, "version", nothing)),
        default_input_modes = _string_vector(get(values, "defaultInputModes", Any[])),
        default_output_modes = _string_vector(get(values, "defaultOutputModes", Any[])),
        capabilities = _dict(get(values, "capabilities", Dict{String, Any}())),
        skills = [_dict(skill) for skill in get(values, "skills", Any[]) if skill isa AbstractDict],
        additional_properties = additional,
        raw_representation = values,
    )
end

function a2a_artifact_from_dict(data::AbstractDict)::A2AArtifact
    values = _dict(data)
    parts = get(values, "parts", Any[])
    contents = Content[a2a_part_to_content(_dict(part)) for part in parts]
    return A2AArtifact(
        artifact_id = _maybe_string(get(values, "artifactId", get(values, "id", nothing))),
        name = _maybe_string(get(values, "name", nothing)),
        description = _maybe_string(get(values, "description", nothing)),
        contents = contents,
        additional_properties = _dict(get(values, "metadata", Dict{String, Any}())),
        raw_representation = values,
    )
end

function a2a_task_from_dict(data::AbstractDict)::A2ATask
    values = _dict(data)
    haskey(values, "id") || throw(A2AError("A2A task is missing `id`"))

    status_data = _dict(get(values, "status", Dict{String, Any}()))
    status_message = get(status_data, "message", nothing)

    return A2ATask(
        id = string(values["id"]),
        context_id = _maybe_string(get(values, "contextId", nothing)),
        status = A2ATaskStatus(
            state = parse_task_state(get(status_data, "state", "unknown")),
            message = status_message isa AbstractDict ? a2a_message_to_message(status_message) : nothing,
            timestamp = _maybe_string(get(status_data, "timestamp", nothing)),
            raw_representation = status_data,
        ),
        artifacts = [a2a_artifact_from_dict(item) for item in get(values, "artifacts", Any[]) if item isa AbstractDict],
        history = [a2a_message_to_message(item) for item in get(values, "history", Any[]) if item isa AbstractDict],
        metadata = _dict(get(values, "metadata", Dict{String, Any}())),
        raw_representation = values,
    )
end

function task_messages(task::A2ATask)::Vector{Message}
    if !isempty(task.artifacts)
        return [
            Message(
                role = :assistant,
                contents = copy(artifact.contents),
                additional_properties = copy(artifact.additional_properties),
                raw_representation = artifact.raw_representation,
            )
            for artifact in task.artifacts
        ]
    elseif !isempty(task.history)
        message = task.history[end]
        return [
            Message(
                role = message.role,
                contents = copy(message.contents),
                author_name = message.author_name,
                message_id = message.message_id,
                additional_properties = copy(message.additional_properties),
                raw_representation = message.raw_representation,
            ),
        ]
    elseif task.status.message !== nothing
        message = task.status.message
        return [
            Message(
                role = message.role,
                contents = copy(message.contents),
                author_name = message.author_name,
                message_id = message.message_id,
                additional_properties = copy(message.additional_properties),
                raw_representation = message.raw_representation,
            ),
        ]
    end

    return Message[]
end

function task_to_response(task::A2ATask)::AgentResponse
    token = build_continuation_token(task)
    return AgentResponse(
        messages = task_messages(task),
        response_id = task.id,
        finish_reason = token === nothing ? _task_finish_reason(task) : nothing,
        additional_properties = copy(task.metadata),
        continuation_token = token,
        raw_representation = task,
    )
end

function response_from_a2a_result(data::AbstractDict)::AgentResponse
    values = _dict(data)
    kind = lowercase(string(get(values, "kind", "")))

    if kind == "message"
        message = a2a_message_to_message(values)
        return AgentResponse(
            messages = [message],
            response_id = _maybe_string(get(values, "messageId", nothing)),
            finish_reason = STOP,
            raw_representation = values,
        )
    elseif kind == "task"
        return task_to_response(a2a_task_from_dict(values))
    end

    throw(A2AProtocolError("Unsupported A2A result kind: $(get(values, "kind", nothing))", nothing))
end
