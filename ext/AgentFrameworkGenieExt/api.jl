"""
API route handler functions for the DevUI.

Each function handles a specific API endpoint. They are called from the server
route definitions and return appropriate HTTP responses.
"""

# ─── Entity Info Serialization ────────────────────────────────────────────────

function entity_to_dict(info::EntityInfo)
    tools = if info.type == AGENT_ENTITY && hasproperty(info.entity, :tools)
        [Dict{String, Any}(
            "name" => t.name,
            "description" => t.description,
        ) for t in info.entity.tools]
    else
        Dict{String, Any}[]
    end
    Dict{String, Any}(
        "id" => info.id,
        "name" => info.name,
        "description" => info.description,
        "type" => info.type == AGENT_ENTITY ? "agent" : "workflow",
        "tools" => tools,
        "metadata" => info.metadata,
    )
end

# ─── Conversation Serialization ───────────────────────────────────────────────

function message_to_api_dict(msg::ConversationMessage)
    Dict{String, Any}(
        "id" => msg.id,
        "role" => string(msg.role),
        "content" => msg.content,
        "tool_calls" => msg.tool_calls,
        "tool_results" => msg.tool_results,
        "timestamp" => Dates.format(msg.timestamp, Dates.ISODateTimeFormat),
        "metadata" => msg.metadata,
    )
end

function conversation_to_dict(conv::Conversation; include_messages::Bool = true)
    d = Dict{String, Any}(
        "id" => conv.id,
        "entity_id" => conv.entity_id,
        "title" => conv.title,
        "message_count" => length(conv.messages),
        "created_at" => Dates.format(conv.created_at, Dates.ISODateTimeFormat),
        "updated_at" => Dates.format(conv.updated_at, Dates.ISODateTimeFormat),
    )
    if include_messages
        d["messages"] = [message_to_api_dict(m) for m in conv.messages]
    end
    return d
end

function _set_initial_conversation_title!(conv_store::ConversationStore,
                                          conv::Conversation,
                                          message::AbstractString)
    if conv.title == "New conversation" && !isempty(message)
        title = length(message) > 50 ? message[1:50] * "…" : String(message)
        rename_conversation!(conv_store, conv.id, title)
    end
end

function _first_user_text(messages::Vector{AgentFramework.Message})
    for message in messages
        if message.role == :user
            text = AgentFramework.get_text(message)
            isempty(text) || return text
        end
    end
    return nothing
end

function _seed_agui_conversation_history!(conv_store::ConversationStore,
                                          conv::Conversation,
                                          messages::Vector{AgentFramework.Message})
    for message in messages
        content, tool_calls, tool_results = _framework_message_parts(message)
        add_message!(conv_store, conv.id, message.role, content;
                     tool_calls=tool_calls, tool_results=tool_results)
    end
end

# ─── JSON Response Helpers ────────────────────────────────────────────────────

function json_response(data; status::Int = 200)
    body = JSON3.write(data)
    return HTTP.Response(status, [
        "Content-Type" => "application/json",
        "Access-Control-Allow-Origin" => "*",
    ]; body=body)
end

function error_response(message::String; status::Int = 400)
    return json_response(Dict("error" => message); status=status)
end

# ─── Route Handlers ───────────────────────────────────────────────────────────

"""
    handle_health() → HTTP.Response

Health check endpoint.
"""
function handle_health()
    return json_response(Dict("status" => "ok"))
end

"""
    handle_meta(config, registry) → HTTP.Response

Metadata about the DevUI instance.
"""
function handle_meta(config::DevUIConfig, registry::EntityRegistry)
    entities = list_entities(registry)
    return json_response(Dict{String, Any}(
        "title" => config.title,
        "version" => "0.1.0",
        "capabilities" => Dict{String, Any}(
            "streaming" => true,
            "tool_calls" => true,
            "conversations" => true,
            "ag_ui" => true,
        ),
        "entity_count" => length(entities),
        "agent_count" => count(e -> e.type == AGENT_ENTITY, entities),
        "workflow_count" => count(e -> e.type == WORKFLOW_ENTITY, entities),
    ))
end

"""
    handle_list_entities(registry) → HTTP.Response

List all registered entities.
"""
function handle_list_entities(registry::EntityRegistry)
    entities = list_entities(registry)
    return json_response([entity_to_dict(e) for e in entities])
end

"""
    handle_get_entity(registry, id) → HTTP.Response

Get details of a specific entity.
"""
function handle_get_entity(registry::EntityRegistry, id::AbstractString)
    info = get_entity(registry, id)
    if info === nothing
        return error_response("Entity not found: $id"; status=404)
    end
    return json_response(entity_to_dict(info))
end

"""
    handle_chat(registry, conv_store, body) → HTTP.Response

Process a chat message. Creates a conversation if needed, runs the agent,
and returns the response.

Expected JSON body:
```json
{
  "entity_id": "...",
  "message": "...",
  "conversation_id": "..." (optional)
}
```
"""
function handle_chat(registry::EntityRegistry, conv_store::ConversationStore, body::String)
    local parsed
    try
        parsed = JSON3.read(body, Dict{String, Any})
        if !(parsed isa Dict)
            return error_response("Invalid JSON body: expected object")
        end
    catch e
        return error_response("Invalid JSON body: $(sprint(showerror, e))")
    end

    entity_id = get(parsed, "entity_id", nothing)
    message = get(parsed, "message", nothing)
    conversation_id = get(parsed, "conversation_id", nothing)

    if entity_id === nothing || message === nothing
        return error_response("Missing required fields: entity_id, message")
    end

    info = get_entity(registry, entity_id)
    if info === nothing
        return error_response("Entity not found: $entity_id"; status=404)
    end

    # Get or create conversation
    conv = if conversation_id !== nothing
        get_conversation(conv_store, conversation_id)
    else
        nothing
    end
    if conv === nothing
        session = if info.type == AGENT_ENTITY
            AgentFramework.AgentSession()
        else
            nothing
        end
        conv = create_conversation!(conv_store, entity_id; session=session)
    end

    # Record user message
    add_message!(conv_store, conv.id, :user, message)

    # Auto-set title from first user message if still default
    _set_initial_conversation_title!(conv_store, conv, message)

    # Run the agent or workflow
    try
        if info.type == AGENT_ENTITY
            response = AgentFramework.run_agent(info.entity, message; session=conv.session)
            response_text = AgentFramework.get_text(response)
            tool_calls, tool_results = _response_tool_parts(response)

            assistant_msg = add_message!(conv_store, conv.id, :assistant, response_text;
                                         tool_calls=tool_calls, tool_results=tool_results)

            return json_response(Dict{String, Any}(
                "conversation_id" => conv.id,
                "message" => message_to_api_dict(assistant_msg),
                "finish_reason" => response.finish_reason !== nothing ?
                    string(response.finish_reason) : "stop",
            ))
        else
            # Workflow execution
            result = AgentFramework.run_workflow(info.entity, message)
            outputs = AgentFramework.get_outputs(result)
            response_text = isempty(outputs) ? "Workflow completed." : string(first(outputs))
            assistant_msg = add_message!(conv_store, conv.id, :assistant, response_text)

            return json_response(Dict{String, Any}(
                "conversation_id" => conv.id,
                "message" => message_to_api_dict(assistant_msg),
                "finish_reason" => "stop",
            ))
        end
    catch e
        @error "Chat error" exception=(e, catch_backtrace())
        return error_response("Agent execution error: $(sprint(showerror, e))"; status=500)
    end
end

"""
    handle_chat_stream(registry, conv_store, body, io) → nothing

SSE streaming chat endpoint. Writes directly to the IO stream.
"""
function handle_chat_stream(registry::EntityRegistry, conv_store::ConversationStore,
                            body::String, io::IO)
    local parsed
    try
        parsed = JSON3.read(body, Dict{String, Any})
    catch e
        writer = SSEWriter(io)
        write_sse(writer, "error", Dict("error" => "Invalid JSON body"))
        write_sse_done(writer)
        return
    end

    entity_id = get(parsed, "entity_id", nothing)
    message = get(parsed, "message", nothing)
    conversation_id = get(parsed, "conversation_id", nothing)

    writer = SSEWriter(io)

    if entity_id === nothing || message === nothing
        write_sse(writer, "error", Dict("error" => "Missing required fields: entity_id, message"))
        write_sse_done(writer)
        return
    end

    info = get_entity(registry, entity_id)
    if info === nothing
        write_sse(writer, "error", Dict("error" => "Entity not found: $entity_id"))
        write_sse_done(writer)
        return
    end

    if info.type != AGENT_ENTITY
        write_sse(writer, "error", Dict("error" => "Streaming is only supported for agents"))
        write_sse_done(writer)
        return
    end

    # Get or create conversation
    conv = if conversation_id !== nothing
        get_conversation(conv_store, conversation_id)
    else
        nothing
    end
    if conv === nothing
        session = AgentFramework.AgentSession()
        conv = create_conversation!(conv_store, entity_id; session=session)
    end

    # Send conversation_id so the client can continue the conversation
    write_sse(writer, "conversation.info", Dict("conversation_id" => conv.id))

    add_message!(conv_store, conv.id, :user, message)

    # Auto-set title from first user message if still default
    _set_initial_conversation_title!(conv_store, conv, message)

    # Stream the response and save the accumulated text
    response_text = stream_agent_response(writer, info.entity, message, conv.session)
    if !isempty(response_text)
        add_message!(conv_store, conv.id, :assistant, response_text)
    end
end

function handle_chat_stream_agui(registry::EntityRegistry, conv_store::ConversationStore,
                                 entity_id::AbstractString, body::String, io::IO)
    writer = SSEWriter(io)

    local request
    try
        request = parse_agui_request(body)
    catch e
        write_agui_error(writer, "Invalid AG-UI request: $(sprint(showerror, e))";
                         code="INVALID_REQUEST")
        return
    end

    info = get_entity(registry, entity_id)
    if info === nothing
        write_agui_error(writer, "Entity not found: $entity_id"; code="ENTITY_NOT_FOUND")
        return
    end

    local input_messages
    try
        input_messages = agui_messages_to_framework(request["messages"])
    catch e
        write_agui_error(writer, "Invalid AG-UI messages: $(sprint(showerror, e))";
                         code="INVALID_MESSAGES")
        return
    end

    local history_messages
    local current_message
    try
        history_messages, current_message = split_agui_messages(input_messages)
    catch e
        write_agui_error(writer, sprint(showerror, e); code="INVALID_MESSAGES")
        return
    end

    requested_thread_id = request["thread_id"]
    conv = requested_thread_id !== nothing ? get_conversation(conv_store, requested_thread_id) : nothing
    if conv !== nothing && conv.entity_id != entity_id
        write_agui_error(writer, "Thread does not belong to entity: $entity_id";
                         code="THREAD_MISMATCH")
        return
    end

    is_new_conversation = conv === nothing
    if conv === nothing
        conversation_id = requested_thread_id === nothing ? string(UUIDs.uuid4()) : String(requested_thread_id)
        session = info.type == AGENT_ENTITY ? AgentFramework.AgentSession(id=conversation_id) : nothing
        conv = create_conversation!(conv_store, entity_id;
                                    session=session, conversation_id=conversation_id)
    end

    if is_new_conversation && !isempty(history_messages)
        _seed_agui_conversation_history!(conv_store, conv, history_messages)
        if info.type == AGENT_ENTITY && conv.session !== nothing
            _save_history_messages!(info.entity, conv.session, history_messages)
        end
    end

    title_source = _first_user_text(input_messages)
    title_source !== nothing && _set_initial_conversation_title!(conv_store, conv, title_source)

    user_text = AgentFramework.get_text(current_message)
    add_message!(conv_store, conv.id, :user, user_text)

    run_id = String(request["run_id"])
    if info.type == AGENT_ENTITY
        result = stream_agent_response_agui(writer, info.entity, user_text, conv.session;
                                            thread_id=conv.id, run_id=run_id)
        if result.response !== nothing
            tool_calls, tool_results = _response_tool_parts(result.response)
            assistant_text = AgentFramework.get_text(result.response)
            if !isempty(assistant_text) || !isempty(tool_calls) || !isempty(tool_results)
                add_message!(conv_store, conv.id, :assistant, assistant_text;
                             tool_calls=tool_calls, tool_results=tool_results)
            end
        end
    else
        response_text = stream_workflow_response_agui(writer, info.entity, user_text;
                                                      thread_id=conv.id, run_id=run_id)
        if !isempty(response_text)
            add_message!(conv_store, conv.id, :assistant, response_text)
        end
    end
end

"""
    handle_list_conversations(conv_store) → HTTP.Response

List all conversations.
"""
function handle_list_conversations(conv_store::ConversationStore)
    convs = list_conversations(conv_store)
    return json_response([conversation_to_dict(c; include_messages=false) for c in convs])
end

"""
    handle_get_conversation(conv_store, id) → HTTP.Response

Get a conversation with all messages.
"""
function handle_get_conversation(conv_store::ConversationStore, id::AbstractString)
    conv = get_conversation(conv_store, id)
    if conv === nothing
        return error_response("Conversation not found: $id"; status=404)
    end
    return json_response(conversation_to_dict(conv))
end

"""
    handle_delete_conversation(conv_store, id) → HTTP.Response

Delete a conversation.
"""
function handle_delete_conversation(conv_store::ConversationStore, id::AbstractString)
    delete_conversation!(conv_store, id)
    return HTTP.Response(204, ["Access-Control-Allow-Origin" => "*"])
end

"""
    handle_rename_conversation(conv_store, id, body) → HTTP.Response

Rename a conversation. Body: `{"title": "new name"}`
"""
function handle_rename_conversation(conv_store::ConversationStore, id::AbstractString, body::AbstractString)
    local parsed
    try
        parsed = JSON3.read(body, Dict{String, Any})
    catch e
        return error_response("Invalid JSON body")
    end
    title = get(parsed, "title", nothing)
    if title === nothing || !isa(title, AbstractString)
        return error_response("Missing required field: title")
    end
    if rename_conversation!(conv_store, id, title)
        return json_response(Dict("status" => "ok", "title" => title))
    else
        return error_response("Conversation not found: $id"; status=404)
    end
end
