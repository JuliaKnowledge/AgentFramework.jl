# Model Context Protocol (MCP) client support for AgentFramework.jl
# Implements JSON-RPC 2.0 over stdio (subprocess) and HTTP transports.

# ─── MCP Types ───────────────────────────────────────────────────────────────

"""MCP tool description from server."""
Base.@kwdef struct MCPToolInfo
    name::String
    description::String = ""
    input_schema::Dict{String, Any} = Dict{String, Any}()
end

"""MCP resource description."""
Base.@kwdef struct MCPResource
    uri::String
    name::String = ""
    description::String = ""
    mime_type::Union{Nothing, String} = nothing
end

"""MCP prompt description."""
Base.@kwdef struct MCPPrompt
    name::String
    description::String = ""
    arguments::Vector{Dict{String, Any}} = Dict{String, Any}[]
end

"""Result from calling an MCP tool."""
Base.@kwdef struct MCPToolResult
    content::Vector{Dict{String, Any}} = Dict{String, Any}[]
    is_error::Bool = false
end

"""MCP server capabilities discovered during initialization."""
Base.@kwdef struct MCPServerCapabilities
    tools::Bool = false
    resources::Bool = false
    prompts::Bool = false
    logging::Bool = false
end

# ─── Abstract MCP Client ────────────────────────────────────────────────────

"""Abstract base for MCP clients."""
abstract type AbstractMCPClient end

"""Get available tools from the MCP server."""
function list_tools end

"""Call a tool on the MCP server."""
function call_tool end

"""Get available resources."""
function list_resources end

"""Read a resource by URI."""
function read_resource end

"""Get available prompts."""
function list_prompts end

"""Get a prompt by name with arguments."""
function get_prompt end

"""Close the MCP connection."""
function close_mcp! end

"""Check if client is connected."""
function is_connected end

"""Connect to the MCP server and perform initialization handshake."""
function connect! end

# ─── Stdio MCP Client ───────────────────────────────────────────────────────

"""
    StdioMCPClient <: AbstractMCPClient

MCP client using subprocess stdio transport.

# Example
```julia
client = StdioMCPClient(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
connect!(client)
tools = list_tools(client)
result = call_tool(client, "read_file", Dict("path" => "/tmp/test.txt"))
close_mcp!(client)
```
"""
Base.@kwdef mutable struct StdioMCPClient <: AbstractMCPClient
    command::String
    args::Vector{String} = String[]
    env::Dict{String, String} = Dict{String, String}()
    server_name::String = "mcp-server"
    _process::Union{Nothing, Base.Process} = nothing
    _input::Union{Nothing, IO} = nothing
    _output::Union{Nothing, IO} = nothing
    _request_id::Int = 0
    _lock::ReentrantLock = ReentrantLock()
    _capabilities::MCPServerCapabilities = MCPServerCapabilities()
    _initialized::Bool = false
end

# ─── HTTP MCP Client ────────────────────────────────────────────────────────

"""
    HTTPMCPClient <: AbstractMCPClient

MCP client using HTTP transport (Streamable HTTP).

# Example
```julia
client = HTTPMCPClient(url="http://localhost:8080/mcp")
connect!(client)
tools = list_tools(client)
close_mcp!(client)
```
"""
Base.@kwdef mutable struct HTTPMCPClient <: AbstractMCPClient
    url::String
    headers::Dict{String, String} = Dict{String, String}()
    server_name::String = "mcp-http-server"
    _request_id::Int = 0
    _lock::ReentrantLock = ReentrantLock()
    _capabilities::MCPServerCapabilities = MCPServerCapabilities()
    _initialized::Bool = false
    _session_id::Union{Nothing, String} = nothing
end

# ─── Shared Helpers ──────────────────────────────────────────────────────────

function _next_id!(client::AbstractMCPClient)
    lock(client._lock) do
        client._request_id += 1
        return client._request_id
    end
end

"""Normalize MCP tool/prompt names to valid identifier-like pattern [A-Za-z0-9_.-]."""
function _normalize_mcp_name(name::String)::String
    replace(name, r"[^A-Za-z0-9_.\-]" => "-")
end

"""Build prefixed MCP tool name from a normalized name and optional prefix."""
function _build_prefixed_mcp_name(normalized_name::String, prefix::Union{Nothing, String})::String
    prefix === nothing && return normalized_name
    norm_prefix = rstrip(_normalize_mcp_name(prefix), ['_', '.', '-'])
    isempty(norm_prefix) && return normalized_name
    trimmed = lstrip(normalized_name, ['_', '.', '-'])
    isempty(trimmed) && return norm_prefix
    return "$(norm_prefix)_$(trimmed)"
end

"""Per-tool approval specification for MCP tools."""
Base.@kwdef struct MCPSpecificApproval
    always_require_approval::Union{Nothing, Vector{String}} = nothing
    never_require_approval::Union{Nothing, Vector{String}} = nothing
end

# ─── Stdio Transport ────────────────────────────────────────────────────────

function _write_stdio_message(io::IO, payload::AbstractString)
    write(io, "Content-Length: $(sizeof(payload))\r\n\r\n")
    write(io, payload)
    flush(io)
    return nothing
end

function _read_stdio_message(io::IO)::String
    headers = Dict{String, String}()

    while true
        line = try
            String(readuntil(io, UInt8('\n')))
        catch e
            if e isa EOFError
                throw(AgentFrameworkError("Unexpected EOF while reading MCP stdio headers", e))
            end
            rethrow(e)
        end

        line = strip(line, ['\r', '\n'])
        isempty(line) && break

        parts = split(line, ":"; limit=2)
        length(parts) == 2 || throw(AgentFrameworkError("Invalid MCP stdio header: $line"))
        headers[strip(parts[1])] = strip(parts[2])
    end

    length_header = get(headers, "Content-Length", nothing)
    length_header === nothing && throw(AgentFrameworkError("MCP stdio message missing Content-Length header"))
    content_length = tryparse(Int, length_header)
    content_length === nothing && throw(AgentFrameworkError("Invalid MCP Content-Length header: $length_header"))

    body = Vector{UInt8}(undef, content_length)
    bytes_read = readbytes!(io, body, content_length)
    bytes_read == content_length || throw(AgentFrameworkError("Unexpected EOF while reading MCP stdio body"))

    return String(body)
end

function _send_request(client::StdioMCPClient, method::String, params::Dict{String, Any} = Dict{String, Any}())
    id = _next_id!(client)
    request = Dict{String, Any}(
        "jsonrpc" => "2.0",
        "id" => id,
        "method" => method,
        "params" => params,
    )
    msg = JSON3.write(request)
    _write_stdio_message(client._input, msg)

    response = JSON3.read(_read_stdio_message(client._output), Dict{String, Any})

    if haskey(response, "error")
        err = response["error"]
        throw(AgentFrameworkError("MCP error $(get(err, "code", -1)): $(get(err, "message", "unknown"))"))
    end

    return get(response, "result", Dict{String, Any}())
end

function _send_notification(client::StdioMCPClient, method::String, params::Dict{String, Any} = Dict{String, Any}())
    notification = Dict{String, Any}(
        "jsonrpc" => "2.0",
        "method" => method,
    )
    if !isempty(params)
        notification["params"] = params
    end
    msg = JSON3.write(notification)
    _write_stdio_message(client._input, msg)
end

function connect!(client::StdioMCPClient)
    cmd = Cmd(`$(client.command) $(client.args)`)
    if !isempty(client.env)
        cmd = addenv(cmd, client.env...)
    end

    client._process = open(cmd, "r+")
    client._input = client._process.in
    client._output = client._process.out

    _mcp_initialize!(client)
    return client
end

function close_mcp!(client::StdioMCPClient)
    if client._process !== nothing
        try close(client._input) catch end
        try kill(client._process) catch end
        client._process = nothing
        client._input = nothing
        client._output = nothing
        client._initialized = false
    end
end

function is_connected(client::StdioMCPClient)::Bool
    client._process !== nothing && process_running(client._process)
end

# ─── HTTP Transport ──────────────────────────────────────────────────────────

function _send_request(client::HTTPMCPClient, method::String, params::Dict{String, Any} = Dict{String, Any}())
    id = _next_id!(client)
    request = Dict{String, Any}(
        "jsonrpc" => "2.0",
        "id" => id,
        "method" => method,
        "params" => params,
    )

    headers = merge(
        Dict("Content-Type" => "application/json", "Accept" => "application/json"),
        client.headers,
    )
    if client._session_id !== nothing
        headers["Mcp-Session-Id"] = client._session_id
    end

    body = JSON3.write(request)
    resp = HTTP.post(client.url, collect(pairs(headers)), body; status_exception = false)

    if resp.status != 200
        throw(AgentFrameworkError("MCP HTTP error: status $(resp.status)"))
    end

    session_header = HTTP.header(resp, "Mcp-Session-Id", "")
    if !isempty(session_header)
        client._session_id = session_header
    end

    response = JSON3.read(String(resp.body), Dict{String, Any})

    if haskey(response, "error")
        err = response["error"]
        throw(AgentFrameworkError("MCP error $(get(err, "code", -1)): $(get(err, "message", "unknown"))"))
    end

    return get(response, "result", Dict{String, Any}())
end

function _send_notification(client::HTTPMCPClient, method::String, params::Dict{String, Any} = Dict{String, Any}())
    notification = Dict{String, Any}("jsonrpc" => "2.0", "method" => method)
    if !isempty(params)
        notification["params"] = params
    end
    headers = merge(Dict("Content-Type" => "application/json"), client.headers)
    if client._session_id !== nothing
        headers["Mcp-Session-Id"] = client._session_id
    end
    HTTP.post(client.url, collect(pairs(headers)), JSON3.write(notification); status_exception = false)
end

function connect!(client::HTTPMCPClient)
    _mcp_initialize!(client)
    return client
end

function close_mcp!(client::HTTPMCPClient)
    client._initialized = false
    client._session_id = nothing
end

function is_connected(client::HTTPMCPClient)::Bool
    client._initialized
end

# ─── Protocol Handshake ─────────────────────────────────────────────────────

function _mcp_initialize!(client::AbstractMCPClient)
    result = _send_request(client, "initialize", Dict{String, Any}(
        "protocolVersion" => "2024-11-05",
        "capabilities" => Dict{String, Any}(),
        "clientInfo" => Dict{String, Any}(
            "name" => "AgentFramework.jl",
            "version" => "0.1.0",
        ),
    ))

    caps = get(result, "capabilities", Dict{String, Any}())
    client._capabilities = MCPServerCapabilities(
        tools = haskey(caps, "tools"),
        resources = haskey(caps, "resources"),
        prompts = haskey(caps, "prompts"),
        logging = haskey(caps, "logging"),
    )

    _send_notification(client, "notifications/initialized")
    client._initialized = true
end

# ─── Tool Operations (generic for both transports) ──────────────────────────

function list_tools(client::AbstractMCPClient)::Vector{MCPToolInfo}
    client._initialized || throw(AgentFrameworkError("MCP client not initialized. Call connect! first."))

    tools = MCPToolInfo[]
    cursor = nothing

    while true
        params = Dict{String, Any}()
        cursor !== nothing && (params["cursor"] = cursor)

        result = _send_request(client, "tools/list", params)

        for tool_data in get(result, "tools", [])
            push!(tools, MCPToolInfo(
                name = tool_data["name"],
                description = get(tool_data, "description", ""),
                input_schema = get(tool_data, "inputSchema", Dict{String, Any}()),
            ))
        end

        cursor = get(result, "nextCursor", nothing)
        cursor === nothing && break
    end

    return tools
end

function call_tool(client::AbstractMCPClient, name::String, arguments::Dict{String, Any} = Dict{String, Any}())::MCPToolResult
    client._initialized || throw(AgentFrameworkError("MCP client not initialized. Call connect! first."))

    result = _send_request(client, "tools/call", Dict{String, Any}(
        "name" => name,
        "arguments" => arguments,
    ))

    MCPToolResult(
        content = get(result, "content", Dict{String, Any}[]),
        is_error = get(result, "isError", false),
    )
end

function list_resources(client::AbstractMCPClient)::Vector{MCPResource}
    client._initialized || throw(AgentFrameworkError("MCP client not initialized. Call connect! first."))
    result = _send_request(client, "resources/list")
    [MCPResource(
        uri = r["uri"],
        name = get(r, "name", ""),
        description = get(r, "description", ""),
        mime_type = get(r, "mimeType", nothing),
    ) for r in get(result, "resources", [])]
end

function read_resource(client::AbstractMCPClient, uri::String)::Vector{Dict{String, Any}}
    client._initialized || throw(AgentFrameworkError("MCP client not initialized. Call connect! first."))
    result = _send_request(client, "resources/read", Dict{String, Any}("uri" => uri))
    get(result, "contents", Dict{String, Any}[])
end

function list_prompts(client::AbstractMCPClient)::Vector{MCPPrompt}
    client._initialized || throw(AgentFrameworkError("MCP client not initialized. Call connect! first."))
    result = _send_request(client, "prompts/list")
    [MCPPrompt(
        name = p["name"],
        description = get(p, "description", ""),
        arguments = get(p, "arguments", Dict{String, Any}[]),
    ) for p in get(result, "prompts", [])]
end

function get_prompt(client::AbstractMCPClient, name::String, arguments::Dict{String, Any} = Dict{String, Any}())::Dict{String, Any}
    client._initialized || throw(AgentFrameworkError("MCP client not initialized. Call connect! first."))
    _send_request(client, "prompts/get", Dict{String, Any}("name" => name, "arguments" => arguments))
end

# ─── MCP → FunctionTool Conversion ──────────────────────────────────────────

"""
    mcp_tool_to_function_tool(client, tool; tool_name_prefix=nothing) -> FunctionTool

Convert an MCPToolInfo to an AgentFramework FunctionTool.

Supports structuredContent fallback: if tool result has no text content,
falls back to structuredContent field. Tool names are normalized and
optionally prefixed for deduplication.
"""
function mcp_tool_to_function_tool(client::AbstractMCPClient, tool::MCPToolInfo;
                                    tool_name_prefix::Union{Nothing, String}=nothing)::FunctionTool
    original_name = tool.name
    normalized = _normalize_mcp_name(tool.name)
    exposed_name = _build_prefixed_mcp_name(normalized, tool_name_prefix)

    invoke_fn = (args::Dict{String, Any}) -> begin
        result = call_tool(client, original_name, args)
        if result.is_error
            content_texts = [get(c, "text", "") for c in result.content if get(c, "type", "") == "text"]
            error_msg = isempty(content_texts) ? "MCP tool error" : join(content_texts, "\n")
            throw(ToolExecutionError(error_msg))
        end
        # Try text content first
        texts = [get(c, "text", "") for c in result.content if get(c, "type", "") == "text"]
        if !isempty(texts)
            return join(texts, "\n")
        end
        # Fallback to structuredContent if present
        for c in result.content
            sc = get(c, "structuredContent", nothing)
            if sc !== nothing
                return sc isa AbstractString ? sc : JSON3.write(sc)
            end
        end
        # Last resort: serialize all content
        return JSON3.write(result.content)
    end

    metadata = Dict{String, Any}()
    if exposed_name != original_name
        metadata["_mcp_original_name"] = original_name
        metadata["_mcp_normalized_name"] = normalized
    end
    if tool_name_prefix !== nothing
        metadata["_mcp_tool_name_prefix"] = tool_name_prefix
    end

    FunctionTool(
        name = exposed_name,
        description = tool.description,
        func = invoke_fn,
        parameters = tool.input_schema,
    )
end

"""Convert MCP tools to AgentFramework FunctionTool objects, with optional name prefixing."""
function mcp_tools_to_function_tools(client::AbstractMCPClient, tools::Vector{MCPToolInfo};
                                      tool_name_prefix::Union{Nothing, String}=nothing)::Vector{FunctionTool}
    seen_names = Set{String}()
    result = FunctionTool[]
    for tool in tools
        ft = mcp_tool_to_function_tool(client, tool; tool_name_prefix=tool_name_prefix)
        # Deduplicate: skip tools with names we've already seen
        if ft.name ∉ seen_names
            push!(seen_names, ft.name)
            push!(result, ft)
        else
            @warn "Skipping duplicate MCP tool" name=ft.name original=tool.name
        end
    end
    return result
end

"""Connect to MCP server and return all tools as FunctionTools."""
function load_mcp_tools(client::AbstractMCPClient;
                         tool_name_prefix::Union{Nothing, String}=nothing)::Vector{FunctionTool}
    if !is_connected(client)
        connect!(client)
    end
    tools = list_tools(client)
    return mcp_tools_to_function_tools(client, tools; tool_name_prefix=tool_name_prefix)
end

# ─── Convenience ─────────────────────────────────────────────────────────────

"""Execute a block with an MCP client, ensuring cleanup."""
function with_mcp_client(f::Function, client::AbstractMCPClient)
    try
        connect!(client)
        return f(client)
    finally
        close_mcp!(client)
    end
end
