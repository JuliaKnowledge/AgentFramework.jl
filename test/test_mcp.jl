using AgentFramework
using Test
using JSON3

@testset "MCP Client" begin

    # ─── Type Construction ───────────────────────────────────────────────────

    @testset "MCPToolInfo construction" begin
        tool = MCPToolInfo(name="read_file", description="Read a file")
        @test tool.name == "read_file"
        @test tool.description == "Read a file"
        @test tool.input_schema == Dict{String, Any}()

        tool2 = MCPToolInfo(
            name = "write",
            description = "Write data",
            input_schema = Dict{String, Any}("type" => "object"),
        )
        @test tool2.input_schema["type"] == "object"
    end

    @testset "MCPResource construction" begin
        r = MCPResource(uri="file:///tmp/test.txt")
        @test r.uri == "file:///tmp/test.txt"
        @test r.name == ""
        @test r.mime_type === nothing

        r2 = MCPResource(uri="file:///a.json", name="a", mime_type="application/json")
        @test r2.name == "a"
        @test r2.mime_type == "application/json"
    end

    @testset "MCPPrompt construction" begin
        p = MCPPrompt(name="summarize")
        @test p.name == "summarize"
        @test p.description == ""
        @test isempty(p.arguments)

        p2 = MCPPrompt(
            name = "translate",
            description = "Translate text",
            arguments = [Dict{String, Any}("name" => "language", "required" => true)],
        )
        @test length(p2.arguments) == 1
        @test p2.arguments[1]["name"] == "language"
    end

    @testset "MCPToolResult construction" begin
        r = MCPToolResult()
        @test isempty(r.content)
        @test r.is_error == false

        r2 = MCPToolResult(
            content = [Dict{String, Any}("type" => "text", "text" => "hello")],
            is_error = true,
        )
        @test length(r2.content) == 1
        @test r2.is_error == true
    end

    @testset "MCPServerCapabilities construction" begin
        caps = MCPServerCapabilities()
        @test caps.tools == false
        @test caps.resources == false
        @test caps.prompts == false
        @test caps.logging == false

        caps2 = MCPServerCapabilities(tools=true, prompts=true)
        @test caps2.tools == true
        @test caps2.prompts == true
        @test caps2.resources == false
    end

    # ─── Client Construction ─────────────────────────────────────────────────

    @testset "StdioMCPClient construction" begin
        client = StdioMCPClient(command="echo", args=["hello"])
        @test client.command == "echo"
        @test client.args == ["hello"]
        @test client.env == Dict{String, String}()
        @test client.server_name == "mcp-server"
        @test client._process === nothing
        @test client._initialized == false
        @test client._request_id == 0
    end

    @testset "HTTPMCPClient construction" begin
        client = HTTPMCPClient(url="http://localhost:8080/mcp")
        @test client.url == "http://localhost:8080/mcp"
        @test client.headers == Dict{String, String}()
        @test client.server_name == "mcp-http-server"
        @test client._initialized == false
        @test client._session_id === nothing

        client2 = HTTPMCPClient(
            url = "http://example.com/mcp",
            headers = Dict("Authorization" => "Bearer tok"),
        )
        @test client2.headers["Authorization"] == "Bearer tok"
    end

    # ─── Name Normalization ──────────────────────────────────────────────────

    @testset "_normalize_mcp_name" begin
        @test AgentFramework._normalize_mcp_name("read-file") == "read-file"
        @test AgentFramework._normalize_mcp_name("my.tool.name") == "my.tool.name"
        @test AgentFramework._normalize_mcp_name("tool/action") == "tool-action"
        @test AgentFramework._normalize_mcp_name("already_valid_123") == "already_valid_123"
        @test AgentFramework._normalize_mcp_name("special@chars!here") == "special-chars-here"
        @test AgentFramework._normalize_mcp_name("with spaces") == "with-spaces"
    end

    @testset "_build_prefixed_mcp_name" begin
        @test AgentFramework._build_prefixed_mcp_name("read-file", nothing) == "read-file"
        @test AgentFramework._build_prefixed_mcp_name("read-file", "myserver") == "myserver_read-file"
        @test AgentFramework._build_prefixed_mcp_name("read-file", "my server") == "my-server_read-file"
        @test AgentFramework._build_prefixed_mcp_name("read-file", "") == "read-file"
        @test AgentFramework._build_prefixed_mcp_name("_leading", "prefix") == "prefix_leading"
        @test AgentFramework._build_prefixed_mcp_name("tool", "prefix_") == "prefix_tool"
    end

    @testset "MCPSpecificApproval" begin
        approval = MCPSpecificApproval(
            always_require_approval=["dangerous_tool"],
            never_require_approval=["safe_tool"],
        )
        @test "dangerous_tool" in approval.always_require_approval
        @test "safe_tool" in approval.never_require_approval

        default_approval = MCPSpecificApproval()
        @test default_approval.always_require_approval === nothing
        @test default_approval.never_require_approval === nothing
    end

    @testset "stdio transport uses Content-Length framing" begin
        payload = """{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"""
        io = IOBuffer()
        AgentFramework._write_stdio_message(io, payload)
        seekstart(io)
        @test AgentFramework._read_stdio_message(io) == payload
    end

    @testset "_send_request writes framed stdio JSON-RPC" begin
        client = StdioMCPClient(command="echo")
        client._input = IOBuffer()
        client._output = IOBuffer()

        AgentFramework._write_stdio_message(
            client._output,
            JSON3.write(Dict{String, Any}(
                "jsonrpc" => "2.0",
                "id" => 1,
                "result" => Dict{String, Any}("ok" => true),
            )),
        )
        seekstart(client._output)

        result = AgentFramework._send_request(client, "ping", Dict{String, Any}("value" => 42))
        @test result["ok"] == true

        request_text = String(take!(client._input))
        parts = split(request_text, "\r\n\r\n"; limit=2)
        @test length(parts) == 2
        @test startswith(parts[1], "Content-Length: ")

        request = JSON3.read(parts[2], Dict{String, Any})
        @test request["jsonrpc"] == "2.0"
        @test request["method"] == "ping"
        @test request["params"]["value"] == 42
    end

    # ─── MCP → FunctionTool Conversion ───────────────────────────────────────

    @testset "mcp_tool_to_function_tool" begin
        tool_info = MCPToolInfo(
            name = "read-file",
            description = "Read a file from disk",
            input_schema = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "path" => Dict{String, Any}("type" => "string", "description" => "File path"),
                ),
                "required" => ["path"],
            ),
        )
        client = StdioMCPClient(command="echo", args=["test"])
        ft = mcp_tool_to_function_tool(client, tool_info)
        @test ft isa FunctionTool
        @test ft.name == "read-file"  # preserves hyphens (valid in MCP names)
        @test ft.description == "Read a file from disk"
        @test ft.parameters == tool_info.input_schema
        @test ft.func isa Function
    end

    @testset "mcp_tool_to_function_tool with prefix" begin
        tool_info = MCPToolInfo(name="list-files", description="List files")
        client = StdioMCPClient(command="echo")
        ft = mcp_tool_to_function_tool(client, tool_info; tool_name_prefix="myserver")
        @test ft.name == "myserver_list-files"
    end

    @testset "mcp_tools_to_function_tools" begin
        tools = [
            MCPToolInfo(name="tool-a", description="Tool A"),
            MCPToolInfo(name="tool_b", description="Tool B"),
            MCPToolInfo(name="tool.c", description="Tool C"),
        ]
        client = StdioMCPClient(command="echo")
        fts = mcp_tools_to_function_tools(client, tools)
        @test length(fts) == 3
        @test fts[1].name == "tool-a"
        @test fts[2].name == "tool_b"
        @test fts[3].name == "tool.c"
    end

    @testset "mcp_tools_to_function_tools deduplication" begin
        tools = [
            MCPToolInfo(name="read file", description="Tool A"),
            MCPToolInfo(name="read!file", description="Tool B"),  # both normalize to "read-file"
        ]
        client = StdioMCPClient(command="echo")
        fts = mcp_tools_to_function_tools(client, tools)
        @test length(fts) == 1  # second is deduplicated
        @test fts[1].description == "Tool A"  # first wins
    end

    @testset "mcp_tools_to_function_tools with prefix" begin
        tools = [
            MCPToolInfo(name="read", description="Read"),
            MCPToolInfo(name="write", description="Write"),
        ]
        client = StdioMCPClient(command="echo")
        fts = mcp_tools_to_function_tools(client, tools; tool_name_prefix="fs")
        @test fts[1].name == "fs_read"
        @test fts[2].name == "fs_write"
    end

    # ─── Connection State ────────────────────────────────────────────────────

    @testset "is_connected returns false for uninitialized client" begin
        client = StdioMCPClient(command="echo")
        @test is_connected(client) == false

        http_client = HTTPMCPClient(url="http://localhost:9999/mcp")
        @test is_connected(http_client) == false
    end

    @testset "close_mcp! on unconnected client doesn't error" begin
        client = StdioMCPClient(command="echo")
        close_mcp!(client)  # should not throw
        @test client._initialized == false

        http_client = HTTPMCPClient(url="http://localhost:9999/mcp")
        close_mcp!(http_client)
        @test http_client._initialized == false
    end

    # ─── Error Handling ──────────────────────────────────────────────────────

    @testset "call_tool on uninitialized client throws" begin
        client = StdioMCPClient(command="echo")
        @test_throws AgentFrameworkError call_tool(client, "some_tool")
    end

    @testset "list_tools on uninitialized client throws" begin
        client = StdioMCPClient(command="echo")
        @test_throws AgentFrameworkError list_tools(client)
    end

    @testset "list_resources on uninitialized client throws" begin
        client = StdioMCPClient(command="echo")
        @test_throws AgentFrameworkError list_resources(client)
    end

    @testset "read_resource on uninitialized client throws" begin
        client = StdioMCPClient(command="echo")
        @test_throws AgentFrameworkError read_resource(client, "file:///tmp/x")
    end

    @testset "list_prompts on uninitialized client throws" begin
        client = StdioMCPClient(command="echo")
        @test_throws AgentFrameworkError list_prompts(client)
    end

    @testset "get_prompt on uninitialized client throws" begin
        client = StdioMCPClient(command="echo")
        @test_throws AgentFrameworkError get_prompt(client, "test")
    end

    @testset "connect! throws for invalid command" begin
        client = StdioMCPClient(command="__nonexistent_binary_42__")
        @test_throws Exception connect!(client)
    end

    @testset "HTTPMCPClient connect! throws for unreachable URL" begin
        client = HTTPMCPClient(url="http://127.0.0.1:1/mcp")
        @test_throws Exception connect!(client)
    end

    # ─── with_mcp_client ensures cleanup ─────────────────────────────────────

    @testset "with_mcp_client ensures cleanup on error" begin
        client = StdioMCPClient(command="__nonexistent_binary_42__")
        try
            with_mcp_client(client) do c
                error("should not reach here")
            end
        catch
            # Expected — connect! fails
        end
        @test client._initialized == false
    end

end
