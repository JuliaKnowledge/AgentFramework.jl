# MCP (Model Context Protocol)

MCP integration allows agents to discover and invoke tools hosted by external
MCP servers over stdio or HTTP transports. Use [`connect!`](@ref) to establish
a connection, [`list_tools`](@ref) to discover available tools, and
[`mcp_tools_to_function_tools`](@ref) to convert them into
[`FunctionTool`](@ref) instances the agent can use directly.

## Client Types

```@docs
AgentFramework.AbstractMCPClient
AgentFramework.StdioMCPClient
AgentFramework.HTTPMCPClient
```

## Data Types

```@docs
AgentFramework.MCPToolInfo
AgentFramework.MCPResource
AgentFramework.MCPPrompt
AgentFramework.MCPToolResult
AgentFramework.MCPServerCapabilities
AgentFramework.MCPSpecificApproval
```

## Connection Management

```@docs
AgentFramework.connect!
AgentFramework.close_mcp!
AgentFramework.is_connected
AgentFramework.with_mcp_client
```

## Tool Discovery and Invocation

```@docs
AgentFramework.list_tools
AgentFramework.call_tool
```

## Resource and Prompt Access

```@docs
AgentFramework.list_resources
AgentFramework.read_resource
AgentFramework.list_prompts
AgentFramework.get_prompt
```

## Tool Conversion

```@docs
AgentFramework.mcp_tool_to_function_tool
AgentFramework.mcp_tools_to_function_tools
AgentFramework.load_mcp_tools
```
