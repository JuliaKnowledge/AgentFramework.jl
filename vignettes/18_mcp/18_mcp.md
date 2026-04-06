# MCP Tool Integration
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [What Is MCP?](#what-is-mcp)
- [StdioMCPClient — Subprocess
  Transport](#stdiomcpclient--subprocess-transport)
- [HTTPMCPClient — HTTP Transport](#httpmcpclient--http-transport)
- [Converting MCP Tools to
  FunctionTools](#converting-mcp-tools-to-functiontools)
- [Using MCP Tools with an Agent](#using-mcp-tools-with-an-agent)
- [Safe Cleanup with
  `with_mcp_client`](#safe-cleanup-with-with_mcp_client)
- [Multiple MCP Servers](#multiple-mcp-servers)
- [Resources and Prompts](#resources-and-prompts)
  - [Listing and Reading Resources](#listing-and-reading-resources)
  - [Listing and Invoking Prompts](#listing-and-invoking-prompts)
- [Key Types Reference](#key-types-reference)
- [Summary](#summary)

## Overview

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) is
an open standard that lets LLM applications access external tools,
resources, and prompts through a uniform interface. Instead of
hard-coding integrations, you connect to any MCP-compliant server and
its capabilities become available to your agent automatically.

In this vignette you will learn how to:

- Understand the MCP architecture and its role in agent systems.
- Connect to an MCP server over **stdio** (subprocess) or **HTTP**
  transports.
- List and call tools, resources, and prompts exposed by an MCP server.
- Convert MCP tools into `FunctionTool` objects for use with an `Agent`.
- Use the `with_mcp_client` convenience pattern for safe cleanup.

## Prerequisites

- **Ollama** running locally with a model:

  ``` bash
  ollama pull qwen3:8b
  ```

- **Node.js / npx** available on your `PATH` (used to launch the example
  filesystem MCP server):

  ``` bash
  npx -y @modelcontextprotocol/server-filesystem /tmp
  ```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using AgentFramework
```

## What Is MCP?

MCP defines a JSON-RPC 2.0 protocol between a **client** (your
application) and one or more **servers** that expose three capability
types:

| Capability    | Description                                        |
|---------------|----------------------------------------------------|
| **Tools**     | Functions the LLM can call (read files, query DBs) |
| **Resources** | Data the client can fetch (files, API responses)   |
| **Prompts**   | Reusable prompt templates with parameters          |

Communication happens over a *transport* — either a **stdio** subprocess
pipe or an **HTTP** endpoint. The protocol flow looks like this:

    Agent ──► FunctionTool ──► MCP Client ──► [stdio / HTTP] ──► MCP Server
                                                                     │
                                                           Tools, Resources, Prompts

The MCP client sends `initialize`, then discovers capabilities via
`tools/list`, `resources/list`, and `prompts/list`. Tool invocations use
`tools/call` with a name and argument dictionary.

## StdioMCPClient — Subprocess Transport

The `StdioMCPClient` launches an MCP server as a child process and
communicates over its stdin/stdout using JSON-RPC 2.0 with
`Content-Length` framing:

``` julia
client = StdioMCPClient(
    command = "npx",
    args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    server_name = "filesystem",
)
connect!(client)
println("Connected: ", is_connected(client))
```

Once connected, you can list the tools the server exposes:

``` julia
tools = list_tools(client)
for tool in tools
    println("  $(tool.name): $(tool.description)")
end
```

And call a tool directly, passing arguments as a `Dict`:

``` julia
result = call_tool(client, "read_file", Dict("path" => "/tmp/test.txt"))
println(result.content)
```

The `MCPToolResult` returned contains a `content` field (the tool output
as a string) and an `is_error` flag indicating whether the server
reported a failure.

Always close the client when done to terminate the subprocess:

``` julia
close_mcp!(client)
```

## HTTPMCPClient — HTTP Transport

For remote or long-running MCP servers, use `HTTPMCPClient`. It sends
JSON-RPC requests over HTTP POST and tracks the server session via an
`Mcp-Session-Id` header:

``` julia
client = HTTPMCPClient(
    url = "http://localhost:8080/mcp",
    headers = Dict("Authorization" => "Bearer my-token"),
    server_name = "my-api",
)
connect!(client)

tools = list_tools(client)
for tool in tools
    println("  $(tool.name): $(tool.description)")
end

close_mcp!(client)
```

Both `StdioMCPClient` and `HTTPMCPClient` implement the same
`AbstractMCPClient` interface, so all downstream code works identically
regardless of transport.

## Converting MCP Tools to FunctionTools

MCP tools are described by `MCPToolInfo` structs (name, description,
input schema). To use them with an `Agent`, convert them into
`FunctionTool` objects that the framework understands:

``` julia
# Convert a list of MCPToolInfo into FunctionTools
function_tools = mcp_tools_to_function_tools(client, tools; tool_name_prefix = "fs")
```

The `tool_name_prefix` avoids name collisions when combining tools from
multiple MCP servers. A tool named `read_file` from the filesystem
server becomes `fs_read_file`.

For a one-step shortcut that lists tools and converts them:

``` julia
function_tools = load_mcp_tools(client; tool_name_prefix = "fs")
```

You can also convert a single tool:

``` julia
single_tool = mcp_tool_to_function_tool(client, tools[1]; tool_name_prefix = "fs")
```

## Using MCP Tools with an Agent

Here is a complete example that connects to a filesystem MCP server and
gives the resulting tools to an agent:

``` julia
# 1. Create a chat client
chat_client = OllamaChatClient(model = "qwen3:8b")

# 2. Connect to the MCP server
mcp = StdioMCPClient(
    command = "npx",
    args = ["-y", "@modelcontextprotocol/server-filesystem", "."],
    server_name = "filesystem",
)
connect!(mcp)

# 3. Load tools from the MCP server
fs_tools = load_mcp_tools(mcp; tool_name_prefix = "fs")
println("Loaded $(length(fs_tools)) tools from MCP server")

# 4. Create an agent with MCP tools
agent = Agent(
    name = "FileAgent",
    instructions = "You can read and list files. Use the available tools to answer questions about the filesystem.",
    client = chat_client,
    tools = fs_tools,
)

# 5. Run the agent
response = run_agent(agent, "List the files in the current directory")
println(response.text)

# 6. Clean up
close_mcp!(mcp)
```

The agent sees `fs_read_file`, `fs_list_directory`, etc. as regular
tools. When the LLM decides to call one, the framework routes the
invocation through the MCP client to the server and returns the result.

## Safe Cleanup with `with_mcp_client`

The `with_mcp_client` function follows the Julia `do`-block pattern to
guarantee the MCP client is closed even if an error occurs — similar to
a Python context manager or C# `using` block:

``` julia
chat_client = OllamaChatClient(model = "qwen3:8b")

with_mcp_client(StdioMCPClient(
    command = "npx",
    args = ["-y", "@modelcontextprotocol/server-filesystem", "."],
    server_name = "filesystem",
)) do mcp
    tools = load_mcp_tools(mcp; tool_name_prefix = "fs")
    agent = Agent(
        name = "FileAgent",
        instructions = "You can read and list files.",
        client = chat_client,
        tools = tools,
    )
    response = run_agent(agent, "What files are here?")
    println(response.text)
end  # mcp client automatically closed
```

This is the recommended pattern for production code.

## Multiple MCP Servers

You can combine tools from several MCP servers into one agent. Use
distinct prefixes to keep tool names unambiguous:

``` julia
chat_client = OllamaChatClient(model = "qwen3:8b")

fs_mcp = StdioMCPClient(
    command = "npx",
    args = ["-y", "@modelcontextprotocol/server-filesystem", "."],
    server_name = "filesystem",
)
connect!(fs_mcp)

api_mcp = HTTPMCPClient(
    url = "http://localhost:8080/mcp",
    server_name = "api",
)
connect!(api_mcp)

# Combine tools with different prefixes
all_tools = vcat(
    load_mcp_tools(fs_mcp; tool_name_prefix = "fs"),
    load_mcp_tools(api_mcp; tool_name_prefix = "api"),
)

agent = Agent(
    name = "MultiServerAgent",
    instructions = "You have access to filesystem and API tools.",
    client = chat_client,
    tools = all_tools,
)

response = run_agent(agent, "List local files and fetch the API status")
println(response.text)

close_mcp!(fs_mcp)
close_mcp!(api_mcp)
```

## Resources and Prompts

Beyond tools, MCP servers can expose **resources** (readable data) and
**prompts** (reusable templates). These are accessed through the same
client:

### Listing and Reading Resources

``` julia
resources = list_resources(client)
for r in resources
    println("  $(r.name) ($(r.mime_type)): $(r.uri)")
end

# Read the content of a resource by URI
content = read_resource(client, resources[1].uri)
println(content)
```

Resources are useful for injecting context into an agent without going
through the tool-call cycle — for example, loading a configuration file
or schema.

### Listing and Invoking Prompts

``` julia
prompts = list_prompts(client)
for p in prompts
    println("  $(p.name): $(p.description)")
end

# Get a prompt with arguments
prompt_result = get_prompt(client, "summarize", Dict("text" => "Hello world"))
println(prompt_result)
```

Prompts let MCP servers provide curated instruction templates that can
be injected into agent conversations.

## Key Types Reference

| Type | Description |
|----|----|
| `AbstractMCPClient` | Base abstract type for all MCP clients |
| `StdioMCPClient` | Subprocess-based MCP client (stdio transport) |
| `HTTPMCPClient` | HTTP-based MCP client |
| `MCPToolInfo` | Tool description (name, description, input schema) |
| `MCPToolResult` | Result from `call_tool` (content, is_error) |
| `MCPResource` | Resource description (uri, name, mime_type) |
| `MCPPrompt` | Prompt template description (name, description, args) |
| `MCPServerCapabilities` | Server capability flags (tools, resources, prompts) |

## Summary

- **MCP** provides a standardised protocol for LLMs to access external
  tools, resources, and prompts without custom integration code.
- **`StdioMCPClient`** connects to subprocess-based servers;
  **`HTTPMCPClient`** connects to HTTP endpoints. Both share the same
  `AbstractMCPClient` interface.
- **`load_mcp_tools`** converts MCP tools into `FunctionTool` objects
  that plug directly into an `Agent`.
- **`with_mcp_client`** ensures safe cleanup via a `do`-block pattern.
- Use **`tool_name_prefix`** to namespace tools when combining multiple
  servers.
- **Resources** and **prompts** provide additional server capabilities
  beyond tool calling.

Next, see [19 — Declarative
Agents](../19_declarative/19_declarative.qmd) to learn how to define
agents and workflows from configuration files.
