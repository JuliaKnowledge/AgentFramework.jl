"""
MCP Tool Integration (Python)

This sample demonstrates how to connect to MCP (Model Context Protocol) servers,
list and call tools, and use MCP tools with an agent.
It mirrors the Julia vignette 18_mcp.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - Node.js / npx available on PATH
  - pip install agent-framework-ollama
"""

import asyncio

from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
from agent_framework.ollama import OllamaChatClient


# ── StdioMCPClient — subprocess transport ────────────────────────────────────

async def stdio_example() -> None:
    """Connect to a filesystem MCP server over stdio."""
    print("=== Stdio MCP Example ===\n")

    # Create and connect to the MCP server.
    mcp = MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        approval_mode="never_require",
        load_prompts=False,
    )
    await mcp.connect()

    # List available tools.
    print("Available tools:")
    for fn in mcp.functions:
        print(f"  {fn.name}: {fn.description}")

    print(f"\nLoaded {len(mcp.functions)} tools from MCP stdio server")
    await mcp.close()


# ── HTTPMCPClient — HTTP transport ───────────────────────────────────────────

async def http_example() -> None:
    """Connect to an MCP server over HTTP (Streamable HTTP)."""
    print("\n=== HTTP MCP Example ===\n")

    mcp = MCPStreamableHTTPTool(
        name="my-api",
        url="http://localhost:8080/mcp",
        approval_mode="never_require",
        load_prompts=False,
    )
    await mcp.connect()

    print("Available tools:")
    for fn in mcp.functions:
        print(f"  {fn.name}: {fn.description}")

    await mcp.close()


# ── Using MCP tools with an Agent ────────────────────────────────────────────

async def agent_with_mcp() -> None:
    """Full example: MCP tools driving an agent."""
    print("\n=== Agent with MCP Tools ===\n")

    # Connect to the MCP server.
    mcp = MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        approval_mode="never_require",
        load_prompts=False,
    )
    await mcp.connect()

    print(f"Loaded {len(mcp.functions)} tools from MCP server")

    # Create a chat client and agent with MCP tools.
    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    agent = client.as_agent(
        name="FileAgent",
        instructions="You can read and list files. Use the available tools to answer questions about the filesystem.",
        tools=mcp.functions,
    )

    # Run a simple agent query using MCP tools.
    result = await agent.run("List the files in the current directory")
    print(f"Agent: {result.text[:500] if hasattr(result, 'text') else result}")

    await mcp.close()

    await mcp.close()


# ── Multiple MCP servers ─────────────────────────────────────────────────────

async def multi_server_example() -> None:
    """Combine tools from multiple MCP servers into one agent."""
    print("\n=== Multiple MCP Servers ===\n")

    fs_mcp = MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        approval_mode="never_require",
        load_prompts=False,
    )
    await fs_mcp.connect()

    # In production, you would also connect an HTTP server:
    # api_mcp = MCPStreamableHTTPTool(name="api", url="http://localhost:8080/mcp")
    # await api_mcp.connect()

    # Combine tools from all servers.
    all_tools = list(fs_mcp.functions)
    # all_tools.extend(api_mcp.functions)

    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    agent = client.as_agent(
        name="MultiServerAgent",
        instructions="You have access to filesystem tools. Use them to answer questions.",
        tools=all_tools,
    )

    result = await agent.run("What files are in the current directory?")
    print(f"Agent: {result}")

    await fs_mcp.close()
    # await api_mcp.close()


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Run the stdio example (requires npx and filesystem MCP server).
    await stdio_example()

    # HTTP example — uncomment if you have an HTTP MCP server running:
    # await http_example()

    # Agent with MCP tools.
    await agent_with_mcp()

    # Multiple servers — uncomment to test:
    # await multi_server_example()


if __name__ == "__main__":
    asyncio.run(main())
