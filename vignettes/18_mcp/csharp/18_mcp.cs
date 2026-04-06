// MCP Tool Integration (C#)
//
// This sample demonstrates how to connect to MCP (Model Context Protocol)
// servers, list and call tools, and use MCP tools with an agent.
// It mirrors the Julia vignette 18_mcp.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - Node.js / npx available on PATH
//   - dotnet restore
//   - NuGet: ModelContextProtocol, Microsoft.Agents.AI, OllamaSharp

using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// ── StdioMCPClient — subprocess transport ───────────────────────────────────

Console.WriteLine("=== Stdio MCP Example ===\n");

// Create a stdio transport to launch the filesystem MCP server.
var stdioTransport = new StdioClientTransport(new StdioClientTransportOptions
{
    Command = "npx",
    Arguments = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    Name = "filesystem",
});

// Connect to the MCP server.
await using var stdioClient = await McpClient.CreateAsync(stdioTransport);

// List available tools.
var tools = await stdioClient.ListToolsAsync();
Console.WriteLine("Available tools:");
foreach (var tool in tools)
{
    Console.WriteLine($"  {tool.Name}: {tool.Description}");
}

// Call a tool directly.
var readResult = await stdioClient.CallToolAsync(
    "read_file",
    new Dictionary<string, object?> { ["path"] = "/tmp/test.txt" });
foreach (var content in readResult.Content)
{
    if (content is TextContentBlock textBlock)
        Console.WriteLine($"  Content: {textBlock.Text}");
    else
        Console.WriteLine($"  Content: {content}");
}

// ── Using MCP tools with an Agent ───────────────────────────────────────────

Console.WriteLine("\n=== Agent with MCP Tools ===\n");

// Get MCP tools as AIFunctions for use with an agent.
var mcpTools = await stdioClient.ListToolsAsync();
var aiFunctions = new List<AITool>();
foreach (var mcpTool in mcpTools)
{
    aiFunctions.Add(mcpTool);
}

Console.WriteLine($"Loaded {aiFunctions.Count} tools from MCP server");

// Create the agent with MCP tools.
AIAgent agent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        instructions: "You can read and list files. Use the available tools to answer questions about the filesystem.",
        tools: aiFunctions);

Console.WriteLine(await agent.RunAsync("List the files in /tmp"));

// ── HTTPMCPClient — HTTP transport ──────────────────────────────────────────

Console.WriteLine("\n=== HTTP MCP Example ===\n");

// For remote MCP servers, use HttpClientTransport.
// Uncomment and configure if you have an HTTP MCP server running:
//
// var httpTransport = new HttpClientTransport(new()
// {
//     Endpoint = new Uri("http://localhost:8080/mcp"),
//     Name = "my-api",
// });
// await using var httpClient = await McpClient.CreateAsync(httpTransport);
//
// var httpTools = await httpClient.ListToolsAsync();
// Console.WriteLine("HTTP server tools:");
// foreach (var tool in httpTools)
// {
//     Console.WriteLine($"  {tool.Name}: {tool.Description}");
// }

Console.WriteLine("(Skipped — no HTTP MCP server running)");

// ── Multiple MCP Servers ────────────────────────────────────────────────────

Console.WriteLine("\n=== Multiple MCP Servers ===\n");

// You can combine tools from several MCP servers into one agent.
// Each server gets its own transport and client.

var fsTransport = new StdioClientTransport(new StdioClientTransportOptions
{
    Command = "npx",
    Arguments = ["-y", "@modelcontextprotocol/server-filesystem", "."],
    Name = "filesystem",
});
await using var fsClient = await McpClient.CreateAsync(fsTransport);

// In production, connect additional servers:
// await using var apiClient = await McpClient.ConnectAsync(apiTransport);

var allTools = new List<AITool>();
foreach (var t in await fsClient.ListToolsAsync())
{
    allTools.Add(t);
}
// foreach (var t in await apiClient.ListToolsAsync())
// {
//     allTools.Add(t.AsAIFunction());
// }

AIAgent multiAgent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        instructions: "You have access to filesystem tools. Use them to answer questions.",
        tools: allTools);

Console.WriteLine(await multiAgent.RunAsync("What files are in the current directory?"));

// ── Resources and Prompts ───────────────────────────────────────────────────

Console.WriteLine("\n=== Resources and Prompts ===\n");

// List resources (if supported by the server).
try
{
    var resources = await stdioClient.ListResourcesAsync();
    Console.WriteLine("Resources:");
    foreach (var resource in resources)
    {
        Console.WriteLine($"  {resource.Name} ({resource.MimeType}): {resource.Uri}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Resources not supported: {ex.Message}");
}

// List prompts (if supported by the server).
try
{
    var prompts = await stdioClient.ListPromptsAsync();
    Console.WriteLine("Prompts:");
    foreach (var prompt in prompts)
    {
        Console.WriteLine($"  {prompt.Name}: {prompt.Description}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Prompts not supported: {ex.Message}");
}

Console.WriteLine("\nDone.");
