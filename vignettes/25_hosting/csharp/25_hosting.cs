// Hosting and Agent-to-Agent Protocol — C#
//
// This sample demonstrates two related capabilities:
//   1. Hosted Runtime — manage agent lifecycles with session persistence
//      and HTTP serving.
//   2. A2A Protocol — connect to remote agents using the Agent-to-Agent
//      standard (JSON-RPC 2.0 over HTTP).
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using A2A;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.A2A;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// ══════════════════════════════════════════════════════════════════════════ //
//  Part 1 — Hosted Runtime                                                  //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== Part 1 — Hosted Runtime ===\n");

var chatClient = new OllamaApiClient(new Uri(endpoint), modelName);

// 1. Create agents
ChatClientAgent helper = new(chatClient,
    "You are a helpful assistant.",
    "Helper");

ChatClientAgent coder = new(chatClient,
    "You write C# code. Always include XML doc comments.",
    "Coder");

// 2. Wrap in a hosted agent with session persistence
var sessionStore = new InMemoryAgentSessionStore();
var hostedHelper = new AIHostAgent(helper, sessionStore);

// 3. Run the agent
Console.WriteLine("--- Run Helper Agent ---\n");
var session = await hostedHelper.GetOrCreateSessionAsync("conv-1");
List<ChatMessage> messages = [new ChatMessage(ChatRole.User, "What is 2+2?")];
var response = await helper.RunAsync(messages, session);

foreach (var msg in response.Messages)
{
    Console.WriteLine($"  {msg.Role}: {msg.Text}");
}

// 4. Multi-turn with session persistence
Console.WriteLine("\n--- Multi-Turn Session ---\n");

session = await hostedHelper.GetOrCreateSessionAsync("conv-2");
List<ChatMessage> turn1Messages = [new ChatMessage(ChatRole.User, "My name is Alice")];
var r1 = await helper.RunAsync(turn1Messages, session);
await hostedHelper.SaveSessionAsync("conv-2", session);
Console.WriteLine($"  Turn 1: {r1.Text}");

session = await hostedHelper.GetOrCreateSessionAsync("conv-2");
List<ChatMessage> turn2Messages = [new ChatMessage(ChatRole.User, "What's my name?")];
var r2 = await helper.RunAsync(turn2Messages, session);
await hostedHelper.SaveSessionAsync("conv-2", session);
Console.WriteLine($"  Turn 2: {r2.Text}");

// ══════════════════════════════════════════════════════════════════════════ //
//  Part 2 — A2A Protocol                                                    //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("\n=== Part 2 — A2A Protocol ===\n");

// NOTE: A2A demo requires a running A2A server on port 8080.
// Start one first, then uncomment the code below.

/*
// 1. Create an A2A client and agent
var a2aClient = new A2AClient("http://localhost:8080");
var remoteAgent = new A2AAgent(a2aClient, "RemoteHelper");

// 2. Discover capabilities via the agent card
Console.WriteLine("--- A2A Agent Card ---\n");
// The agent card is served at /.well-known/agent.json

// 3. Run the remote agent (non-streaming)
Console.WriteLine("--- A2A Non-Streaming ---\n");
var remoteSession = await remoteAgent.CreateSessionAsync();
List<ChatMessage> a2aMessages = [new ChatMessage(ChatRole.User, "What is 2+2?")];
var a2aResponse = await remoteAgent.RunAsync(a2aMessages, remoteSession);
Console.WriteLine($"  Response: {a2aResponse.Text}");

// 4. Run the remote agent (streaming)
Console.WriteLine("\n--- A2A Streaming ---\n");
List<ChatMessage> streamMessages = [new ChatMessage(ChatRole.User, "Tell me a story about C#")];
await foreach (var update in remoteAgent.RunStreamingAsync(streamMessages, remoteSession))
{
    Console.Write(update.Text);
}
Console.WriteLine();

// 5. Multi-turn with session persistence
Console.WriteLine("\n--- A2A Multi-Turn ---\n");
var a2aSession = await remoteAgent.CreateSessionAsync();

List<ChatMessage> a2aTurn1 = [new ChatMessage(ChatRole.User, "My name is Alice")];
var ar1 = await remoteAgent.RunAsync(a2aTurn1, a2aSession);
Console.WriteLine($"  Turn 1: {ar1.Text}");

List<ChatMessage> a2aTurn2 = [new ChatMessage(ChatRole.User, "What's my name?")];
var ar2 = await remoteAgent.RunAsync(a2aTurn2, a2aSession);
Console.WriteLine($"  Turn 2: {ar2.Text}");
*/

Console.WriteLine("(Skipped — start an A2A server first, then uncomment the A2A section)\n");
