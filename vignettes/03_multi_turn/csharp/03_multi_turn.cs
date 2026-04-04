// Multi-Turn Conversations (C#)
//
// This sample shows how to maintain conversation context across multiple
// agent calls by reusing a session object. It mirrors the Julia vignette
// 03_multi_turn.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// Create an agent backed by the local Ollama instance.
AIAgent agent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        instructions: "You are a friendly assistant. Keep your answers brief.",
        name: "ConversationAgent");

// Create a session to maintain conversation history across turns.
AgentSession session = await agent.CreateSessionAsync();

// Turn 1 — introduce yourself.
Console.WriteLine($"Turn 1: {await agent.RunAsync("My name is Alice and I love hiking.", session)}");
Console.WriteLine();

// Turn 2 — the agent should remember the user's name and hobby.
Console.WriteLine($"Turn 2: {await agent.RunAsync("What do you remember about me?", session)}");
Console.WriteLine();

// Turn 3 — continue building on the conversation.
Console.WriteLine($"Turn 3: {await agent.RunAsync("Suggest a hiking trail for me.", session)}");
Console.WriteLine();

// Demonstrate multiple independent sessions.
AgentSession sessionA = await agent.CreateSessionAsync();
AgentSession sessionB = await agent.CreateSessionAsync();

await agent.RunAsync("My name is Alice.", sessionA);
await agent.RunAsync("My name is Bob.", sessionB);

Console.WriteLine($"Session A: {await agent.RunAsync("What is my name?", sessionA)}");
Console.WriteLine($"Session B: {await agent.RunAsync("What is my name?", sessionB)}");
