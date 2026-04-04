// Hello Agent — Simplest possible agent (C#)
//
// This sample creates a minimal agent using OllamaChatClient,
// sends a single prompt, and prints the response. It mirrors the Julia
// vignette 01_hello_agent.
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
        name: "HelloAgent");

// Non-streaming: get the complete response at once.
Console.WriteLine(await agent.RunAsync("What is the capital of France?"));

// Streaming: receive tokens as they are generated.
Console.Write("Agent (streaming): ");
await foreach (var update in agent.RunStreamingAsync("Tell me a one-sentence fun fact."))
{
    Console.Write(update);
}
Console.WriteLine();
