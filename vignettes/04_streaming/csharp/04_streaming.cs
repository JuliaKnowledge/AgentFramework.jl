// Streaming Responses — Real-time token delivery (C#)
//
// This sample demonstrates how to stream responses from an agent,
// receiving tokens as they are generated rather than waiting for the
// full response. It mirrors the Julia vignette 04_streaming.
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
        instructions: "You are a helpful assistant. Provide detailed answers.",
        name: "StreamingAgent");

// ── Non-streaming: wait for the full response ────────────────────────────
Console.WriteLine("=== Non-streaming ===");
Console.WriteLine(await agent.RunAsync("Explain the water cycle in one paragraph."));
Console.WriteLine();

// ── Streaming: receive tokens as they arrive ─────────────────────────────
Console.WriteLine("=== Streaming ===");
Console.Write("Agent: ");
await foreach (var update in agent.RunStreamingAsync(
    "Explain the water cycle in one paragraph."))
{
    Console.Write(update);
}
Console.WriteLine();
Console.WriteLine();

// ── Streaming with token inspection ──────────────────────────────────────
Console.WriteLine("=== Token inspection ===");
var tokenCount = 0;
await foreach (var update in agent.RunStreamingAsync("What is 2 + 2?"))
{
    var text = update.ToString();
    if (!string.IsNullOrEmpty(text))
    {
        tokenCount++;
        Console.WriteLine($"  Fragment {tokenCount}: \"{text}\"");
    }
}
Console.WriteLine($"Total fragments: {tokenCount}");
