// Structured Output — Getting typed responses from the LLM (C#)
//
// This sample demonstrates how to request structured JSON output from an
// agent, so you get validated C# objects instead of free-form text.
// It mirrors the Julia vignette 05_structured_output.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
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
        instructions: "You are a movie critic. Return structured reviews when asked.",
        name: "MovieCritic");

// ── Structured output with ResponseFormat ────────────────────────────────
Console.WriteLine("=== Movie Review (structured) ===");

// Use RunAsync<T> for type-safe structured output.
AgentResponse<MovieReview> response = await agent.RunAsync<MovieReview>(
    "Review the movie 'Inception' (2010).");

MovieReview review = response.Result;
Console.WriteLine($"Title: {review.Title}");
Console.WriteLine($"Rating: {review.Rating}");
Console.WriteLine($"Summary: {review.Summary}");
Console.WriteLine($"Recommended: {review.Recommended}");
Console.WriteLine();

// ── Manual approach: deserialize from response text ──────────────────────
Console.WriteLine("=== Movie Review (manual parse) ===");

// Create agent with explicit response format in options.
AIAgent agentWithFormat = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(new ChatClientAgentOptions()
    {
        Name = "MovieCritic",
        ChatOptions = new()
        {
            Instructions = "You are a movie critic. Return structured reviews when asked.",
            ResponseFormat = ChatResponseFormat.ForJsonSchema<MovieReview>()
        }
    });

AgentResponse rawResponse = await agentWithFormat.RunAsync(
    "Review 'The Shawshank Redemption'.");

Console.WriteLine($"Raw: {rawResponse.Text}");
MovieReview parsed = JsonSerializer.Deserialize<MovieReview>(rawResponse.Text)!;
Console.WriteLine($"Title: {parsed.Title}");
Console.WriteLine($"Rating: {parsed.Rating}");
Console.WriteLine();

// ── Nested types ─────────────────────────────────────────────────────────
Console.WriteLine("=== Book Review (nested types) ===");

AgentResponse<BookReview> bookResponse = await agent.RunAsync<BookReview>(
    "Review '1984' by George Orwell.");

BookReview book = bookResponse.Result;
Console.WriteLine($"Title: {book.Title}");
Console.WriteLine($"Author: {book.Author.Name} ({book.Author.Nationality})");
Console.WriteLine($"Rating: {book.Rating}");
Console.WriteLine($"Themes: {string.Join(", ", book.Themes)}");

// ── Response type definitions ────────────────────────────────────────────

[Description("A structured movie review")]
public sealed class MovieReview
{
    [JsonPropertyName("title")]
    public string Title { get; set; } = "";

    [JsonPropertyName("rating")]
    public int Rating { get; set; }

    [JsonPropertyName("summary")]
    public string Summary { get; set; } = "";

    [JsonPropertyName("recommended")]
    public bool Recommended { get; set; }
}

[Description("Information about an author")]
public sealed class Author
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("nationality")]
    public string Nationality { get; set; } = "";
}

[Description("A structured book review")]
public sealed class BookReview
{
    [JsonPropertyName("title")]
    public string Title { get; set; } = "";

    [JsonPropertyName("author")]
    public Author Author { get; set; } = new();

    [JsonPropertyName("rating")]
    public int Rating { get; set; }

    [JsonPropertyName("themes")]
    public List<string> Themes { get; set; } = [];
}
