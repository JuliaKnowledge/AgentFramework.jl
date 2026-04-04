// Adding Tools to Agents (C#)
//
// This sample demonstrates how to define function tools and attach them
// to an agent so the model can call them automatically.
// It mirrors the Julia vignette 02_tools.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// Define tool functions with [Description] attributes.
[Description("Get the current weather for a location.")]
static string GetWeather(
    [Description("The location to get the weather for.")] string location)
{
    string[] conditions = ["sunny", "cloudy", "rainy", "stormy"];
    var rng = new Random();
    return $"The weather in {location} is {conditions[rng.Next(conditions.Length)]} with a high of {rng.Next(10, 31)}°C.";
}

[Description("Get the approximate population of a country in millions.")]
static string GetPopulation(
    [Description("The country to look up.")] string country)
{
    var populations = new Dictionary<string, int>
    {
        ["France"] = 68,
        ["Germany"] = 84,
        ["Japan"] = 125,
        ["Brazil"] = 214,
        ["Australia"] = 26,
    };
    return populations.TryGetValue(country, out var pop)
        ? $"{country} has approximately {pop} million people."
        : $"Population data not available for {country}.";
}

[Description("Evaluate a mathematical expression and return the result.")]
static string Calculate(
    [Description("A mathematical expression to evaluate.")] string expression)
{
    // Simple evaluation via DataTable for demo purposes.
    var result = new System.Data.DataTable().Compute(expression, null);
    return result?.ToString() ?? "Error";
}

// Create the agent with tools.
AIAgent agent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        instructions: "You are a helpful assistant. Use the available tools to answer questions accurately. Be concise.",
        tools:
        [
            AIFunctionFactory.Create(GetWeather),
            AIFunctionFactory.Create(GetPopulation),
            AIFunctionFactory.Create(Calculate),
        ]);

// The agent will automatically call the GetWeather tool.
Console.WriteLine(await agent.RunAsync("What's the weather like in Tokyo?"));
Console.WriteLine();

// Ask about population — triggers the GetPopulation tool.
Console.WriteLine(await agent.RunAsync("What is the population of Brazil?"));
Console.WriteLine();

// A question that may require multiple tool calls.
Console.WriteLine(await agent.RunAsync("What is the combined population of France and Germany?"));
