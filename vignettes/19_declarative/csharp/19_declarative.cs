// Declarative Agents — C#
//
// This sample demonstrates how to define agents and workflows declaratively
// using YAML configuration, then load and run them with the agent factory.
// It mirrors the Julia vignette 19_declarative.
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

// ══════════════════════════════════════════════════════════════════════════ //
//  A. Agent from YAML string                                                //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== A. Agent from YAML String ===\n");

const string weatherAgentYaml = """
    kind: Prompt
    name: WeatherHelper
    description: A weather assistant that answers concisely
    instructions: You answer weather questions concisely.
    options:
      temperature: 0.7
      max_tokens: 500
    """;

var chatClient = new OllamaApiClient(new Uri(endpoint), modelName);

// Create an agent factory backed by the Ollama chat client.
var agentFactory = new ChatClientPromptAgentFactory(chatClient);
AIAgent weatherAgent = await agentFactory.CreateFromYamlAsync(weatherAgentYaml);

Console.WriteLine(await weatherAgent.RunAsync("Is it sunny in Paris today?"));
Console.WriteLine();

// ══════════════════════════════════════════════════════════════════════════ //
//  B. Agent from file                                                       //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== B. Agent from File ===\n");

var yamlPath = Path.Combine("agents", "weather.yaml");
if (File.Exists(yamlPath))
{
    var yamlContent = await File.ReadAllTextAsync(yamlPath);
    AIAgent fileAgent = await agentFactory.CreateFromYamlAsync(yamlContent);
    Console.WriteLine(await fileAgent.RunAsync("Will it rain in Tokyo tomorrow?"));
}
else
{
    Console.WriteLine($"  Skipped — {yamlPath} not found.");
}
Console.WriteLine();

// ══════════════════════════════════════════════════════════════════════════ //
//  C. Agent with tools                                                      //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== C. Agent with Registered Tools ===\n");

[Description("Get the current weather for a location.")]
static string GetWeather(
    [Description("The location to get the weather for.")] string location)
{
    string[] conditions = ["sunny", "cloudy", "rainy", "stormy"];
    var rng = new Random();
    return $"The weather in {location} is {conditions[rng.Next(conditions.Length)]} at {rng.Next(10, 31)}°C.";
}

// Create a separate factory with the tool registered.
var toolAgentFactory = new ChatClientPromptAgentFactory(chatClient, [AIFunctionFactory.Create(GetWeather)]);

const string toolAgentYaml = """
    kind: Prompt
    name: WeatherAgent
    description: Uses tools to answer weather questions
    instructions: Use the available tools to answer weather questions accurately.
    """;

AIAgent toolAgent = await toolAgentFactory.CreateFromYamlAsync(toolAgentYaml);

Console.WriteLine(await toolAgent.RunAsync("What's the weather in London?"));
Console.WriteLine();

// ══════════════════════════════════════════════════════════════════════════ //
//  D. Workflow from YAML (agents wired as a pipeline)                       //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== D. Workflow via Declarative Agents ===\n");

const string researcherYaml = """
    kind: Prompt
    name: Researcher
    instructions: You find key facts about a topic. Be thorough but concise.
    """;

const string summarizerYaml = """
    kind: Prompt
    name: Summarizer
    instructions: You summarize research findings into a brief paragraph.
    """;

AIAgent researcher = await agentFactory.CreateFromYamlAsync(researcherYaml);
AIAgent summarizer = await agentFactory.CreateFromYamlAsync(summarizerYaml);

// Chain the two agents: researcher produces text, summarizer condenses it.
var researchResponse = await researcher.RunAsync("Recent advances in quantum computing");
string researchResult = researchResponse.Text;
Console.WriteLine($"[Researcher]: {researchResult}\n");

var summaryResponse = await summarizer.RunAsync($"Summarize the following:\n{researchResult}");
string summary = summaryResponse.Text;
Console.WriteLine($"[Summarizer]: {summary}\n");

// ══════════════════════════════════════════════════════════════════════════ //
//  E. Round-trip serialization                                              //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== E. Round-Trip Serialization ===\n");

// Create from YAML, inspect, and recreate.
AIAgent original = await agentFactory.CreateFromYamlAsync(weatherAgentYaml);
Console.WriteLine($"  Original agent: {original.Name}");

// Recreate from the same YAML — demonstrates reproducibility.
AIAgent recreated = await agentFactory.CreateFromYamlAsync(weatherAgentYaml);
Console.WriteLine($"  Recreated agent: {recreated.Name}");

// ══════════════════════════════════════════════════════════════════════════ //
//  F. Environment variable substitution                                     //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("\n=== F. Environment Variable Substitution ===\n");

// In production YAML you would write:
//
//   model:
//     api_key: ${OPENAI_API_KEY}
//     model: gpt-4
//
// The factory resolves ${VAR} references from IConfiguration or
// Environment.GetEnvironmentVariable before parsing.

var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
Console.WriteLine(apiKey is not null
    ? "  OPENAI_API_KEY is set — would be substituted in YAML."
    : "  OPENAI_API_KEY is not set — substitution would raise an error.");
