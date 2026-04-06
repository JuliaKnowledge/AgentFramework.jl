// Skills System (C#)
//
// This sample demonstrates the Agent Skills system: reusable, discoverable
// instruction sets that extend agent capabilities via progressive disclosure.
// It mirrors the Julia vignette 23_skills.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore
//   - NuGet: Microsoft.Agents.AI, OllamaSharp

using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// ── Code-Defined Skills ─────────────────────────────────────────────────────

// Define a skill entirely in code using AgentInlineSkill.
// No SKILL.md files needed — skills, resources, and scripts are programmatic.

Console.WriteLine("=== Code-Defined Skills ===\n");

var unitConverterSkill = new AgentInlineSkill(
    name: "unit-converter",
    description: "Convert between common units using a multiplication factor",
    instructions: """
        Use this skill when the user asks to convert between units.

        1. Review the conversion-tables resource to find the factor.
        2. Check the conversion-policy resource for rounding rules.
        3. Use the convert script, passing the value and factor.
        """)
    // Static resource: inline content.
    .AddResource(
        "conversion-tables",
        """
        # Conversion Tables

        Formula: **result = value × factor**

        | From        | To          | Factor   |
        |-------------|-------------|----------|
        | miles       | kilometers  | 1.60934  |
        | kilometers  | miles       | 0.621371 |
        | pounds      | kilograms   | 0.453592 |
        | kilograms   | pounds      | 2.20462  |
        """)
    // Dynamic resource: computed at runtime.
    .AddResource("conversion-policy", () =>
    {
        const int Precision = 4;
        return $"""
            # Conversion Policy

            **Decimal places:** {Precision}
            **Format:** Always show both the original and converted values with units
            **Generated at:** {DateTime.UtcNow:O}
            """;
    })
    // Code script: executable delegate the agent can invoke.
    .AddScript("convert", (double value, double factor) =>
    {
        double result = Math.Round(value * factor, 4);
        return JsonSerializer.Serialize(new { value, factor, result });
    });

// Create a SkillsProvider with the code-defined skill.
var codeSkillsProvider = new AgentSkillsProvider(unitConverterSkill);

AIAgent codeAgent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(new ChatClientAgentOptions
    {
        Name = "UnitConverterAgent",
        ChatOptions = new()
        {
            Instructions = "You are a helpful assistant that can convert units.",
        },
        AIContextProviders = [codeSkillsProvider],
    });

Console.WriteLine("Converting units with code-defined skills");
Console.WriteLine(new string('-', 60));

AgentResponse codeResponse = await codeAgent.RunAsync(
    "How many kilometers is a marathon (26.2 miles)?");
Console.WriteLine($"Agent: {codeResponse.Text}\n");

// ── File-Based Skills ───────────────────────────────────────────────────────

// Discover skills from SKILL.md files in a directory tree.
// Each skill lives in its own folder:
//
//   skills/
//   ├── data_analysis/
//   │   ├── SKILL.md
//   │   └── schema.json
//   └── code_review/
//       ├── SKILL.md
//       └── style_guide.md

Console.WriteLine("=== File-Based Skills ===\n");

// Uncomment when you have a skills/ directory:
// var fileSkillsProvider = new AgentSkillsProvider(
//     Path.Combine(AppContext.BaseDirectory, "skills"));
//
// AIAgent fileAgent = new OllamaApiClient(new Uri(endpoint), modelName)
//     .AsAIAgent(new ChatClientAgentOptions
//     {
//         Name = "SkillfulAgent",
//         ChatOptions = new()
//         {
//             Instructions = "You have access to various skills. Use them when appropriate.",
//         },
//         AIContextProviders = [fileSkillsProvider],
//     });
//
// AgentResponse fileResponse = await fileAgent.RunAsync(
//     "Analyze this data: name,age\nAlice,30\nBob,25");
// Console.WriteLine($"Agent: {fileResponse.Text}\n");

Console.WriteLine("(Skipped — no skills/ directory present)\n");

// ── Mixed Skills (Code + File) ──────────────────────────────────────────────

// Use AgentSkillsProviderBuilder to combine multiple skill sources.

Console.WriteLine("=== Mixed Skills ===\n");

// Uncomment when you have a skills/ directory:
// var mixedProvider = new AgentSkillsProviderBuilder()
//     .UseFileSkill(Path.Combine(AppContext.BaseDirectory, "skills"))
//     .UseSkill(unitConverterSkill)
//     .Build();
//
// AIAgent mixedAgent = new OllamaApiClient(new Uri(endpoint), modelName)
//     .AsAIAgent(new ChatClientAgentOptions
//     {
//         Name = "MultiSkillAgent",
//         ChatOptions = new()
//         {
//             Instructions = "You are a helpful assistant with access to various skills.",
//         },
//         AIContextProviders = [mixedProvider],
//     });
//
// AgentResponse mixedResponse = await mixedAgent.RunAsync(
//     "How many kilometers is 26.2 miles? And how many pounds is 75 kilograms?");
// Console.WriteLine($"Agent: {mixedResponse.Text}\n");

Console.WriteLine("(Skipped — no skills/ directory present)\n");

Console.WriteLine("Done.");
