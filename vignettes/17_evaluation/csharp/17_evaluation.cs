// Evaluation Framework — Testing and validating agent behaviour (C#)
//
// The agent-framework .NET SDK does not include a built-in evaluation API,
// so this sample demonstrates how to build a lightweight evaluation harness
// using plain C# to test and validate agent behaviour.
// It mirrors the Julia vignette 17_evaluation.
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

// ── Weather tool ──────────────────────────────────────────────────────────

[Description("Get the current weather for a city")]
static string GetWeather(string city)
{
    return city.ToLowerInvariant() switch
    {
        "paris"  => "Paris: Sunny, 22°C",
        "london" => "London: Rainy, 14°C",
        _        => $"{city}: Cloudy, 18°C",
    };
}

// ── Create the agent ──────────────────────────────────────────────────────

AIAgent weatherAgent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        instructions: "You are a weather assistant. Use the GetWeather tool to answer weather questions. Always include the temperature in your response.",
        name: "WeatherBot",
        tools: [AIFunctionFactory.Create(GetWeather)]);

// ══════════════════════════════════════════════════════════════════════════ //
//  Evaluation types                                                         //
// ══════════════════════════════════════════════════════════════════════════ //

// ── Built-in checks ───────────────────────────────────────────────────────

Console.WriteLine("=== Built-in Checks ===");

// Define reusable check functions that inspect agent responses.

static EvalCheck KeywordCheck(params string[] keywords) => new(
    "keyword_check",
    (text, _, _) => keywords.All(kw => text.Contains(kw, StringComparison.OrdinalIgnoreCase)));

static EvalCheck ToolCalledCheck(string toolName) => new(
    $"tool_called({toolName})",
    (_, raw, _) => raw.Messages
        .SelectMany(m => m.Contents.OfType<FunctionCallContent>())
        .Any(tc => tc.Name == toolName));

static EvalCheck ToolCallsPresent() => new(
    "tool_calls_present",
    (_, raw, _) => raw.Messages
        .SelectMany(m => m.Contents.OfType<FunctionCallContent>())
        .Any());

static EvalCheck ResponseLengthCheck(int minLength) => new(
    "response_length",
    (text, _, _) => text.Length >= minLength);

static EvalCheck SemanticMatchCheck() => new(
    "semantic_match",
    (text, _, expected) =>
        expected is not null && text.Contains(expected, StringComparison.OrdinalIgnoreCase));

var keywordCheck    = KeywordCheck("weather", "temperature");
var toolCheck       = ToolCalledCheck("GetWeather");
var anyToolCheck    = ToolCallsPresent();
var lengthCheck     = ResponseLengthCheck(50);
var similarityCheck = SemanticMatchCheck();

Console.WriteLine("  Defined: keyword_check, tool_called, tool_calls_present, response_length, semantic_match\n");

// ── Evaluate a single agent ───────────────────────────────────────────────

Console.WriteLine("=== evaluate_agent ===");

var evalResults = await EvaluateAgentAsync(
    weatherAgent,
    ["What's the weather in Paris?", "Will it rain in London?"],
    [keywordCheck, lengthCheck, toolCheck, anyToolCheck]);

PrintResults(evalResults, "weather_basic");
Console.WriteLine();

// ── Evaluate with expected output ─────────────────────────────────────────

Console.WriteLine("=== evaluate_agent with expected output ===");

var expectedResults = await EvaluateAgentAsync(
    weatherAgent,
    ["What's the weather in Paris?"],
    [similarityCheck],
    expectedOutputs: ["Sunny"]);

PrintResults(expectedResults, "expected_output");
Console.WriteLine();

// ── Evaluate with expected tool calls ─────────────────────────────────────

Console.WriteLine("=== evaluate_agent with expected tool calls ===");

EvalCheck toolArgsCheck = new(
    "tool_call_args_match",
    (_, raw, _) => raw.Messages
        .SelectMany(m => m.Contents.OfType<FunctionCallContent>())
        .Any(tc => tc.Name == "GetWeather"
            && tc.Arguments is not null
            && tc.Arguments.TryGetValue("city", out var city)
            && "Paris".Equals(city?.ToString(), StringComparison.OrdinalIgnoreCase)));

var toolCallResults = await EvaluateAgentAsync(
    weatherAgent,
    ["What's the weather in Paris?"],
    [toolArgsCheck, anyToolCheck]);

PrintResults(toolCallResults, "tool_calls");
Console.WriteLine();

// ── EvalItem with pre-built conversation ──────────────────────────────────

Console.WriteLine("=== EvalItem ===");

var item = new EvalItem(
    Query: "What's the weather?",
    Response: "It's sunny and 22°C in Paris.",
    ExpectedOutput: "sunny");

Console.WriteLine($"Query   : {item.Query}");
Console.WriteLine($"Response: {item.Response}");

var itemChecks = RunChecks(item.Response, null, item.ExpectedOutput, [similarityCheck, lengthCheck]);
foreach (var c in itemChecks)
    Console.WriteLine($"  {c.Name}: {(c.Passed ? "PASS" : "FAIL")}");
Console.WriteLine();

// ── raise_for_status ──────────────────────────────────────────────────────

Console.WriteLine("=== RaiseForStatus ===");

var statusResults = await EvaluateAgentAsync(
    weatherAgent,
    ["What's the weather in Paris?"],
    [keywordCheck, lengthCheck, toolCheck, anyToolCheck]);

try
{
    RaiseForStatus(statusResults, "Quality gate failed");
    Console.WriteLine("All checks passed!");
}
catch (Exception ex)
{
    Console.WriteLine($"Evaluation failed: {ex.Message}");
}

// ══════════════════════════════════════════════════════════════════════════ //
//  Helper types and functions                                               //
// ══════════════════════════════════════════════════════════════════════════ //

static List<CheckResult> RunChecks(
    string text, AgentResponse? raw, string? expected, IEnumerable<EvalCheck> checks) =>
    checks.Select(c => new CheckResult(c.Name, c.Check(text, raw!, expected))).ToList();

static async Task<List<EvalItemResult>> EvaluateAgentAsync(
    AIAgent agent,
    string[] queries,
    EvalCheck[] checks,
    string[]? expectedOutputs = null)
{
    var results = new List<EvalItemResult>();
    for (int i = 0; i < queries.Length; i++)
    {
        var response = await agent.RunAsync(queries[i]);
        var text = response.Text;
        var expected = expectedOutputs is not null && i < expectedOutputs.Length
            ? expectedOutputs[i] : null;
        var checkResults = RunChecks(text, response, expected, checks);
        results.Add(new EvalItemResult(i, queries[i], text, checkResults));
    }
    return results;
}

static void PrintResults(List<EvalItemResult> results, string evalName)
{
    int totalPassed = results.Sum(r => r.Checks.Count(c => c.Passed));
    int totalChecks = results.Sum(r => r.Checks.Count);
    bool allOk = results.All(r => r.AllPassed);

    Console.WriteLine($"Eval     : {evalName}");
    Console.WriteLine($"Passed   : {totalPassed} / {totalChecks}");
    Console.WriteLine($"All OK?  : {allOk}");

    foreach (var r in results)
    {
        Console.WriteLine($"  Item {r.ItemId}: {(r.AllPassed ? "PASS" : "FAIL")}");
        foreach (var c in r.Checks)
            Console.WriteLine($"    {c.Name}: {(c.Passed ? "PASS" : "FAIL")}");
    }
}

static void RaiseForStatus(List<EvalItemResult> results, string message)
{
    int totalPassed = results.Sum(r => r.Checks.Count(c => c.Passed));
    int totalChecks = results.Sum(r => r.Checks.Count);
    if (!results.All(r => r.AllPassed))
        throw new Exception($"{message}: {totalPassed}/{totalChecks} checks passed");
}

// ── Records ───────────────────────────────────────────────────────────────

record EvalCheck(string Name, Func<string, AgentResponse, string?, bool> Check);
record CheckResult(string Name, bool Passed);
record EvalItem(string Query, string Response, string? ExpectedOutput = null);

record EvalItemResult(int ItemId, string Query, string Response, List<CheckResult> Checks)
{
    public bool AllPassed => Checks.All(c => c.Passed);
}
