// Middleware — Intercepting and modifying agent behaviour (C#)
//
// This sample demonstrates how to use middleware to add logging, timing,
// and guardrail capabilities to agents. It mirrors the Julia vignette
// 06_middleware.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using System.ComponentModel;
using System.Diagnostics;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// ── Define a tool ────────────────────────────────────────────────────────

[Description("Get the population of a city.")]
static string GetPopulation(
    [Description("The city name.")] string city) =>
    city switch
    {
        "Paris" => "2.1 million",
        "London" => "8.8 million",
        "Tokyo" => "14 million",
        _ => "Unknown"
    };

// ── Create base agent with tool ──────────────────────────────────────────

AIAgent baseAgent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        instructions: "You are a helpful city information assistant. Use the GetPopulation tool when asked about population.",
        name: "CityBot",
        tools: [AIFunctionFactory.Create(GetPopulation)]);

// ── Function invocation middleware: tool logging ─────────────────────────

async ValueTask<object?> ToolLoggingMiddleware(
    AIAgent agent,
    FunctionInvocationContext context,
    Func<FunctionInvocationContext, CancellationToken, ValueTask<object?>> next,
    CancellationToken cancellationToken)
{
    Console.WriteLine($"    [FuncMW] Calling tool: {context.Function.Name}");

    var result = await next(context, cancellationToken);

    Console.WriteLine($"    [FuncMW] Tool {context.Function.Name} returned: {result}");
    return result;
}

// ── Agent middleware: logging ────────────────────────────────────────────

async Task<AgentResponse> LoggingAgentMiddleware(
    IEnumerable<ChatMessage> messages,
    AgentSession? session,
    AgentRunOptions? options,
    AIAgent innerAgent,
    CancellationToken cancellationToken)
{
    var msgCount = messages.Count();
    Console.WriteLine($"[AgentMW] Starting {innerAgent.Name} with {msgCount} message(s)");

    var response = await innerAgent.RunAsync(messages, session, options, cancellationToken);

    Console.WriteLine($"[AgentMW] {innerAgent.Name} completed");
    return response;
}

// ── Chat client middleware: timing ───────────────────────────────────────

async Task<ChatResponse> TimingChatMiddleware(
    IEnumerable<ChatMessage> messages,
    ChatOptions? options,
    IChatClient innerClient,
    CancellationToken cancellationToken)
{
    var msgCount = messages.Count();
    Console.WriteLine($"  [ChatMW] Sending {msgCount} messages to LLM...");

    var sw = Stopwatch.StartNew();
    var response = await innerClient.GetResponseAsync(messages, options, cancellationToken);
    sw.Stop();

    Console.WriteLine($"  [ChatMW] LLM responded in {sw.Elapsed.TotalSeconds:F2}s");
    return response;
}

// ── Security guardrail middleware ────────────────────────────────────────

async Task<AgentResponse> SecurityMiddleware(
    IEnumerable<ChatMessage> messages,
    AgentSession? session,
    AgentRunOptions? options,
    AIAgent innerAgent,
    CancellationToken cancellationToken)
{
    foreach (var msg in messages)
    {
        var text = msg.Text?.ToLowerInvariant() ?? "";
        if (text.Contains("password") || text.Contains("secret"))
        {
            Console.WriteLine("[Security] Blocked request: sensitive content detected");
            return new AgentResponse([new ChatMessage(ChatRole.Assistant,
                "I cannot process requests containing sensitive information.")]);
        }
    }

    return await innerAgent.RunAsync(messages, session, options, cancellationToken);
}

// ── Build agent with full middleware pipeline ────────────────────────────

Console.WriteLine("=== Full middleware pipeline ===");

var middlewareAgent = baseAgent
    .AsBuilder()
    .Use(ToolLoggingMiddleware)
    .Use(LoggingAgentMiddleware, null)
    .Build();

Console.WriteLine(await middlewareAgent.RunAsync("What is the population of Paris?"));
Console.WriteLine();

// ── Security guardrail demo ──────────────────────────────────────────────

Console.WriteLine("=== Security guardrail ===");

AIAgent secureAgent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        instructions: "You are a helpful assistant.",
        name: "SecureBot");

var guardedAgent = secureAgent
    .AsBuilder()
    .Use(SecurityMiddleware, null)
    .Build();

// This request is blocked.
Console.WriteLine($"Blocked: {await guardedAgent.RunAsync("What is my password?")}");

// This request goes through.
Console.WriteLine($"Allowed: {await guardedAgent.RunAsync("What is the capital of France?")}");
