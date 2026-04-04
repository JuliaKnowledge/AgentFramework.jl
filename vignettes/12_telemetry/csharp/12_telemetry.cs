// Telemetry — Observability and tracing (C#)
//
// This sample demonstrates how to add observability to agents using
// Activity/DiagnosticSource patterns. It mirrors the Julia vignette
// 12_telemetry.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using System.Diagnostics;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// ── 1. Activity/Span Basics ─────────────────────────────────────────────────

Console.WriteLine("=== Span Basics (using System.Diagnostics.Activity) ===");

var source = new ActivitySource("AgentFramework.Sample");

using (var activity = source.StartActivity("my_operation", ActivityKind.Client))
{
    activity?.SetTag("custom.key", "some_value");
    activity?.AddEvent(new ActivityEvent("checkpoint_reached",
        tags: new ActivityTagsCollection { { "step", 1 } }));
    activity?.AddEvent(new ActivityEvent("data_processed",
        tags: new ActivityTagsCollection { { "records", 42 } }));
    await Task.Delay(10);
    activity?.SetStatus(ActivityStatusCode.Ok);
    Console.WriteLine($"  Activity: {activity?.OperationName}, "
        + $"duration: {activity?.Duration.TotalMilliseconds:F0}ms, "
        + $"status: {activity?.Status}");
}

// ── 2. In-Memory Span Collection ────────────────────────────────────────────

Console.WriteLine("\n=== In-Memory Span Collection ===");

var collectedSpans = new List<TelemetryRecord>();

// Record spans manually (mirrors InMemoryTelemetryBackend)
var testSpan = new TelemetryRecord("test.operation", "client", DateTimeOffset.UtcNow);
testSpan.Attributes["gen_ai.request.model"] = "qwen3:8b";
testSpan.Finish("ok");
collectedSpans.Add(testSpan);

Console.WriteLine($"  Recorded spans: {collectedSpans.Count}");
Console.WriteLine($"  First span: {collectedSpans[0].Name}");
collectedSpans.Clear();
Console.WriteLine($"  After clear: {collectedSpans.Count}");

// ── 3. Instrumented Workflow with OpenTelemetry ─────────────────────────────

Console.WriteLine("\n=== Instrumented Workflow ===");

var upperExecutor = new UppercaseExecutor();
var reverseExecutor = new ReverseTextExecutor();

var workflow = new WorkflowBuilder(upperExecutor)
    .AddEdge(upperExecutor, reverseExecutor)
    .WithOpenTelemetry(
        configure: cfg => cfg.EnableSensitiveData = true,
        activitySource: source)
    .Build();

await using (Run run = await InProcessExecution.RunAsync(workflow, "Hello, World!"))
{
    foreach (WorkflowEvent evt in run.NewEvents)
    {
        if (evt is ExecutorCompletedEvent executorComplete)
        {
            Console.WriteLine($"  {executorComplete.ExecutorId}: {executorComplete.Data}");
        }
    }
}

// ── 4. Instrumented Agent Call ──────────────────────────────────────────────

Console.WriteLine("\n=== Instrumented Agent ===");

AIAgent agent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        name: "TracedAgent",
        instructions: "You are a helpful assistant. Keep answers brief.");

var agentSpan = new TelemetryRecord("agent.run", "internal", DateTimeOffset.UtcNow);
agentSpan.Attributes["gen_ai.agent.name"] = "TracedAgent";

var chatSpan = new TelemetryRecord("chat.completion", "client", DateTimeOffset.UtcNow);
chatSpan.Attributes["gen_ai.request.model"] = modelName;

var agentResponse = await agent.RunAsync("What is 2 + 2?");
Console.WriteLine($"  Answer: {agentResponse.Text}");

chatSpan.Finish("ok");
agentSpan.Attributes["gen_ai.response.model"] = modelName;
agentSpan.Finish("ok");

collectedSpans.AddRange([agentSpan, chatSpan]);

Console.WriteLine($"\n  Total spans: {collectedSpans.Count}");
foreach (var s in collectedSpans)
{
    var model = s.Attributes.GetValueOrDefault("gen_ai.request.model", "—");
    Console.WriteLine($"    [{s.Kind}] {s.Name} — {s.DurationMs}ms, "
        + $"model={model}, status={s.Status}");
}

// ── Supporting Types ────────────────────────────────────────────────────────

/// <summary>Simple telemetry record mirroring TelemetrySpan.</summary>
class TelemetryRecord
{
    public string Name { get; }
    public string Kind { get; }
    public DateTimeOffset StartTime { get; }
    public DateTimeOffset? EndTime { get; private set; }
    public string Status { get; private set; } = "unset";
    public Dictionary<string, object> Attributes { get; } = new();

    public long? DurationMs => EndTime.HasValue
        ? (long)(EndTime.Value - StartTime).TotalMilliseconds : null;

    public TelemetryRecord(string name, string kind, DateTimeOffset startTime)
    {
        Name = name;
        Kind = kind;
        StartTime = startTime;
    }

    public void Finish(string status = "ok")
    {
        EndTime = DateTimeOffset.UtcNow;
        Status = status;
    }
}

/// <summary>Converts input text to uppercase.</summary>
internal sealed class UppercaseExecutor() : Executor<string, string>("UppercaseExecutor")
{
    public override async ValueTask<string> HandleAsync(string message,
        IWorkflowContext context, CancellationToken cancellationToken = default)
        => message.ToUpperInvariant();
}

/// <summary>Reverses input text.</summary>
internal sealed class ReverseTextExecutor() : Executor<string, string>("ReverseTextExecutor")
{
    public override async ValueTask<string> HandleAsync(string message,
        IWorkflowContext context, CancellationToken cancellationToken = default)
        => new(message.Reverse().ToArray());
}
