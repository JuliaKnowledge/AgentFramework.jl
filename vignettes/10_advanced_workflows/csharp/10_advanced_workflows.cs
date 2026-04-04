// Advanced Workflows — Fan-out, fan-in, conditional routing (C#)
//
// This sample demonstrates advanced workflow patterns: parallel fan-out,
// fan-in aggregation, conditional routing, and shared state. It mirrors
// the Julia vignette 10_advanced_workflows.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// ── Fan-Out / Fan-In Demo ───────────────────────────────────────────────────

Console.WriteLine("=== Fan-Out / Fan-In ===");

var dispatcher = new FanOutDispatcher();
var workerA = new TextWorker("WorkerA", "A processed");
var workerB = new TextWorker("WorkerB", "B processed");
var collector = new FanInCollector();

var fanWorkflow = new WorkflowBuilder(dispatcher)
    .AddFanOutEdge(dispatcher, [workerA, workerB])
    .AddFanInBarrierEdge([workerA, workerB], collector)
    .WithOutputFrom(collector)
    .Build();

await using (StreamingRun run = await InProcessExecution.RunStreamingAsync(fanWorkflow, input: "hello"))
{
    await foreach (WorkflowEvent evt in run.WatchStreamAsync())
    {
        if (evt is WorkflowOutputEvent output)
        {
            Console.WriteLine($"  Output: {output.Data}");
        }
    }
}

// ── Text Analysis Pipeline (requires Ollama) ────────────────────────────────

Console.WriteLine("\n=== Text Analysis Pipeline ===");

var chatClient = new OllamaApiClient(new Uri(endpoint), modelName);

ChatClientAgent sentimentAgent = new(
    chatClient,
    name: "Sentiment",
    instructions: "Analyze sentiment. Reply with one word: Positive, Negative, or Neutral."
);
ChatClientAgent keywordAgent = new(
    chatClient,
    name: "Keywords",
    instructions: "Extract 3-5 keywords. Return them comma-separated."
);
ChatClientAgent summaryAgent = new(
    chatClient,
    name: "Summary",
    instructions: "Summarize the text in one sentence."
);

var analysisDispatcher = new AnalysisDispatcher();
var analysisMerger = new AnalysisMerger();

var pipeline = new WorkflowBuilder(analysisDispatcher)
    .AddFanOutEdge(analysisDispatcher, [sentimentAgent, keywordAgent, summaryAgent])
    .AddFanInBarrierEdge([sentimentAgent, keywordAgent, summaryAgent], analysisMerger)
    .WithOutputFrom(analysisMerger)
    .Build();

var inputText = "Julia is a high-level, high-performance programming language for "
    + "technical computing. It combines the ease of Python with the speed of C.";

await using (StreamingRun analysisRun = await InProcessExecution.RunStreamingAsync(pipeline, input: inputText))
{
    await foreach (WorkflowEvent evt in analysisRun.WatchStreamAsync())
    {
        if (evt is WorkflowOutputEvent output)
        {
            Console.WriteLine(output.Data);
        }
    }
}

// ── Executor Definitions ────────────────────────────────────────────────────

/// <summary>Broadcasts input to downstream executors (fan-out).</summary>
internal sealed class FanOutDispatcher() : Executor<string>("FanOutDispatcher")
{
    public override async ValueTask HandleAsync(string message, IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        await context.SendMessageAsync(message, cancellationToken: cancellationToken);
    }
}

/// <summary>Processes text with a prefix label.</summary>
internal sealed class TextWorker(string id, string prefix) :
    Executor<string, string>(id)
{
    public override async ValueTask<string> HandleAsync(string message,
        IWorkflowContext context, CancellationToken cancellationToken = default)
        => $"{prefix}: {message}";
}

/// <summary>Collects fan-in results and yields combined output.</summary>
internal sealed class FanInCollector() :
    Executor<List<string>>("FanInCollector")
{
    public override async ValueTask HandleAsync(List<string> messages, IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        var combined = string.Join(Environment.NewLine, messages);
        await context.YieldOutputAsync(combined, cancellationToken);
    }
}

/// <summary>Dispatches text to analysis agents with a TurnToken.</summary>
internal sealed class AnalysisDispatcher() : Executor<string>("AnalysisDispatcher")
{
    public override async ValueTask HandleAsync(string message, IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        await context.SendMessageAsync(
            new ChatMessage(ChatRole.User, message), cancellationToken: cancellationToken);
        await context.SendMessageAsync(
            new TurnToken(emitEvents: true), cancellationToken: cancellationToken);
    }
}

/// <summary>Merges responses from parallel analysis agents.</summary>
internal sealed class AnalysisMerger() :
    Executor<List<ChatMessage>>("AnalysisMerger")
{
    public override async ValueTask HandleAsync(List<ChatMessage> messages, IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        var report = string.Join(Environment.NewLine,
            messages.Select(m => $"{m.AuthorName}: {m.Text}"));
        await context.YieldOutputAsync($"=== Analysis Results ===\n{report}", cancellationToken);
    }
}
