// DAG-Based Workflows — C#
//
// This sample demonstrates the workflow engine:
//   1. Define executors (function-based and class-based).
//   2. Build a pipeline with WorkflowBuilder.
//   3. Run the workflow and inspect outputs.
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

// ══════════════════════════════════════════════════════════════════════════ //
//  1. Simple Text Pipeline: uppercase → reverse → exclaim                  //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== Text Pipeline: uppercase → reverse → exclaim ===\n");

// Function-based executor: bind a lambda as an executor.
Func<string, string> uppercaseFunc = s => s.ToUpperInvariant();
var uppercase = uppercaseFunc.BindAsExecutor("uppercase");

// Class-based executors.
ReverseTextExecutor reverse = new();
ExclaimExecutor exclaim = new();

// Build the workflow DAG.
WorkflowBuilder builder = new(uppercase);
builder
    .AddEdge(uppercase, reverse)
    .AddEdge(reverse, exclaim)
    .WithOutputFrom(exclaim);
var workflow = builder.Build();

// Run the workflow.
await using Run run = await InProcessExecution.RunAsync(workflow, "hello world");
foreach (WorkflowEvent evt in run.NewEvents)
{
    if (evt is ExecutorCompletedEvent completed)
    {
        Console.WriteLine($"  [{completed.ExecutorId}]: {completed.Data}");
    }
}

// ══════════════════════════════════════════════════════════════════════════ //
//  2. Agent-Based Workflow: writer → reviewer                              //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("\n=== Agent Workflow: writer → reviewer ===\n");

var chatClient = new OllamaApiClient(new Uri(endpoint), modelName);

ChatClientAgent writer = new(chatClient,
    "You write creative one-sentence slogans.",
    "Writer");

ChatClientAgent reviewer = new(chatClient,
    "You review slogans and provide brief, actionable feedback.",
    "Reviewer");

var agentWorkflow = new WorkflowBuilder(writer)
    .AddEdge(writer, reviewer)
    .Build();

// Run with streaming to see agent responses as they arrive.
List<ChatMessage> agentMessages = [new ChatMessage(ChatRole.User, "Create a slogan for an electric bicycle.")];
await using StreamingRun agentRun = await InProcessExecution.RunStreamingAsync(
    agentWorkflow,
    agentMessages);
await agentRun.TrySendMessageAsync(new TurnToken(emitEvents: true));

string? lastAuthor = null;
await foreach (WorkflowEvent evt in agentRun.WatchStreamAsync())
{
    if (evt is AgentResponseUpdateEvent update)
    {
        if (update.ExecutorId != lastAuthor)
        {
            lastAuthor = update.ExecutorId;
            Console.WriteLine($"\n[{update.ExecutorId}]");
        }
        Console.Write(update.Update.Text);
    }
    else if (evt is WorkflowOutputEvent)
    {
        break;
    }
}
Console.WriteLine();

// ══════════════════════════════════════════════════════════════════════════ //
//  Executor Definitions                                                     //
// ══════════════════════════════════════════════════════════════════════════ //

/// <summary>
/// Reverses the input string and sends it to the next executor.
/// </summary>
internal sealed class ReverseTextExecutor()
    : Executor<string, string>("reverse")
{
    public override ValueTask<string> HandleAsync(
        string message,
        IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        return ValueTask.FromResult(string.Concat(message.Reverse()));
    }
}

/// <summary>
/// Appends exclamation marks and yields the result as workflow output.
/// </summary>
internal sealed class ExclaimExecutor()
    : Executor<string, string>("exclaim")
{
    public override ValueTask<string> HandleAsync(
        string message,
        IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        return ValueTask.FromResult($"{message}!!!");
    }
}
