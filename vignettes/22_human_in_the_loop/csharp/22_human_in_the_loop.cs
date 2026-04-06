// Human-in-the-Loop Workflows — C#
//
// This sample demonstrates human-in-the-loop patterns in workflows:
//   1. Using RequestPort to pause a workflow and request human input.
//   2. Resuming a workflow with SendResponseAsync.
//   3. A multi-step approval pipeline with writer, checker, and approver.
//
// A RequestPort acts as a workflow node that emits a RequestInfoEvent when
// it receives a message, pausing execution until an ExternalResponse is
// provided via SendResponseAsync.
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
//  1. Simple Review Workflow with Human-in-the-Loop                        //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== Review Workflow with Human-in-the-Loop ===\n");

var chatClient = new OllamaApiClient(new Uri(endpoint), modelName);

// Writer agent generates a draft from the user's topic.
ChatClientAgent writer = new(chatClient,
    "You write a short paragraph about the given topic.",
    "Writer");

// RequestPort pauses the workflow and surfaces the draft to a human reviewer.
// It receives a ReviewSignal (the draft text) and expects a string response.
RequestPort reviewPort = RequestPort.Create<ReviewSignal, string>("Review");

// After the human responds, this executor decides the outcome.
ReviewResultExecutor reviewResult = new();

var reviewWorkflow = new WorkflowBuilder(writer)
    .AddEdge(writer, reviewPort)
    .AddEdge(reviewPort, reviewResult)
    .WithOutputFrom(reviewResult)
    .Build();

List<ChatMessage> input = [new ChatMessage(ChatRole.User, "Write about the Julia programming language")];

await using StreamingRun reviewRun = await InProcessExecution.RunStreamingAsync(
    reviewWorkflow, input);

await foreach (WorkflowEvent evt in reviewRun.WatchStreamAsync())
{
    switch (evt)
    {
        case RequestInfoEvent requestInfoEvt:
            Console.WriteLine($"Workflow paused — request ID: {requestInfoEvt.Request.RequestId}");
            if (requestInfoEvt.Request.TryGetDataAs<ReviewSignal>(out var reviewData))
            {
                Console.WriteLine($"  Draft preview: {reviewData.Draft[..Math.Min(100, reviewData.Draft.Length)]}...");
            }

            // Simulate human approval
            Console.WriteLine("  Human response: approve");
            ExternalResponse response = requestInfoEvt.Request.CreateResponse("approve");
            await reviewRun.SendResponseAsync(response);
            break;

        case WorkflowOutputEvent outputEvt:
            Console.WriteLine($"\nOutput: {outputEvt.Data}");
            goto done_review;
    }
}
done_review:

// ══════════════════════════════════════════════════════════════════════════ //
//  2. Multi-Step Approval Pipeline                                         //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("\n=== Multi-Step Approval Pipeline ===\n");

ChatClientAgent pipelineWriter = new(chatClient,
    "You write a one-paragraph article about the topic.",
    "Writer");

ChatClientAgent factChecker = new(chatClient,
    "You review the text for factual accuracy. Output PASS or list corrections.",
    "FactChecker");

// Approval gate: pauses for human sign-off after fact-checking.
RequestPort approvalPort = RequestPort.Create<ApprovalSignal, string>("Approval");

ApprovalResultExecutor approvalResult = new();

var pipeline = new WorkflowBuilder(pipelineWriter)
    .AddEdge(pipelineWriter, factChecker)
    .AddEdge(factChecker, approvalPort)
    .AddEdge(approvalPort, approvalResult)
    .WithOutputFrom(approvalResult)
    .Build();

List<ChatMessage> pipelineInput = [new ChatMessage(ChatRole.User,
    "The history of the Julia programming language")];

await using StreamingRun pipelineRun = await InProcessExecution.RunStreamingAsync(
    pipeline, pipelineInput);

await foreach (WorkflowEvent evt in pipelineRun.WatchStreamAsync())
{
    switch (evt)
    {
        case RequestInfoEvent requestInfoEvt:
            Console.WriteLine($"Approval requested — request ID: {requestInfoEvt.Request.RequestId}");
            if (requestInfoEvt.Request.TryGetDataAs<ApprovalSignal>(out var approvalData))
            {
                Console.WriteLine($"  Content preview: {approvalData.Content[..Math.Min(200, approvalData.Content.Length)]}...");
            }

            // Human approves
            Console.WriteLine("  Human response: approve");
            ExternalResponse approvalResponse = requestInfoEvt.Request.CreateResponse("approve");
            await pipelineRun.SendResponseAsync(approvalResponse);
            break;

        case WorkflowOutputEvent outputEvt:
            Console.WriteLine($"\n{outputEvt.Data}");
            goto done_pipeline;
    }
}
done_pipeline:
Console.WriteLine();

// ══════════════════════════════════════════════════════════════════════════ //
//  Signal & Executor Definitions                                            //
// ══════════════════════════════════════════════════════════════════════════ //

/// <summary>Signal sent to the review RequestPort carrying the draft text.</summary>
internal sealed record ReviewSignal(string Draft);

/// <summary>Signal sent to the approval RequestPort carrying fact-checked content.</summary>
internal sealed record ApprovalSignal(string Content);

/// <summary>
/// Processes the human reviewer's response (the string returned through the
/// RequestPort) and yields a final verdict as workflow output.
/// </summary>
[YieldsOutput(typeof(string))]
internal sealed class ReviewResultExecutor() : Executor<string>("ReviewResult")
{
    public override async ValueTask HandleAsync(
        string message,
        IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        if (message.Trim().Equals("approve", StringComparison.OrdinalIgnoreCase))
        {
            await context.YieldOutputAsync("✅ Approved and published", cancellationToken);
        }
        else
        {
            await context.YieldOutputAsync($"❌ Revision needed: {message}", cancellationToken);
        }
    }
}

/// <summary>
/// Processes the human approver's response and yields a publication decision.
/// </summary>
[YieldsOutput(typeof(string))]
internal sealed class ApprovalResultExecutor() : Executor<string>("ApprovalResult")
{
    public override async ValueTask HandleAsync(
        string message,
        IWorkflowContext context,
        CancellationToken cancellationToken = default)
    {
        if (message.Trim().Equals("approve", StringComparison.OrdinalIgnoreCase))
        {
            await context.YieldOutputAsync("✅ Published", cancellationToken);
        }
        else
        {
            await context.YieldOutputAsync($"❌ Rejected: {message}", cancellationToken);
        }
    }
}
