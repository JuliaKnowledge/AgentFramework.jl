// Multi-Agent Handoffs — C#
//
// This sample demonstrates agent-to-agent delegation:
//   1. A triage agent that routes to specialist agents.
//   2. Using AsAIFunction() to expose an agent as a callable tool.
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

var chatClient = new OllamaApiClient(new Uri(endpoint), modelName);

// ── Specialist agents ──────────────────────────────────────────────────── //

AIAgent mathAgent = chatClient.AsAIAgent(
    instructions: "You are a math expert. Solve problems step by step. "
        + "Show your working clearly. Only answer math questions.",
    name: "MathExpert",
    description: "Specialist for math and arithmetic questions.");

AIAgent generalAgent = chatClient.AsAIAgent(
    instructions: "You are a general knowledge assistant. "
        + "Answer questions about history, science, geography, and culture. "
        + "Keep answers concise.",
    name: "GeneralAssistant",
    description: "Handles general knowledge questions.");

// ── Triage agent (uses specialists as function tools) ──────────────────── //
// AsAIFunction() converts an agent into a callable tool that the triage
// agent can invoke to hand off questions.

AIAgent triageAgent = chatClient.AsAIAgent(
    instructions: "You are a routing agent. Your ONLY job is to decide which "
        + "specialist should handle the user's question:\n"
        + "- For math/arithmetic → call MathExpert\n"
        + "- For everything else → call GeneralAssistant\n"
        + "Always delegate. Never answer directly.",
    name: "TriageAgent",
    tools: [mathAgent.AsAIFunction(), generalAgent.AsAIFunction()]);

// ── Run: math question ─────────────────────────────────────────────────── //

Console.WriteLine("=== Math Question ===\n");
Console.WriteLine(await triageAgent.RunAsync("What is the integral of x^2?"));

// ── Run: general knowledge question ────────────────────────────────────── //

Console.WriteLine("\n=== General Knowledge Question ===\n");
Console.WriteLine(await triageAgent.RunAsync("Who painted the Mona Lisa?"));

// ── Alternative: AgentWorkflowBuilder handoff pattern ──────────────────── //
// For more complex multi-turn handoff workflows, use AgentWorkflowBuilder.

Console.WriteLine("\n=== Workflow-Based Handoffs ===\n");

ChatClientAgent triageWf = new(chatClient,
    "You determine which specialist to use. Always hand off to another agent.",
    "triage",
    "Routes messages to the appropriate specialist");
ChatClientAgent mathWf = new(chatClient,
    "You solve math problems step by step.",
    "math_expert",
    "Specialist for math questions");
ChatClientAgent generalWf = new(chatClient,
    "You answer general knowledge questions concisely.",
    "general_expert",
    "Specialist for general knowledge questions");

var workflow = AgentWorkflowBuilder.CreateHandoffBuilderWith(triageWf)
    .WithHandoffs(triageWf, [mathWf, generalWf])
    .WithHandoffs([mathWf, generalWf], triageWf)
    .Build();

List<ChatMessage> messages = [new ChatMessage(ChatRole.User, "What is 42 * 17?")];
await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
    workflow,
    messages);
await run.TrySendMessageAsync(new TurnToken(emitEvents: true));

string? lastExecutor = null;
await foreach (WorkflowEvent evt in run.WatchStreamAsync())
{
    if (evt is AgentResponseUpdateEvent e)
    {
        if (e.ExecutorId != lastExecutor)
        {
            lastExecutor = e.ExecutorId;
            Console.WriteLine($"\n[{e.ExecutorId}]");
        }
        Console.Write(e.Update.Text);
    }
    else if (evt is WorkflowOutputEvent)
    {
        break;
    }
}
Console.WriteLine();
