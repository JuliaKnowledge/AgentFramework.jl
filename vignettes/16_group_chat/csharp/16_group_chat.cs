// Group Chat Orchestrations — C#
//
// This sample demonstrates three multi-agent group chat patterns:
//   1. Round-robin: agents take turns in fixed order.
//   2. Selector-based: an orchestrator agent picks the next speaker.
//   3. Magentic: a manager plans, tracks progress, and adapts.
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

// ── Debate participants ─────────────────────────────────────────────────── //

ChatClientAgent optimist = new(chatClient,
    instructions: "You are an optimistic philosopher. You see the bright side "
        + "of every argument. Keep responses to 2-3 sentences. Build on what "
        + "others have said and find the silver lining.",
    name: "Optimist",
    description: "Sees the positive side of every argument");

ChatClientAgent pessimist = new(chatClient,
    instructions: "You are a pessimistic philosopher. You challenge assumptions "
        + "and point out risks. Keep responses to 2-3 sentences. Respectfully "
        + "counter the previous speaker's points.",
    name: "Pessimist",
    description: "Challenges assumptions and highlights risks");

ChatClientAgent moderator = new(chatClient,
    instructions: "You are a neutral moderator. Summarise the discussion so far "
        + "and ask a probing follow-up question to deepen the debate. Keep "
        + "responses to 2-3 sentences.",
    name: "Moderator",
    description: "Summarises discussion and asks follow-up questions");

// ══════════════════════════════════════════════════════════════════════════ //
//  1. Round-Robin Group Chat                                                //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== Round-Robin Group Chat ===\n");

var roundRobinWorkflow = AgentWorkflowBuilder
    .CreateGroupChatBuilderWith(agents => new RoundRobinGroupChatManager(agents) { MaximumIterationCount = 6 })
    .AddParticipants(optimist, pessimist, moderator)
    .WithName("PhilosophyRoundRobin")
    .Build();

List<ChatMessage> rrMessages = [new ChatMessage(ChatRole.User,
    "Is technology making humanity better or worse?")];

await using (StreamingRun rrRun = await InProcessExecution.RunStreamingAsync(
    roundRobinWorkflow, rrMessages))
{
    await rrRun.TrySendMessageAsync(new TurnToken(emitEvents: true));

    string? lastAgent = null;
    await foreach (WorkflowEvent evt in rrRun.WatchStreamAsync())
    {
        if (evt is AgentResponseUpdateEvent update)
        {
            if (update.ExecutorId != lastAgent)
            {
                if (lastAgent is not null) Console.WriteLine();
                lastAgent = update.ExecutorId;
                Console.Write($"[{update.ExecutorId}] ");
            }
            Console.Write(update.Update.Text);
        }
        else if (evt is WorkflowOutputEvent)
        {
            break;
        }
    }
    Console.WriteLine("\n");
}

// ══════════════════════════════════════════════════════════════════════════ //
//  2. Selector-Based Group Chat                                             //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== Selector-Based Group Chat ===\n");

ChatClientAgent coordinator = new(chatClient,
    instructions: "You are a debate coordinator. Given the conversation so far, "
        + "decide which participant should speak next. Consider who would add "
        + "the most value. Reply with ONLY the name: Optimist, Pessimist, or "
        + "Moderator.",
    name: "Coordinator",
    description: "Selects the next speaker in a debate");

// The coordinator agent drives selection by examining conversation history
// and returning the next speaker's name as structured output.

var selectorWorkflow = AgentWorkflowBuilder
    .CreateGroupChatBuilderWith(agents => new SelectorGroupChatManager(agents, coordinator) { MaximumIterationCount = 6 })
    .AddParticipants(optimist, pessimist, moderator)
    .WithName("PhilosophySelector")
    .Build();

List<ChatMessage> selMessages = [new ChatMessage(ChatRole.User,
    "Should artificial intelligence have rights?")];

await using (StreamingRun selRun = await InProcessExecution.RunStreamingAsync(
    selectorWorkflow, selMessages))
{
    await selRun.TrySendMessageAsync(new TurnToken(emitEvents: true));

    string? lastAgent = null;
    await foreach (WorkflowEvent evt in selRun.WatchStreamAsync())
    {
        if (evt is AgentResponseUpdateEvent update)
        {
            if (update.ExecutorId != lastAgent)
            {
                if (lastAgent is not null) Console.WriteLine();
                lastAgent = update.ExecutorId;
                Console.Write($"[{update.ExecutorId}] ");
            }
            Console.Write(update.Update.Text);
        }
        else if (evt is WorkflowOutputEvent)
        {
            break;
        }
    }
    Console.WriteLine("\n");
}

// ══════════════════════════════════════════════════════════════════════════ //
//  3. Magentic Orchestration                                                //
// ══════════════════════════════════════════════════════════════════════════ //

Console.WriteLine("=== Magentic Orchestration ===\n");

ChatClientAgent researcher = new(chatClient,
    instructions: "You are a philosophical researcher. Provide relevant "
        + "historical context, cite key thinkers, and identify core tensions. "
        + "Keep responses to 3-4 sentences.",
    name: "Researcher",
    description: "Provides historical context and identifies core tensions");

ChatClientAgent analyst = new(chatClient,
    instructions: "You are a philosophical analyst. Evaluate arguments for "
        + "logical consistency, identify fallacies, and suggest stronger "
        + "formulations. Keep responses to 3-4 sentences.",
    name: "Analyst",
    description: "Evaluates logical consistency and strengthens arguments");

ChatClientAgent planner = new(chatClient,
    instructions: "You are an orchestrator managing a team of agents. "
        + "Given the conversation so far and the available participants, decide "
        + "which agent should contribute next. Reply with ONLY the agent name.",
    name: "Planner",
    description: "Plans which agent should contribute next");

// Magentic orchestration with planning and progress tracking.
// The manager creates a task ledger, selects participants based on the plan,
// and replans if the conversation stalls.

var magenticWorkflow = AgentWorkflowBuilder
    .CreateGroupChatBuilderWith(agents => new MagenticGroupChatManager(agents, planner) { MaximumIterationCount = 6 })
    .AddParticipants(researcher, analyst)
    .WithName("PhilosophyMagentic")
    .Build();

List<ChatMessage> magMessages = [new ChatMessage(ChatRole.User,
    "Analyse the trolley problem and its implications for autonomous vehicles.")];

await using (StreamingRun magRun = await InProcessExecution.RunStreamingAsync(
    magenticWorkflow, magMessages))
{
    await magRun.TrySendMessageAsync(new TurnToken(emitEvents: true));

    string? lastAgent = null;
    await foreach (WorkflowEvent evt in magRun.WatchStreamAsync())
    {
        if (evt is AgentResponseUpdateEvent update)
        {
            if (update.ExecutorId != lastAgent)
            {
                if (lastAgent is not null) Console.WriteLine();
                lastAgent = update.ExecutorId;
                Console.Write($"[{update.ExecutorId}] ");
            }
            Console.Write(update.Update.Text);
        }
        else if (evt is WorkflowOutputEvent output)
        {
            Console.WriteLine($"\n\n=== Final Answer ===\n{output.Data}");
            break;
        }
    }
    Console.WriteLine();
}

// ── Custom Group Chat Managers ───────────────────────────────────────────── //

/// <summary>
/// Selector-based group chat manager that uses a coordinator agent to
/// pick the next speaker based on conversation history.
/// </summary>
class SelectorGroupChatManager(
    IReadOnlyList<AIAgent> agents,
    AIAgent selectorAgent) : GroupChatManager
{
    protected override async ValueTask<AIAgent> SelectNextAgentAsync(
        IReadOnlyList<ChatMessage> history,
        CancellationToken cancellationToken = default)
    {
        var names = string.Join(", ", agents.Select(a => a.Name));
        var historyText = string.Join("\n",
            history.Select(m => $"[{m.AuthorName ?? m.Role.ToString()}]: {m.Text}"));

        var prompt = $"Conversation so far:\n{historyText}\n\n"
            + $"Participants: {names}\nWho should speak next? Reply with ONLY the name.";

        var response = await selectorAgent.RunAsync(prompt, cancellationToken: cancellationToken);
        var chosen = response.Text?.Trim();

        return agents.FirstOrDefault(a =>
            chosen is not null && chosen.Contains(a.Name!, StringComparison.OrdinalIgnoreCase))
            ?? agents[IterationCount % agents.Count];
    }
}

/// <summary>
/// Magentic-style manager that uses a planner agent to orchestrate
/// participants based on their descriptions and conversation progress.
/// </summary>
class MagenticGroupChatManager(
    IReadOnlyList<AIAgent> agents,
    AIAgent plannerAgent) : GroupChatManager
{
    protected override async ValueTask<AIAgent> SelectNextAgentAsync(
        IReadOnlyList<ChatMessage> history,
        CancellationToken cancellationToken = default)
    {
        var agentInfo = string.Join("\n", agents.Select(a => $"- {a.Name}: {a.Description}"));
        var historyText = string.Join("\n",
            history.Select(m => $"[{m.AuthorName ?? m.Role.ToString()}]: {m.Text}"));

        var prompt = $"You are orchestrating a team. Available agents:\n{agentInfo}\n\n"
            + $"Conversation so far:\n{historyText}\n\n"
            + "Based on the progress, which agent should contribute next? Reply with ONLY the agent name.";

        var response = await plannerAgent.RunAsync(prompt, cancellationToken: cancellationToken);
        var chosen = response.Text?.Trim();

        return agents.FirstOrDefault(a =>
            chosen is not null && chosen.Contains(a.Name!, StringComparison.OrdinalIgnoreCase))
            ?? agents[IterationCount % agents.Count];
    }
}
