// Message Compaction — Managing context windows (C#)
//
// This sample demonstrates the compaction strategies available in the C#
// agent-framework for reducing message history when conversations exceed
// the model's context window.  It mirrors the Julia vignette 24_compaction.
//
// Prerequisites:
//   - dotnet restore

using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Compaction;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

var logger = NullLogger.Instance;

// ── Build a long conversation ────────────────────────────────────────────

List<ChatMessage> BuildConversation(int turns = 50)
{
    var messages = new List<ChatMessage>
    {
        new(ChatRole.System, "You are a helpful assistant.")
    };
    for (int i = 1; i <= turns; i++)
    {
        messages.Add(new ChatMessage(ChatRole.User, $"Question {i}: {new string('x', 200)}"));
        messages.Add(new ChatMessage(ChatRole.Assistant, $"Answer {i}: {new string('y', 300)}"));
    }
    return messages;
}

List<ChatMessage> BuildToolConversation()
{
    return
    [
        new(ChatRole.System, "You are a calculator agent."),
        new(ChatRole.User, "What is 2+2?"),
        new(ChatRole.Assistant,
        [
            new FunctionCallContent("call_1", "add", new Dictionary<string, object?> { ["a"] = 2, ["b"] = 2 })
        ]),
        new(ChatRole.Tool, [new FunctionResultContent("call_1", "4")]),
        new(ChatRole.Assistant, "2+2 = 4"),
        new(ChatRole.User, "And 10*3?"),
        new(ChatRole.Assistant,
        [
            new FunctionCallContent("call_2", "multiply", new Dictionary<string, object?> { ["a"] = 10, ["b"] = 3 })
        ]),
        new(ChatRole.Tool, [new FunctionResultContent("call_2", "30")]),
        new(ChatRole.Assistant, "10*3 = 30"),
    ];
}

// ── Sliding Window ───────────────────────────────────────────────────────

Console.WriteLine("=== Sliding Window ===");
{
    var messages = BuildConversation(50);
    var strategy = new SlidingWindowCompactionStrategy(
        trigger: CompactionTriggers.TurnsExceed(10),
        target: CompactionTriggers.TurnsExceed(6));

    var compacted = await CompactionProvider.CompactAsync(
        strategy, messages, logger);

    Console.WriteLine($"Kept {compacted.Count()} of {messages.Count} messages");
}

// ── Truncation ───────────────────────────────────────────────────────────

Console.WriteLine("\n=== Truncation ===");
{
    var messages = BuildConversation(50);
    var strategy = new TruncationCompactionStrategy(
        trigger: CompactionTriggers.MessagesExceed(20),
        target: CompactionTriggers.MessagesExceed(10));

    var compacted = await CompactionProvider.CompactAsync(
        strategy, messages, logger);

    Console.WriteLine($"Kept {compacted.Count()} of {messages.Count} messages");
}

// ── Pipeline (chained strategies) ────────────────────────────────────────

Console.WriteLine("\n=== Pipeline ===");
{
    var messages = BuildConversation(50);
    // PipelineCompactionStrategy takes an ordered sequence of strategies.
    // Each child strategy evaluates its own trigger independently.
    var pipeline = new PipelineCompactionStrategy(
        new CompactionStrategy[]
        {
            new TruncationCompactionStrategy(
                trigger: CompactionTriggers.MessagesExceed(40),
                minimumPreservedGroups: 20),
            new SlidingWindowCompactionStrategy(
                trigger: CompactionTriggers.TurnsExceed(10),
                minimumPreservedTurns: 4),
        });

    var compacted = await CompactionProvider.CompactAsync(
        pipeline, messages, logger);

    Console.WriteLine($"Pipeline kept {compacted.Count()} of {messages.Count} messages");
}

// ── Tool-Call Compaction (atomic group handling) ─────────────────────────

Console.WriteLine("\n=== Tool-Call Compaction ===");
{
    // Tool-call/result pairs are grouped atomically — they are always
    // removed together, never split across a compaction boundary.
    var messages = BuildToolConversation();
    var strategy = new TruncationCompactionStrategy(
        trigger: CompactionTriggers.MessagesExceed(4),
        minimumPreservedGroups: 2);

    var compacted = await CompactionProvider.CompactAsync(
        strategy, messages, logger);

    Console.WriteLine($"Kept {compacted.Count()} of {messages.Count} messages (tool pairs stay atomic)");
}

// ── Strategy Summary ─────────────────────────────────────────────────────

Console.WriteLine("\n=== Strategy Summary ===");
Console.WriteLine("TruncationCompactionStrategy   — Remove oldest non-system groups");
Console.WriteLine("SlidingWindowCompactionStrategy — Keep N most recent turns");
Console.WriteLine("PipelineCompactionStrategy     — Chain strategies sequentially");
Console.WriteLine("CompactionTriggers             — Predicate factory (tokens, messages, turns)");
Console.WriteLine("CompactionMessageIndex         — Low-level group/message index");
