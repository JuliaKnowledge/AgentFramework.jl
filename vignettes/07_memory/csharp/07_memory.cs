// Memory and Context Providers — C#
//
// This sample demonstrates persistent memory using AIContextProvider:
//   1. A custom memory component that remembers user preferences.
//   2. Session-based state that persists across agent invocations.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using System.Text;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// Create the agent with a custom memory provider.
AIAgent agent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new()
        {
            Instructions = "You are a friendly assistant. Keep answers brief. "
                + "Always address the user by name if you know it."
        },
        Name = "MemoryBot",
        AIContextProviders = [new UserPreferenceMemory()]
    });

// ── 1. Stateless behavior (new session each time) ──────────────────────
Console.WriteLine("=== Stateless (separate sessions) ===\n");
Console.WriteLine(await agent.RunAsync("My name is Alice."));
Console.WriteLine(await agent.RunAsync("What is my name?"));

// ── 2. Session-based memory (same session across calls) ────────────────
Console.WriteLine("\n=== Session-based Memory ===\n");
AgentSession session = await agent.CreateSessionAsync();

Console.WriteLine(await agent.RunAsync("My name is Alice.", session));
Console.WriteLine(await agent.RunAsync("What is my name?", session));

// ── 3. Custom DateTime context injection ───────────────────────────────
Console.WriteLine("\n=== DateTime Context Provider ===\n");

AIAgent datetimeAgent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new()
        {
            Instructions = "You are a helpful assistant. Keep answers brief."
        },
        Name = "TimeAwareBot",
        AIContextProviders = [new DateTimeContextProvider()]
    });

AgentSession dtSession = await datetimeAgent.CreateSessionAsync();
Console.WriteLine(await datetimeAgent.RunAsync("What time is it right now?", dtSession));

// -------------------------------------------------------------------------- //
// Custom AIContextProvider: remembers user preferences in session state.      //
// -------------------------------------------------------------------------- //

/// <summary>
/// A simple memory component that extracts and remembers the user's name
/// from conversation messages, injecting it as context on subsequent calls.
/// </summary>
internal sealed class UserPreferenceMemory : AIContextProvider
{
    private readonly ProviderSessionState<UserPrefs> _state;

    public UserPreferenceMemory()
    {
        _state = new ProviderSessionState<UserPrefs>(
            _ => new UserPrefs(),
            nameof(UserPreferenceMemory));
    }

    public override IReadOnlyList<string> StateKeys => [_state.StateKey];

    protected override ValueTask<AIContext> ProvideAIContextAsync(
        InvokingContext context, CancellationToken ct = default)
    {
        var prefs = _state.GetOrInitializeState(context.Session);
        var sb = new StringBuilder();

        if (prefs.UserName is not null)
            sb.AppendLine($"The user's name is {prefs.UserName}.");
        else
            sb.AppendLine("You don't know the user's name yet.");

        return ValueTask.FromResult(new AIContext { Instructions = sb.ToString() });
    }

    protected override ValueTask StoreAIContextAsync(
        InvokedContext context, CancellationToken ct = default)
    {
        var prefs = _state.GetOrInitializeState(context.Session);

        // Simple extraction: look for "my name is X" in user messages.
        foreach (var msg in context.RequestMessages)
        {
            if (msg.Role != ChatRole.User) continue;
            var text = msg.Text ?? "";
            var idx = text.IndexOf("my name is", StringComparison.OrdinalIgnoreCase);
            if (idx >= 0)
            {
                var rest = text[(idx + 11)..].Trim().TrimEnd('.');
                var name = rest.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                    .FirstOrDefault();
                if (name is not null)
                    prefs.UserName = name;
            }
        }

        _state.SaveState(context.Session, prefs);
        return ValueTask.CompletedTask;
    }
}

internal sealed class UserPrefs
{
    public string? UserName { get; set; }
}

// -------------------------------------------------------------------------- //
// DateTimeContextProvider: injects current date/time into system instructions //
// -------------------------------------------------------------------------- //

internal sealed class DateTimeContextProvider : AIContextProvider
{
    protected override ValueTask<AIContext> ProvideAIContextAsync(
        InvokingContext context, CancellationToken ct = default)
    {
        var now = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        return ValueTask.FromResult(new AIContext
        {
            Instructions = $"Current date and time: {now}"
        });
    }
}
