// Mem0 Persistent Memory — C#
//
// This sample demonstrates Mem0 integration for persistent agent memory:
//   1. Connecting to Mem0 with HttpClient (platform deployment).
//   2. Using Mem0Provider for automatic memory injection.
//   3. Scoping memories by user, agent, and application.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore
//   - A Mem0 API key (set MEM0_API_KEY env var) or local Mem0 instance

using System;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Mem0;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";
var mem0ApiKey = Environment.GetEnvironmentVariable("MEM0_API_KEY")
    ?? "your-mem0-api-key";

IChatClient chatClient = new OllamaApiClient(new Uri(endpoint), modelName);

// ── Configure Mem0 HTTP client ─────────────────────────────────────────────
using var httpClient = new HttpClient();
httpClient.BaseAddress = new Uri("https://api.mem0.ai");
httpClient.DefaultRequestHeaders.Authorization =
    new AuthenticationHeaderValue("Token", mem0ApiKey);

// ── 1. Basic Mem0 memory across sessions ───────────────────────────────────
Console.WriteLine("=== 1. Basic Mem0 Memory Across Sessions ===\n");

var basicProvider = new Mem0Provider(
    httpClient,
    session => new Mem0Provider.State(
        new Mem0ProviderScope { UserId = "user-123" }
    )
);

AIAgent memoryAgent = chatClient.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new()
    {
        Instructions = "You remember what users tell you across conversations. "
            + "Keep answers brief."
    },
    Name = "MemoryAgent",
    AIContextProviders = [basicProvider]
});

// First conversation — agent learns about the user
AgentSession session1 = await memoryAgent.CreateSessionAsync();
Console.WriteLine(await memoryAgent.RunAsync(
    "My name is Alice and I love Julia programming.", session1));

// New session — memories persist via Mem0
AgentSession session2 = await memoryAgent.CreateSessionAsync();
Console.WriteLine(await memoryAgent.RunAsync(
    "What do you know about me?", session2));

// ── 2. Agent-scoped memory isolation ───────────────────────────────────────
Console.WriteLine("\n=== 2. Agent-Scoped Memory Isolation ===\n");

var personalProvider = new Mem0Provider(
    httpClient,
    session => new Mem0Provider.State(
        new Mem0ProviderScope
        {
            AgentId = "personal-assistant",
            UserId = "user-123"
        }
    )
);

var workProvider = new Mem0Provider(
    httpClient,
    session => new Mem0Provider.State(
        new Mem0ProviderScope
        {
            AgentId = "work-assistant",
            UserId = "user-123"
        }
    )
);

AIAgent personalAgent = chatClient.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new()
    {
        Instructions = "You help with personal tasks and remember preferences."
    },
    Name = "PersonalAssistant",
    AIContextProviders = [personalProvider]
});

AIAgent workAgent = chatClient.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new()
    {
        Instructions = "You help with professional tasks and remember work context."
    },
    Name = "WorkAssistant",
    AIContextProviders = [workProvider]
});

// Store personal information
AgentSession pSession1 = await personalAgent.CreateSessionAsync();
Console.WriteLine(await personalAgent.RunAsync(
    "Remember that I exercise at 6 AM and prefer outdoor activities.", pSession1));

// Store work information
AgentSession wSession1 = await workAgent.CreateSessionAsync();
Console.WriteLine(await workAgent.RunAsync(
    "Remember that I have team meetings every Tuesday at 2 PM.", wSession1));

// Each agent only sees its own scoped memories
AgentSession pSession2 = await personalAgent.CreateSessionAsync();
Console.WriteLine(await personalAgent.RunAsync(
    "What do you know about my schedule?", pSession2));

AgentSession wSession2 = await workAgent.CreateSessionAsync();
Console.WriteLine(await workAgent.RunAsync(
    "What do you know about my schedule?", wSession2));

// ── 3. Application-scoped memory ──────────────────────────────────────────
Console.WriteLine("\n=== 3. Application-Scoped Memory ===\n");

var appProvider = new Mem0Provider(
    httpClient,
    session => new Mem0Provider.State(
        new Mem0ProviderScope
        {
            ApplicationId = "my-app",
            UserId = "user-456"
        }
    )
);

AIAgent appAgent = chatClient.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new()
    {
        Instructions = "You are an assistant scoped to a specific application."
    },
    Name = "AppAgent",
    AIContextProviders = [appProvider]
});

AgentSession appSession1 = await appAgent.CreateSessionAsync();
Console.WriteLine(await appAgent.RunAsync(
    "I prefer dark mode and metric units.", appSession1));

AgentSession appSession2 = await appAgent.CreateSessionAsync();
Console.WriteLine(await appAgent.RunAsync(
    "What are my preferences?", appSession2));

// ── 4. Separate storage and search scopes ─────────────────────────────────
Console.WriteLine("\n=== 4. Dual Scope (Storage vs Search) ===\n");

var dualScopeProvider = new Mem0Provider(
    httpClient,
    session => new Mem0Provider.State(
        storageScope: new Mem0ProviderScope
        {
            UserId = "user-789",
            AgentId = "dual-agent"
        },
        searchScope: new Mem0ProviderScope
        {
            UserId = "user-789",
            AgentId = "dual-agent",
            ApplicationId = "my-app"
        }
    )
);

AIAgent dualAgent = chatClient.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new()
    {
        Instructions = "You remember user preferences with fine-grained scoping."
    },
    Name = "DualScopeAgent",
    AIContextProviders = [dualScopeProvider]
});

AgentSession dualSession = await dualAgent.CreateSessionAsync();
Console.WriteLine(await dualAgent.RunAsync(
    "I am working on the data pipeline project.", dualSession));
