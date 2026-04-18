# AgentFramework.jl

*A Julia framework for building, orchestrating, and deploying AI agents.*

AgentFramework.jl is a Julia port of [Microsoft Agent Framework](https://github.com/microsoft/agent-framework), bringing its layered pipeline architecture to the Julia ecosystem. It provides a composable system for creating AI agents powered by large language models, with built-in support for tool calling, middleware pipelines, multi-agent workflows, streaming, and multiple LLM providers.

## Architecture

```
User ─── "What's the weather in London?" ───▶ Agent.run_agent()
                                                │
                                     ┌──────────┴──────────┐
                                     │  Context Providers   │
                                     │  (history, memory,   │
                                     │   instructions)      │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────┴──────────┐
                                     │  Agent Middleware     │
                                     │  (telemetry, logging) │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────┴──────────┐
                                     │  Chat Middleware      │
                                     │  (retry, rate limit)  │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────┴──────────┐
                                     │  Chat Client          │
                                     │  (Ollama, OpenAI,     │
                                     │   Azure, Anthropic)   │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────┴──────────┐
                                     │  Tool Execution Loop  │
                                     │  (call tools, feed    │
                                     │   results back)       │
                                     └──────────┬──────────┘
                                                │
                                     ◀── AgentResponse ─────┘
```

## Key Features

- **Multiple LLM Providers** — [`OllamaChatClient`](@ref), [`OpenAIChatClient`](@ref), [`AzureOpenAIChatClient`](@ref), [`AnthropicChatClient`](@ref), [`FoundryChatClient`](@ref)
- **Tool Calling** — Define tools from Julia functions with the [`@tool`](@ref) macro; automatic JSON Schema generation
- **Middleware Pipelines** — Three-layer interception (agent, chat, function) for logging, retry, rate limiting, and custom logic
- **Streaming** — Real-time token-by-token output via [`run_agent_streaming`](@ref) and `Channel`-based iteration
- **Multi-Agent Workflows** — DAG-based orchestration with [`WorkflowBuilder`](@ref), fan-out/fan-in edges, and checkpointing
- **Session & Memory** — Persistent conversation history with [`AgentSession`](@ref), multiple memory stores, and context providers
- **Structured Output** — Parse LLM responses into typed Julia structs
- **Declarative Agents** — Define agents and workflows from YAML/JSON configuration
- **Resilience** — Built-in retry with exponential backoff and rate limiting via [`RetryConfig`](@ref)
- **Capability System** — Query provider features with traits like [`HasStreaming`](@ref), [`HasToolCalling`](@ref)
- **Telemetry** — OpenTelemetry-aligned spans and metrics via [`TelemetrySpan`](@ref)
- **MCP Integration** — Connect to external tools via Model Context Protocol ([`StdioMCPClient`](@ref), [`HTTPMCPClient`](@ref))
- **Evaluation** — Test agent quality with [`evaluate_agent`](@ref) and assertion-based checks

## Quick Example

```julia
using AgentFramework

# Connect to a local Ollama instance
client = OllamaChatClient(model="qwen3:8b")

# Define a tool from a regular Julia function
@tool function get_weather(location::String, unit::String="celsius")
    "Get the current weather for a location."
    return "Sunny, 22°C in $(location)"
end

# Create an agent with instructions and tools
agent = Agent(
    name = "WeatherBot",
    instructions = "You are a helpful weather assistant. Use the get_weather tool when asked about weather.",
    client = client,
    tools = [get_weather],
)

# Run the agent
response = run_agent(agent, "What's the weather in London?")
println(response.text)
```

## Guide

Learn how to use AgentFramework.jl step by step:

| Section | Description |
|:--------|:------------|
| [Getting Started](@ref) | Installation, prerequisites, and your first agent |
| [Agents](agents.md) | Agent types, builder pattern, and running agents |
| [Tools](tools.md) | The `@tool` macro, schemas, and tool invocation |
| [Middleware](middleware.md) | Three-layer middleware pipeline for interception |
| [Sessions & Memory](@ref) | Conversation state, history providers, and memory stores |
| [Workflows](workflows.md) | Multi-agent orchestration with DAG-based workflows |
| [Providers](@ref) | LLM provider configuration and capabilities |
| [Streaming](streaming.md) | Real-time streaming responses |
| [Advanced Topics](@ref) | Declarative agents, MCP, compaction, evaluation, and more |

## Submodules

AgentFramework.jl includes several integrated submodules for extended functionality:

- **`A2A`** — Agent-to-Agent protocol for REST-based inter-agent communication
- **[`Hosting`](@ref)** — Deploy agents as HTTP services
- **`Mem0Integration`** — Long-term memory via Mem0
- **[`Bedrock`](@ref)** — AWS Bedrock provider support
- **`CodingAgents`** — Specialized agents for code generation and editing

## API Reference

For complete API documentation, see the [API Reference](api/agents.md) section.
