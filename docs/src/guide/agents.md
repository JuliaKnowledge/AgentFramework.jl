# Agents

Agents are the central abstraction in AgentFramework.jl. An agent wraps an LLM chat client with instructions, tools, context providers, and middleware to create an intelligent assistant that can reason, use tools, and maintain conversation state.

## Agent Type Hierarchy

```
AbstractAgent
├── Agent                  # Full-featured agent with middleware and tool execution
├── ChatCompletionAgent    # Lightweight wrapper (single LLM call, no tool loop)
└── AssistantAgent         # Service-backed agent (Azure AI Agent Service)
```

[`AbstractAgent`](@ref) is the base type for all agents. Most users will work with [`Agent`](@ref), which provides the complete feature set.

## Creating an Agent

The [`Agent`](@ref) constructor uses keyword arguments with sensible defaults:

```julia
using AgentFramework

client = OllamaChatClient(model="qwen3:8b")

agent = Agent(
    name = "MyAgent",                    # Display name
    description = "A helpful assistant", # Used when agent is nested as a tool
    instructions = "You are helpful.",   # System prompt prepended to every conversation
    client = client,                     # Required: the LLM chat client
    tools = [],                          # Tools available to this agent
    context_providers = [],              # Context engineering pipeline
    agent_middlewares = [],              # Agent-level middleware
    chat_middlewares = [],               # Chat-level middleware
    function_middlewares = [],           # Tool invocation middleware
    options = ChatOptions(),             # Default chat options
    max_tool_iterations = 10,            # Max tool call rounds before stopping
)
```

Only `client` is required. Everything else has defaults:

```julia
# Minimal agent — just needs a client
agent = Agent(client=client)
```

## Agent Builder Pattern

AgentFramework.jl provides immutable builder functions that create modified copies of an agent, leaving the original unchanged:

```julia
base_agent = Agent(
    name = "Base",
    client = client,
    instructions = "You are a general assistant.",
)

# Create specialized variants
math_agent = with_instructions(base_agent, "You are a math tutor. Show your work step by step.")
math_agent = with_name(math_agent, "MathTutor")
math_agent = with_tools(math_agent, [my_extra_tool])

# Override chat options
fast_agent = with_options(base_agent, ChatOptions(temperature=0.0, max_tokens=100))
```

Available builder functions:

| Function | Description |
|:---------|:------------|
| [`with_instructions`](@ref) | Replace system instructions |
| [`with_tools`](@ref) | Replace the tool set |
| [`with_name`](@ref) | Change the agent name |
| [`with_options`](@ref) | Override default [`ChatOptions`](@ref) |

## Running Agents

### Synchronous Execution

[`run_agent`](@ref) sends input to the agent and returns a complete [`AgentResponse`](@ref):

```julia
# String input
response = run_agent(agent, "Hello, world!")

# Message input
msg = Message(:user, "Hello, world!")
response = run_agent(agent, msg)

# Multiple messages
response = run_agent(agent, [
    Message(:user, "What is 2+2?"),
])

# With session for conversation continuity
session = create_session(agent)
response = run_agent(agent, "Remember this: the code is 42", session=session)
response = run_agent(agent, "What was the code?", session=session)
```

### Streaming Execution

[`run_agent_streaming`](@ref) returns a [`ResponseStream`](@ref) that yields [`AgentResponseUpdate`](@ref) items as they arrive:

```julia
stream = run_agent_streaming(agent, "Write a haiku about Julia.")
for update in stream
    print(get_text(update))  # Print tokens as they arrive
end
println()

# Get the complete response after streaming
final = get_final_response(stream)
```

See the [Streaming](@ref) guide for more details.

### Structured Output

You can parse LLM responses directly into typed Julia structs by passing the type as the first argument:

```julia
Base.@kwdef struct MovieReview
    title::String
    rating::Int
    summary::String
end

result = run_agent(MovieReview, agent, "Review The Matrix")
println(result.value.title)    # "The Matrix"
println(result.value.rating)   # 9
println(result.value.summary)  # "A groundbreaking sci-fi film..."
```

This automatically sets `response_format` on the chat options to request JSON output conforming to the struct's schema.

## AgentResponse

[`AgentResponse`](@ref) is the return type of [`run_agent`](@ref):

```julia
response = run_agent(agent, "Hello!")

# Convenience accessor for text content
println(response.text)

# Detailed fields
response.messages          # Vector{Message} — full response messages
response.finish_reason     # FinishReason — STOP, LENGTH, TOOL_CALLS, etc.
response.usage_details     # UsageDetails — token counts
response.model_id          # String — model that produced the response
response.response_id       # String — provider response identifier
response.conversation_id   # String — provider conversation/session ID
```

## AgentResponseUpdate

[`AgentResponseUpdate`](@ref) is yielded during streaming:

```julia
stream = run_agent_streaming(agent, "Hello!")
for update in stream
    update.text            # Text content in this chunk
    update.role            # :assistant (usually set on first update)
    update.finish_reason   # Set on the final update
    update.contents        # Vector{Content} — raw content items
end
```

## Agent as Tool

Use [`as_tool`](@ref) to convert an agent into a [`FunctionTool`](@ref), enabling agent nesting — one agent can invoke another as a tool:

```julia
# Create a specialist agent
researcher = Agent(
    name = "Researcher",
    description = "Researches topics thoroughly and provides detailed summaries.",
    instructions = "You are a research specialist. Provide detailed, well-sourced answers.",
    client = client,
)

# Create a main agent that uses the researcher as a tool
main_agent = Agent(
    name = "Coordinator",
    instructions = "You coordinate tasks. Use the Researcher tool for research questions.",
    client = client,
    tools = [as_tool(researcher)],
)

response = run_agent(main_agent, "Research the history of the Julia programming language")
```

Options for [`as_tool`](@ref):

```julia
as_tool(agent;
    description = "Custom description",   # Override the tool description
    propagate_session = true,             # Share the caller's session
)
```

## ChatCompletionAgent

[`ChatCompletionAgent`](@ref) is a lightweight agent that makes a single LLM call without the tool execution loop. Use it when you need a simple LLM wrapper without tool calling:

```julia
simple_agent = ChatCompletionAgent(
    name = "Summarizer",
    instructions = "Summarize the following text concisely.",
    client = client,
)
```

## AssistantAgent

[`AssistantAgent`](@ref) is designed for service-backed agents (e.g., Azure AI Agent Service) where the service manages conversation state, tool execution, and history:

```julia
assistant = AssistantAgent(
    name = "AzureAssistant",
    client = azure_client,
    instructions = "You are a service-managed assistant.",
)
```

## Agent Sessions

Every agent run can optionally accept an [`AgentSession`](@ref) for conversation continuity:

```julia
session = create_session(agent)

# Or create one manually
session = AgentSession(id="my-session-123")

# Pass to run_agent
response = run_agent(agent, "Hello", session=session)
```

See the [Sessions & Memory](@ref) guide for details on session persistence and memory stores.

## Next Steps

- [Tools](@ref) — Learn how to define and configure tools
- [Middleware](@ref) — Add interception logic around agent, chat, and tool calls
- [Workflows](@ref) — Orchestrate multiple agents in complex pipelines
