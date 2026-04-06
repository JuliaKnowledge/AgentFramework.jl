# Getting Started

This guide walks you through installing AgentFramework.jl, setting up a provider, and building your first AI agent.

## Prerequisites

- **Julia 1.9+** (1.10+ recommended)
- **Ollama** installed and running locally ([ollama.com](https://ollama.com)), or an OpenAI API key
- A model pulled in Ollama, e.g. `ollama pull qwen3:8b`

## Installation

AgentFramework.jl is installed directly from its Git repository:

```julia
using Pkg
Pkg.add(url="https://github.com/sdwfrost/AgentFramework.jl")
```

Or in the Pkg REPL (press `]`):

```
pkg> add https://github.com/sdwfrost/AgentFramework.jl
```

## Your First Agent

Let's create a simple conversational agent using Ollama as the LLM provider.

### Step 1: Create a Chat Client

The chat client connects to your LLM provider. [`OllamaChatClient`](@ref) talks to a local Ollama instance:

```julia
using AgentFramework

client = OllamaChatClient(model="qwen3:8b")
```

The default `base_url` is `http://localhost:11434`. Override it if Ollama runs elsewhere:

```julia
client = OllamaChatClient(model="qwen3:8b", base_url="http://myserver:11434")
```

### Step 2: Create an Agent

An [`Agent`](@ref) wraps a chat client with instructions, tools, and middleware:

```julia
agent = Agent(
    name = "Assistant",
    instructions = "You are a helpful assistant. Be concise in your responses.",
    client = client,
)
```

### Step 3: Run the Agent

Use [`run_agent`](@ref) to send a message and get a response:

```julia
response = run_agent(agent, "What is Julia known for?")
println(response.text)
```

The returned [`AgentResponse`](@ref) contains the response messages, token usage, finish reason, and more.

## Adding Tools

Tools let the agent call Julia functions to perform actions or retrieve information. The [`@tool`](@ref) macro converts a function definition into a [`FunctionTool`](@ref) with automatic JSON Schema generation.

### Step 1: Define a Tool

```julia
@tool function get_weather(location::String, unit::String="celsius")
    "Get the current weather for a location."
    # In a real app, call a weather API here
    return "Sunny, 22°C in $(location)"
end
```

The `@tool` macro:
1. Defines the function as normal Julia code
2. Inspects parameter names and types to build a JSON Schema
3. Uses the first string literal in the body as the tool description
4. Assigns a [`FunctionTool`](@ref) to the function name

### Step 2: Give the Tool to an Agent

```julia
agent = Agent(
    name = "WeatherBot",
    instructions = "You are a weather assistant. Use the get_weather tool to answer weather questions.",
    client = client,
    tools = [get_weather],
)

response = run_agent(agent, "What's the weather in Tokyo?")
println(response.text)
```

When the LLM decides to call `get_weather`, the framework automatically:
1. Parses the tool call arguments from the LLM response
2. Invokes the Julia function with those arguments
3. Sends the result back to the LLM
4. Returns the final text response

## Multi-Turn Conversations with Sessions

By default, each [`run_agent`](@ref) call is stateless. To maintain conversation history across turns, use an [`AgentSession`](@ref) with an [`InMemoryHistoryProvider`](@ref):

```julia
# Create a history provider to store conversation turns
history = InMemoryHistoryProvider()

# Create an agent with the history provider
agent = Agent(
    name = "ChatBot",
    instructions = "You are a friendly chatbot. Remember previous messages.",
    client = client,
    context_providers = [history],
)

# Create a session to track conversation state
session = create_session(agent)

# First turn
response1 = run_agent(agent, "My name is Alice.", session=session)
println("Bot: ", response1.text)

# Second turn — the agent remembers the first turn
response2 = run_agent(agent, "What's my name?", session=session)
println("Bot: ", response2.text)
```

The [`InMemoryHistoryProvider`](@ref) automatically saves and loads conversation history for each session. For persistent storage, see the [Sessions & Memory](@ref) guide.

## Complete Example

Here's a complete, runnable example combining everything:

```julia
using AgentFramework

# 1. Connect to Ollama
client = OllamaChatClient(model="qwen3:8b")

# 2. Define tools
@tool function calculate(expression::String)
    "Evaluate a mathematical expression and return the result."
    result = eval(Meta.parse(expression))
    return string(result)
end

@tool function current_time()
    "Get the current date and time."
    return string(Dates.now())
end

# 3. Create the agent
agent = Agent(
    name = "SmartAssistant",
    instructions = """You are a helpful assistant with access to tools.
    Use the calculate tool for math questions.
    Use the current_time tool when asked about the time.""",
    client = client,
    tools = [calculate, current_time],
)

# 4. Set up multi-turn conversation
history = InMemoryHistoryProvider()
agent_with_history = Agent(
    name = agent.name,
    instructions = agent.instructions,
    client = client,
    tools = [calculate, current_time],
    context_providers = [history],
)
session = create_session(agent_with_history)

# 5. Chat loop
messages = [
    "What is 42 * 17?",
    "What time is it?",
    "Add 100 to the first result you gave me.",
]

for msg in messages
    println("\nYou: ", msg)
    response = run_agent(agent_with_history, msg, session=session)
    println("Bot: ", response.text)
end
```

## Next Steps

- [Agents](@ref) — Learn about agent types, the builder pattern, and structured output
- [Tools](@ref) — Deep dive into tool schemas, invocation lifecycle, and advanced options
- [Middleware](@ref) — Add logging, retry logic, and custom interception
- [Streaming](@ref) — Get real-time token-by-token output
- [Providers](@ref) — Configure OpenAI, Azure, Anthropic, and other LLM providers
