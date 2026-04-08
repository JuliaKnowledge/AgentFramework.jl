# Streaming

Streaming lets you display LLM output in real time as tokens arrive, rather than waiting for the complete response. AgentFramework.jl uses Julia's `Channel` system for streaming, providing a natural iteration interface.

## Overview

There are two levels of streaming in AgentFramework.jl:

1. **Agent-level streaming** via [`run_agent_streaming`](@ref) — yields [`AgentResponseUpdate`](@ref) items
2. **Chat client-level streaming** via [`get_response_streaming`](@ref) — yields [`ChatResponseUpdate`](@ref) items

Most users will use agent-level streaming, which handles the tool execution loop automatically.

## Basic Streaming

### run_agent_streaming

[`run_agent_streaming`](@ref) returns a [`ResponseStream{AgentResponseUpdate}`](@ref) that you iterate over:

```julia
using AgentFramework

client = OllamaChatClient(model="qwen3:8b")
agent = Agent(
    name = "Storyteller",
    instructions = "You are a creative storyteller.",
    client = client,
)

# Start streaming
stream = run_agent_streaming(agent, "Tell me a short story about a robot.")

# Print tokens as they arrive
for update in stream
    print(get_text(update))
end
println()  # Final newline
```

### Getting the Final Response

After iterating, retrieve the complete [`AgentResponse`](@ref) with [`get_final_response`](@ref):

```julia
stream = run_agent_streaming(agent, "Hello!")
for update in stream
    print(get_text(update))
end
println()

# Get the aggregated response
response = get_final_response(stream)
println("Model: ", response.model_id)
println("Finish reason: ", response.finish_reason)
if response.usage_details !== nothing
    println("Tokens: ", response.usage_details.input_tokens, " in / ",
            response.usage_details.output_tokens, " out")
end
```

## AgentResponseUpdate

Each [`AgentResponseUpdate`](@ref) contains a chunk of the response:

```julia
stream = run_agent_streaming(agent, "Hello!")
for update in stream
    # Text content (convenience accessor)
    text = update.text

    # Raw content items
    for content in update.contents
        if is_text(content)
            print(get_text(content))
        end
    end

    # Role (usually set on the first update)
    if update.role !== nothing
        println("Role: ", update.role)
    end

    # Finish reason (set on the final update)
    if update.finish_reason !== nothing
        println("\nFinished: ", update.finish_reason)
    end

    # Token usage (usually on the final update)
    if update.usage_details !== nothing
        println("Tokens used: ", update.usage_details.output_tokens)
    end
end
```

## Streaming with Tools

When an agent has tools, streaming works seamlessly with the tool execution loop. Tool calls are assembled from streaming fragments, executed, and the results are sent back to the LLM — which then streams the final response:

```julia
@tool function get_weather(location::String)
    "Get weather for a location."
    return "Sunny, 22°C in $(location)"
end

agent = Agent(
    name = "WeatherBot",
    instructions = "You are a weather assistant.",
    client = client,
    tools = [get_weather],
)

stream = run_agent_streaming(agent, "What's the weather in Paris?")
for update in stream
    # During tool execution, updates may contain tool call content
    # The final response text streams after tool results are processed
    text = get_text(update)
    if !isempty(text)
        print(text)
    end
end
println()
```

## StreamingToolAccumulator

Under the hood, [`StreamingToolAccumulator`](@ref) assembles tool calls from streaming fragments. Tool calls arrive in pieces across multiple chunks:

1. First chunk: tool call ID and function name
2. Subsequent chunks: argument JSON fragments
3. Framework accumulates all fragments into complete tool calls

You typically don't interact with this directly, but it's available for custom streaming logic:

```julia
acc = StreamingToolAccumulator()

# Accumulate fragments (normally done by the framework)
accumulate_tool_call!(acc, 0, call_id="call_123", name="get_weather")
accumulate_tool_call!(acc, 0, arguments_fragment="{\"location\":")
accumulate_tool_call!(acc, 0, arguments_fragment="\"Paris\"}")

# Check state
has_tool_calls(acc)            # true

# Get completed tool calls as Content items
tool_calls = get_accumulated_tool_calls(acc)

# Reset for next iteration
reset_accumulator!(acc)
```

## Streaming with Sessions

Streaming works with sessions for multi-turn conversations:

```julia
history = InMemoryHistoryProvider()
agent = Agent(
    name = "ChatBot",
    instructions = "You are a friendly assistant.",
    client = client,
    context_providers = [history],
)
session = create_session(agent)

# First turn — streaming
stream = run_agent_streaming(agent, "My name is Alice.", session=session)
for update in stream
    print(get_text(update))
end
println()

# Second turn — history is preserved
stream = run_agent_streaming(agent, "What's my name?", session=session)
for update in stream
    print(get_text(update))
end
println()
```

## Chat Client-Level Streaming

For lower-level control, use [`get_response_streaming`](@ref) directly on a chat client:

```julia
client = OllamaChatClient(model="qwen3:8b")
messages = [Message(:user, "Hello!")]
options = ChatOptions()

channel = get_response_streaming(client, messages, options)

updates = ChatResponseUpdate[]
for update in channel
    push!(updates, update)
    print(get_text(update))
end
println()

# Build a complete ChatResponse from the collected updates
response = ChatResponse(updates)
println("Complete text: ", response.text)
```

## Example: Streaming with Progress Display

Here's a complete example with a progress display:

```julia
using AgentFramework

client = OllamaChatClient(model="qwen3:8b")

@tool function analyze_data(dataset::String)
    "Analyze a dataset and return summary statistics."
    sleep(1)  # Simulate work
    return "Dataset '$(dataset)': 1000 rows, 5 columns, mean=42.3, std=12.1"
end

agent = Agent(
    name = "DataAnalyst",
    instructions = "You are a data analysis assistant. Use the analyze_data tool.",
    client = client,
    tools = [analyze_data],
)

println("🤖 Starting analysis...")
stream = run_agent_streaming(agent, "Analyze the sales dataset and explain the findings.")

char_count = Ref(0)
for update in stream
    text = get_text(update)
    if !isempty(text)
        print(text)
        char_count[] += length(text)
    end
end
println("\n\n📊 Response: $(char_count[]) characters")

response = get_final_response(stream)
if response !== nothing && response.usage_details !== nothing
    usage = response.usage_details
    println("📈 Tokens: $(something(usage.input_tokens, 0)) input, $(something(usage.output_tokens, 0)) output")
end
```

## Workflow Streaming

Workflows also support streaming via event channels. See the [Workflows](@ref) guide for details on streaming [`WorkflowEvent`](@ref) items during multi-agent orchestration.

## Next Steps

- [Agents](@ref) — Agent configuration that affects streaming behavior
- [Providers](@ref) — Provider-specific streaming capabilities
- [Middleware](@ref) — How middleware interacts with streaming
