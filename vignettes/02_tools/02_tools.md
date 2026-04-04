# Adding Tools to Agents
AgentFramework.jl

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [What Are Tools?](#what-are-tools)
- [Defining Tools with `@tool`](#defining-tools-with-tool)
- [Manual FunctionTool Construction](#manual-functiontool-construction)
- [Creating an Agent with Tools](#creating-an-agent-with-tools)
- [Running the Agent with Tool Use](#running-the-agent-with-tool-use)
- [Inspecting the Tool-Call Cycle](#inspecting-the-tool-call-cycle)
- [Multiple Tool Calls](#multiple-tool-calls)
- [Summary](#summary)

## Overview

Large language models can reason about when to call external functions —
a capability known as *function calling* or *tool use*. This vignette
shows how to:

1.  Define tools with the `@tool` macro.
2.  Construct a `FunctionTool` manually for full control.
3.  Attach tools to an `Agent` and let the model invoke them
    automatically.
4.  Inspect the tool-call/result cycle in the response messages.

## Prerequisites

You need [Ollama](https://ollama.com) running locally with the
`qwen3:8b` model:

``` bash
ollama pull qwen3:8b
```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..",".."))
using AgentFramework
```

## What Are Tools?

When you attach tools to an agent, the framework sends their JSON Schema
descriptions alongside each prompt. The LLM can then choose to *call*
one or more tools instead of generating a text reply. The framework
executes the function, feeds the result back to the model, and the model
produces a final answer that incorporates the tool output.

    User prompt ──► LLM ──► tool call ──► execute function ──► result ──► LLM ──► final answer

## Defining Tools with `@tool`

The `@tool` macro converts a regular Julia function into a
`FunctionTool`. The first string literal in the function body becomes
the tool description, and parameter types are mapped to JSON Schema
automatically:

``` julia
@tool function get_weather(location::String)
    "Get the current weather for a location."
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    temp = rand(10:30)
    return "The weather in $location is $(rand(conditions)) with a high of $(temp)°C."
end
```

The macro produces a `FunctionTool` bound to the name `get_weather`:

``` julia
println(typeof(get_weather))
println(get_weather)
```

We can also define a calculation tool:

``` julia
@tool function calculate(expression::String)
    "Evaluate a mathematical expression and return the result."
    result = eval(Meta.parse(expression))
    return string(result)
end
```

And a tool that looks up information:

``` julia
@tool function get_population(country::String)
    "Get the approximate population of a country in millions."
    populations = Dict(
        "France" => 68,
        "Germany" => 84,
        "Japan" => 125,
        "Brazil" => 214,
        "Australia" => 26,
    )
    pop = get(populations, country, nothing)
    return pop !== nothing ? "$(country) has approximately $(pop) million people." :
                            "Population data not available for $(country)."
end
```

## Manual FunctionTool Construction

For cases where you need full control over the JSON Schema (e.g., adding
descriptions to individual parameters), you can construct a
`FunctionTool` directly:

``` julia
manual_tool = FunctionTool(
    name = "convert_temperature",
    description = "Convert a temperature between Celsius and Fahrenheit.",
    func = (value, from_unit) -> begin
        if from_unit == "celsius"
            return string(value * 9/5 + 32) * "°F"
        else
            return string((value - 32) * 5/9) * "°C"
        end
    end,
    parameters = Dict{String, Any}(
        "type" => "object",
        "properties" => Dict{String, Any}(
            "value" => Dict{String, Any}(
                "type" => "number",
                "description" => "The temperature value to convert",
            ),
            "from_unit" => Dict{String, Any}(
                "type" => "string",
                "description" => "The unit to convert from: 'celsius' or 'fahrenheit'",
                "enum" => ["celsius", "fahrenheit"],
            ),
        ),
        "required" => ["value", "from_unit"],
    ),
)
println(manual_tool)
```

## Creating an Agent with Tools

Pass a vector of tools when constructing the agent:

``` julia
client = OllamaChatClient(model = "qwen3:8b")

agent = Agent(
    name = "ToolAgent",
    instructions = "You are a helpful assistant. Use the available tools to answer questions accurately. Be concise.",
    client = client,
    tools = [get_weather, calculate, get_population],
)
println(agent)
```

## Running the Agent with Tool Use

When the agent receives a question it cannot answer from its training
data alone, it will invoke the appropriate tool:

``` julia
response = run_agent(agent, "What's the weather like in Tokyo?")
println(response.text)
```

**Expected output:**

    The weather in Tokyo is sunny with a high of 24°C.

## Inspecting the Tool-Call Cycle

The `response.messages` vector reveals the full conversation, including
any tool calls the model made and the results that were fed back:

``` julia
response = run_agent(agent, "What is the population of Brazil?")

for msg in response.messages
    println("--- ", msg.role, " ---")
    for content in msg.contents
        if AgentFramework.is_text(content)
            println("  Text: ", content.text)
        elseif AgentFramework.is_function_call(content)
            println("  Tool call: ", content.name, "(", content.arguments, ")")
        elseif AgentFramework.is_function_result(content)
            println("  Tool result: ", content.result)
        end
    end
end
```

**Expected output:**

    --- assistant ---
      Tool call: get_population({"country":"Brazil"})
    --- tool ---
      Tool result: Brazil has approximately 214 million people.
    --- assistant ---
      Text: Brazil has approximately 214 million people.

## Multiple Tool Calls

The model can chain multiple tools in a single interaction. Here we ask
a question that requires both a calculation and a population lookup:

``` julia
response = run_agent(agent, "What is the combined population of France and Germany?")
println(response.text)
```

**Expected output:**

    The combined population of France and Germany is approximately 152 million people.

## Summary

- **`@tool`** converts a Julia function into a `FunctionTool` with
  automatic JSON Schema generation from type annotations.
- **`FunctionTool(...)`** gives full manual control over the schema when
  needed.
- Pass tools to `Agent(... tools=[...])` and the framework handles the
  call → execute → feed-back loop automatically.
- Inspect `response.messages` to see every step of the tool-calling
  cycle.

Next, see [03 — Multi-Turn
Conversations](../03_multi_turn/03_multi_turn.qmd) to learn how agents
maintain context across multiple exchanges.
