# Hello Agent
AgentFramework.jl

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Creating a Chat Client](#creating-a-chat-client)
- [Building an Agent](#building-an-agent)
- [Running the Agent](#running-the-agent)
- [Inspecting the Response](#inspecting-the-response)
- [Summary](#summary)

## Overview

This vignette demonstrates the simplest possible agent: connect to a
local LLM, send a prompt, and receive a response. By the end you will
know how to:

1.  Create an `OllamaChatClient` pointing at a local Ollama instance.
2.  Build an `Agent` with a name, instructions, and client.
3.  Call `run_agent` and inspect the result.

## Prerequisites

You need [Ollama](https://ollama.com) running locally with the
`qwen3:8b` model pulled:

``` bash
ollama pull qwen3:8b
```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..",".."))
using AgentFramework
```

## Creating a Chat Client

The `OllamaChatClient` wraps Ollama’s OpenAI-compatible API. By default
it connects to `http://localhost:11434`:

``` julia
client = OllamaChatClient(model = "qwen3:8b")
```

    OllamaChatClient("qwen3:8b")

## Building an Agent

An `Agent` combines a chat client with a name and system instructions.
The instructions are prepended as a system message on every call:

``` julia
agent = Agent(
    name = "HelloAgent",
    instructions = "You are a friendly assistant. Keep your answers brief.",
    client = client,
)
```

    Agent("HelloAgent", 0 tools)

## Running the Agent

`run_agent` sends a user prompt through the agent and returns an
`AgentResponse`. You can access the generated text via the `.text`
property:

``` julia
response = run_agent(agent, "What is the capital of France?")
println(response.text)
```

**Expected output:**

    The capital of France is Paris.

## Inspecting the Response

The `AgentResponse` object carries more than just text. You can inspect
the individual messages, the model used, the finish reason, and token
usage:

``` julia
# The response contains one or more assistant messages
for msg in response.messages
    println("Role: ", msg.role)
    println("Text: ", msg.text)
end

# Check which model produced the response
println("Model: ", response.model_id)

# Check the finish reason
println("Finish reason: ", response.finish_reason)
```

**Expected output:**

    Role: assistant
    Text: The capital of France is Paris.
    Model: qwen3:8b
    Finish reason: STOP

## Summary

You created a minimal agent in three steps:

1.  **Client** — `OllamaChatClient(model = "qwen3:8b")` connects to
    Ollama.
2.  **Agent** — `Agent(name=..., instructions=..., client=...)` wraps
    the client with a persona.
3.  **Run** — `run_agent(agent, prompt)` sends the prompt and returns an
    `AgentResponse` whose `.text` property holds the generated answer.

Next, see [02 — Tools](../02_tools/02_tools.qmd) to give your agent the
ability to call functions.
