# Telemetry
AgentFramework.jl

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Why Telemetry?](#why-telemetry)
- [TelemetrySpan Basics](#telemetryspan-basics)
  - [Adding Events](#adding-events)
  - [Finishing a Span](#finishing-a-span)
- [GenAI Semantic Conventions](#genai-semantic-conventions)
- [Telemetry Backends](#telemetry-backends)
  - [InMemoryTelemetryBackend](#inmemorytelemetrybackend)
  - [LoggingTelemetryBackend](#loggingtelemetrybackend)
- [Instrumenting an Agent](#instrumenting-an-agent)
  - [Running the Instrumented Agent](#running-the-instrumented-agent)
  - [Inspecting Collected Spans](#inspecting-collected-spans)
- [Complete Observability Example](#complete-observability-example)
- [Summary](#summary)

## Overview

When agents call LLMs, invoke tools, and orchestrate workflows,
understanding what happened (and how long it took) is essential for
debugging, performance tuning, and cost tracking. This vignette covers
the built-in telemetry system. By the end you will know how to:

1.  Create and manage `TelemetrySpan` objects.
2.  Attach events and attributes to spans.
3.  Use `InMemoryTelemetryBackend` for testing and debugging.
4.  Use `LoggingTelemetryBackend` for production logging.
5.  Instrument an agent with `instrument!` for automatic tracing.
6.  Inspect collected spans for operation names, durations, and token
    usage.

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
using Dates
```

## Why Telemetry?

Agent applications involve multiple layers of processing:

- **Agent middleware** — authorization, logging, pre/post-processing.
- **Chat calls** — the actual LLM API request/response.
- **Tool invocations** — function calls triggered by the model.

Without telemetry, debugging “why did the agent give that answer?”
requires manual logging. With spans, you get structured traces that show
exactly what happened, when, and how long each step took.

## TelemetrySpan Basics

A `TelemetrySpan` represents a unit of work. Create one with a name, add
attributes and events, then finish it:

``` julia
span = TelemetrySpan(name = "my_operation", kind = :client)
span.attributes["custom.key"] = "some_value"
println("Span: $(span.name), status: $(span.status)")
```

    Span: my_operation, status: unset

### Adding Events

Events are timestamped markers within a span — useful for recording
checkpoints or exceptions:

``` julia
add_event!(span, "checkpoint_reached";
    attributes = Dict{String, Any}("step" => 1))
add_event!(span, "data_processed";
    attributes = Dict{String, Any}("records" => 42))
println("Events: ", length(span.events))
```

    Events: 2

### Finishing a Span

Call `finish_span!` to record the end time and set the final status:

``` julia
finish_span!(span; status = :ok)
println("Duration: $(duration_ms(span)) ms")
println("Status: $(span.status)")
```

    Duration: 219 ms
    Status: ok

## GenAI Semantic Conventions

AgentFramework follows [OpenTelemetry Semantic Conventions for
GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for
attribute naming. Key constants live in the `GenAIConventions` module:

``` julia
println("Model attribute: ", GenAIConventions.REQUEST_MODEL)
println("Input tokens:    ", GenAIConventions.USAGE_INPUT_TOKENS)
println("Output tokens:   ", GenAIConventions.USAGE_OUTPUT_TOKENS)
println("Agent name:      ", GenAIConventions.AGENT_NAME)
println("Tool name:       ", GenAIConventions.TOOL_NAME)
```

    Model attribute: gen_ai.request.model
    Input tokens:    gen_ai.usage.input_tokens
    Output tokens:   gen_ai.usage.output_tokens
    Agent name:      gen_ai.agent.name
    Tool name:       gen_ai.tool.name

## Telemetry Backends

### InMemoryTelemetryBackend

Stores all spans in memory — perfect for testing and debugging:

``` julia
backend = InMemoryTelemetryBackend()
```

    InMemoryTelemetryBackend(TelemetrySpan[], ReentrantLock())

Record a span:

``` julia
test_span = TelemetrySpan(name = "test.operation")
test_span.attributes[GenAIConventions.REQUEST_MODEL] = "qwen3:8b"
finish_span!(test_span; status = :ok)
record_span!(backend, test_span)

spans = get_spans(backend)
println("Recorded spans: ", length(spans))
println("First span: ", spans[1].name)
```

    Recorded spans: 1
    First span: test.operation

Clear all spans when you’re done:

``` julia
clear_spans!(backend)
println("Spans after clear: ", length(get_spans(backend)))
```

    Spans after clear: 0

### LoggingTelemetryBackend

Emits structured log messages for each span — suitable for production
where you want spans in your log aggregation system:

``` julia
log_backend = LoggingTelemetryBackend()
```

    LoggingTelemetryBackend(Info)

``` julia
log_span = TelemetrySpan(name = "chat.completion", kind = :client)
log_span.attributes[GenAIConventions.REQUEST_MODEL] = "qwen3:8b"
log_span.attributes[GenAIConventions.USAGE_INPUT_TOKENS] = 25
log_span.attributes[GenAIConventions.USAGE_OUTPUT_TOKENS] = 50
finish_span!(log_span; status = :ok)
record_span!(log_backend, log_span)
```

    ┌ Info: chat.completion
    │   span_id = "fd2fd06e-7602-4af2-b6c7-587554f52372"
    │   duration_ms = 0
    │   status = :ok
    └   attributes = "gen_ai.usage.input_tokens=25, gen_ai.usage.output_tokens=50, gen_ai.request.model=qwen3:8b"

## Instrumenting an Agent

The `instrument!` function adds telemetry middleware to all three agent
pipeline layers (agent, chat, function) in a single call:

``` julia
client = OllamaChatClient(model = "qwen3:8b")
backend = InMemoryTelemetryBackend()

agent = Agent(
    name = "TracedAgent",
    instructions = "You are a helpful assistant. Keep answers brief.",
    client = client,
)

# One call instruments all three middleware layers
instrument!(agent, backend)
```

### Running the Instrumented Agent

``` julia
response = run_agent(agent, "What is 2 + 2?")
println("Answer: ", response.text)
```

**Expected output:**

    Answer: 2 + 2 = 4.

### Inspecting Collected Spans

After running the agent, the backend contains spans for each pipeline
layer:

``` julia
spans = get_spans(backend)
println("Total spans: ", length(spans))

for s in spans
    dur = duration_ms(s)
    model = get(s.attributes, GenAIConventions.REQUEST_MODEL, "—")
    println("  $(s.name) [$(s.kind)] — $(dur)ms, model=$model, status=$(s.status)")

    # Show token usage if available
    input_tok = get(s.attributes, GenAIConventions.USAGE_INPUT_TOKENS, nothing)
    output_tok = get(s.attributes, GenAIConventions.USAGE_OUTPUT_TOKENS, nothing)
    if input_tok !== nothing
        println("    Tokens: $input_tok in / $output_tok out")
    end

    # Show events
    for evt in s.events
        println("    Event: $(evt["name"]) at $(evt["time"])")
    end
end
```

**Expected output:**

    Total spans: 2
      agent.run [internal] — 1250ms, model=—, status=ok
        Tokens: — in / — out
      chat.completion [client] — 1200ms, model=qwen3:8b, status=ok
        Tokens: 25 in / 12 out

The exact number of spans depends on whether the agent calls tools. Each
tool invocation adds a `tool.invoke` span.

## Complete Observability Example

Putting it all together — an agent with tools, instrumented for full
tracing:

``` julia
client = OllamaChatClient(model = "qwen3:8b")
backend = InMemoryTelemetryBackend()

add_numbers = AgentTool(
    name = "add",
    description = "Add two numbers",
    parameters = Dict(
        "a" => Dict("type" => "number", "description" => "First number"),
        "b" => Dict("type" => "number", "description" => "Second number"),
    ),
    handler = (args) -> string(args["a"] + args["b"]),
)

agent = Agent(
    name = "MathAgent",
    instructions = "Use the add tool for arithmetic.",
    client = client,
    tools = [add_numbers],
)

instrument!(agent, backend)

response = run_agent(agent, "What is 17 + 25?")
println("Answer: ", response.text)
println()

# Inspect the full trace
for s in get_spans(backend)
    dur = duration_ms(s)
    println("[$(s.kind)] $(s.name) — $(dur)ms ($(s.status))")
    for (k, v) in s.attributes
        println("    $k = $v")
    end
end
```

**Expected output:**

    Answer: 17 + 25 = 42.

    [internal] agent.run — 2100ms (ok)
        gen_ai.agent.name = MathAgent
        gen_ai.operation.name = chat
        message_count = 1
        gen_ai.response.model = qwen3:8b
    [client] chat.completion — 800ms (ok)
        gen_ai.operation.name = chat
        gen_ai.request.model = qwen3:8b
        message_count = 1
    [internal] tool.invoke — 1ms (ok)
        gen_ai.tool.name = add
        gen_ai.tool.call_id = call_abc123
    [client] chat.completion — 600ms (ok)
        gen_ai.operation.name = chat
        gen_ai.request.model = qwen3:8b
        message_count = 4

## Summary

| Component                  | Purpose                                      |
|----------------------------|----------------------------------------------|
| `TelemetrySpan`            | Unit of work with timing, attributes, events |
| `add_event!`               | Attach timestamped events to a span          |
| `finish_span!`             | Mark span complete with status               |
| `InMemoryTelemetryBackend` | In-memory store for testing                  |
| `LoggingTelemetryBackend`  | Structured log output                        |
| `instrument!`              | One-call instrumentation of all agent layers |
| `GenAIConventions`         | Standard attribute names for GenAI spans     |

The telemetry system is designed to be lightweight and composable.
Implement `AbstractTelemetryBackend` with a custom `record_span!` method
to integrate with OpenTelemetry, Datadog, or any observability platform.
