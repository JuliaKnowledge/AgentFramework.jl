# Declarative Agents
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Why Declarative?](#why-declarative)
- [Agent from YAML](#agent-from-yaml)
  - [Supported Top-Level Fields](#supported-top-level-fields)
- [Agent from File](#agent-from-file)
- [Registering Tools and Clients](#registering-tools-and-clients)
  - [Managing the Registry](#managing-the-registry)
- [Workflow from YAML](#workflow-from-yaml)
  - [Workflow Edge Kinds](#workflow-edge-kinds)
- [Round-Trip Serialization](#round-trip-serialization)
- [Environment Variable
  Substitution](#environment-variable-substitution)
- [Putting It All Together](#putting-it-all-together)
- [Summary](#summary)

## Overview

So far every agent has been built imperatively — constructing types,
wiring tools, and calling `run_agent` from Julia code. **Declarative
agents** flip this around: you describe the agent (or an entire
workflow) in a YAML or JSON file and let the framework materialize it at
runtime. This makes agent definitions portable, version-controllable,
and editable by non-programmers.

By the end you will know how to:

1.  Define an agent purely in YAML and load it with `agent_from_yaml`.
2.  Load agent definitions from files with `agent_from_file`.
3.  Register tools and clients so declarative definitions can reference
    them by name.
4.  Build a multi-step workflow from a YAML specification.
5.  Round-trip between live objects and their YAML/JSON representations.
6.  Use environment-variable substitution in declarative configs.

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

## Why Declarative?

Imperative agent construction is flexible but couples every detail to
Julia source code. Declarative definitions provide several benefits:

| Benefit | Description |
|----|----|
| **Portability** | Share agent definitions across Julia, Python, and C# runtimes |
| **Version control** | YAML diffs are easy to review |
| **No-code editing** | Non-developers can tweak instructions and parameters |
| **Reproducibility** | A single file fully specifies agent behaviour |
| **Composition** | Tools, clients, and handlers are referenced by name |

## Agent from YAML

The simplest declarative agent is a `Prompt` kind — an agent defined
entirely by its instructions and model options. Pass the YAML string to
`agent_from_yaml` along with a default chat client:

``` julia
yaml_str = """
kind: Prompt
name: WeatherHelper
description: A weather assistant that answers concisely
instructions: You answer weather questions concisely.
options:
  temperature: 0.7
  max_tokens: 500
"""

client = OllamaChatClient(model="qwen3:8b")
agent = agent_from_yaml(yaml_str; default_client=client)
```

The returned `agent` is a regular `Agent` — you interact with it the
same way as any imperatively constructed agent:

``` julia
response = run_agent(agent, "Is it sunny in Paris today?")
println(response.text)
```

### Supported Top-Level Fields

| Field | Type | Description |
|----|----|----|
| `kind` | `String` | Agent kind — `"Prompt"` for instruction-based agents |
| `name` | `String` | Agent name (used in logs and multi-agent setups) |
| `description` | `String` | Human-readable description |
| `instructions` | `String` | System prompt |
| `options` | `Dict` | Model options (`temperature`, `max_tokens`, `top_p`, etc.) |
| `tools` | `Vector` | List of tool names (must be registered) |
| `client` | `String` | Name of a registered client |

## Agent from File

For production use, store definitions in `.yaml` files and load them
with `agent_from_file`:

``` julia
agent = agent_from_file("agents/weather.yaml"; default_client=client)
```

The file `agents/weather.yaml` contains the same YAML shown above. The
function reads the file, parses it, and delegates to `agent_from_yaml`.

You can also load JSON files — the loader detects the format
automatically:

``` julia
agent = agent_from_file("agents/weather.json"; default_client=client)
```

## Registering Tools and Clients

When a YAML definition references tools or clients **by name**, those
names must be resolved at load time. Use the global registry functions
to bind names to live objects before loading:

``` julia
# Define a tool the usual way.
@tool function get_weather(location::String)
    "Get the current weather for a location."
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    temp = rand(10:30)
    return "The weather in $location is $(rand(conditions)) at $(temp)°C."
end

# Register the tool and client under string names.
register_client!("ollama", OllamaChatClient(model="qwen3:8b"))
register_tool!("get_weather", get_weather)
```

Now the YAML can reference them without passing objects directly:

``` julia
yaml_str = """
kind: Prompt
name: WeatherAgent
description: Uses tools to answer weather questions
instructions: Use the available tools to answer weather questions accurately.
client: ollama
tools:
  - get_weather
"""

agent = agent_from_yaml(yaml_str)
response = run_agent(agent, "What's the weather in London?")
println(response.text)
```

### Managing the Registry

``` julia
# List registered names.
list_clients()    # ["ollama"]
list_tools()      # ["get_weather"]

# Remove entries when no longer needed.
unregister_client!("ollama")
unregister_tool!("get_weather")

# Clear everything (useful in tests).
clear_registry!()
```

## Workflow from YAML

Declarative definitions extend beyond single agents to multi-step
workflows. A workflow YAML specifies executors, edges, and the
start/output nodes:

``` julia
yaml_str = """
name: ResearchPipeline
start: researcher
outputs:
  - summarizer
executors:
  - id: researcher
    description: Finds information on the given topic
  - id: summarizer
    description: Summarizes the research findings
edges:
  - kind: direct
    source: researcher
    target: summarizer
"""
```

Executors in the YAML are resolved by `id` — you register handler
functions before loading:

``` julia
function researcher_handler(data, ctx)
    result = run_agent(research_agent, data)
    send_message(ctx, result.text)
end

function summarizer_handler(data, ctx)
    result = run_agent(summary_agent, data)
    yield_output(ctx, result.text)
end

register_handler!("researcher", researcher_handler)
register_handler!("summarizer", summarizer_handler)

workflow = workflow_from_yaml(yaml_str)
result = run_workflow(workflow, "Recent advances in quantum computing")
println(result.outputs[1])
```

### Workflow Edge Kinds

| Kind          | Description                              |
|---------------|------------------------------------------|
| `direct`      | One-to-one message passing               |
| `fan_out`     | Broadcast to multiple targets            |
| `fan_in`      | Aggregate messages from multiple sources |
| `conditional` | Route based on a condition expression    |

A more complex example with fan-out and fan-in:

``` julia
yaml_str = """
name: ParallelAnalysis
start: splitter
outputs:
  - aggregator
executors:
  - id: splitter
    description: Splits the query into sub-tasks
  - id: analyst_a
    description: Analyses from perspective A
  - id: analyst_b
    description: Analyses from perspective B
  - id: aggregator
    description: Combines the analyses
edges:
  - kind: fan_out
    source: splitter
    targets:
      - analyst_a
      - analyst_b
  - kind: fan_in
    sources:
      - analyst_a
      - analyst_b
    target: aggregator
"""

workflow = workflow_from_yaml(yaml_str)
```

## Round-Trip Serialization

Live agents and workflows can be serialized back to YAML or JSON,
enabling inspection, modification, and re-loading:

``` julia
# Agent → YAML → Agent
yaml = agent_to_yaml(agent)
println(yaml)

agent2 = agent_from_yaml(yaml; default_client=client)
```

    kind: Prompt
    name: WeatherAgent
    description: Uses tools to answer weather questions
    instructions: Use the available tools to answer weather questions accurately.
    client: ollama
    tools:
      - get_weather

The same works for workflows and for JSON:

``` julia
# Workflow → JSON → Workflow
json_str = workflow_to_json(workflow)
workflow2 = workflow_from_json(json_str; handlers=Dict(
    "researcher" => researcher_handler,
    "summarizer" => summarizer_handler,
))

# Workflow → YAML
yaml = workflow_to_yaml(workflow)
```

Round-trip fidelity means you can programmatically build an agent,
export it, hand-edit the YAML, and re-import without loss.

## Environment Variable Substitution

Declarative configs often need secrets or deployment-specific values
that should not be hard-coded. Use `${VAR_NAME}` syntax for environment
variable references:

``` julia
yaml_str = """
kind: Prompt
name: CloudAgent
instructions: You are a helpful assistant.
model:
  type: OpenAIChatClient
  api_key: \${OPENAI_API_KEY}
  model: gpt-4
options:
  temperature: 0.5
"""

# Substitution happens automatically during loading.
agent = agent_from_yaml(yaml_str)
```

At load time, `${OPENAI_API_KEY}` is replaced with the value of
`ENV["OPENAI_API_KEY"]`. If the variable is not set, loading raises an
`AgentFrameworkException` with a clear message.

You can also provide explicit overrides via the `env` keyword:

``` julia
agent = agent_from_yaml(yaml_str; env=Dict(
    "OPENAI_API_KEY" => "sk-test-...",
))
```

## Putting It All Together

Here is a complete example that defines a tool, registers it, and loads
a declarative agent that uses it:

``` julia
using AgentFramework

# 1. Define and register a tool.
@tool function calculate(expression::String)
    "Evaluate a mathematical expression and return the result."
    return string(eval(Meta.parse(expression)))
end
register_tool!("calculate", calculate)

# 2. Register a client.
register_client!("ollama", OllamaChatClient(model="qwen3:8b"))

# 3. Load the agent from YAML.
agent = agent_from_yaml("""
kind: Prompt
name: MathHelper
description: Answers maths questions using a calculator tool
instructions: |
  You are a helpful maths assistant.
  Use the calculate tool to evaluate expressions.
  Always show your working.
client: ollama
tools:
  - calculate
options:
  temperature: 0.0
""")

# 4. Run the agent.
response = run_agent(agent, "What is the square root of 144 plus 3 squared?")
println(response.text)
```

## Summary

| Concept | Description |
|----|----|
| `agent_from_yaml` | Parse a YAML string into an `Agent` |
| `agent_from_file` | Load an agent definition from a `.yaml` or `.json` file |
| `agent_to_yaml` | Serialize a live `Agent` back to YAML |
| `register_client!` | Bind a string name to a chat client in the global registry |
| `register_tool!` | Bind a string name to a tool in the global registry |
| `register_handler!` | Bind a string name to a workflow handler function |
| `workflow_from_yaml` | Parse a YAML string into a `Workflow` |
| `workflow_to_yaml` / `workflow_to_json` | Serialize a workflow to YAML or JSON |
| `workflow_from_json` | Parse a JSON string into a `Workflow` |
| `${VAR}` substitution | Environment variable expansion in declarative configs |
| `clear_registry!` | Remove all registered clients, tools, and handlers |

Key takeaways:

1.  Declarative definitions decouple **what** an agent does from **how**
    it is instantiated.
2.  YAML and JSON formats are interchangeable — pick whichever suits
    your workflow.
3.  The **registry** (`register_client!`, `register_tool!`,
    `register_handler!`) bridges named references in YAML to live Julia
    objects.
4.  **Round-trip serialization** lets you inspect, modify, and re-import
    definitions without loss.
5.  **Environment variable substitution** keeps secrets out of config
    files.

Next, see [20 — Multimodal Agents](../20_multimodal/20_multimodal.qmd)
to work with images, audio, and other media types in agent
conversations.
