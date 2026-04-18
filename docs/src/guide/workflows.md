# Workflows

Workflows enable multi-agent orchestration by connecting executors (agents or functions) in a directed acyclic graph (DAG). Messages flow between executors along edges, with the engine managing execution order, message routing, and state.

## Overview

The workflow system is inspired by the Pregel computing model: execution proceeds in **supersteps**, where all ready executors run concurrently within a superstep, and messages are delivered at superstep boundaries.

```
           ┌─────────┐
    input──▶│ Planner │
           └────┬────┘
                │
        ┌───────┼───────┐    (fan-out)
        ▼       ▼       ▼
    ┌───────┐┌───────┐┌───────┐
    │Writer ││Coder  ││Review │
    └───┬───┘└───┬───┘└───┬───┘
        │        │        │
        └───────┬┘────────┘    (fan-in)
                ▼
          ┌───────────┐
          │ Assembler │──▶ output
          └───────────┘
```

## WorkflowBuilder

[`WorkflowBuilder`](@ref) provides a fluent API for constructing workflow DAGs:

```julia
using AgentFramework

# Define executors (processing nodes)
upper = ExecutorSpec(
    id = "upper",
    handler = (msg, ctx) -> send_message(ctx, uppercase(string(msg))),
)

reverse_exec = ExecutorSpec(
    id = "reverse",
    handler = (msg, ctx) -> yield_output(ctx, reverse(string(msg))),
)

# Build the workflow
workflow = WorkflowBuilder(name="TextPipeline", start=upper) |>
    b -> add_executor(b, reverse_exec) |>
    b -> add_edge(b, "upper", "reverse") |>
    b -> add_output(b, "reverse") |>
    build
```

### Builder Methods

| Method | Description |
|:-------|:------------|
| `WorkflowBuilder(; name, start)` | Create a builder with a start executor |
| [`add_executor`](@ref) | Add an executor node |
| [`add_edge`](@ref) | Add a direct edge between two executors |
| [`add_fan_out`](@ref) | Add a fan-out edge (one-to-many broadcast) |
| [`add_fan_in`](@ref) | Add a fan-in edge (many-to-one aggregation) |
| [`add_switch`](@ref) | Add conditional routing (switch/case) |
| [`add_output`](@ref) | Mark an executor as an output node |
| [`build`](@ref) | Validate and produce the final [`Workflow`](@ref) |

## Executor Nodes

An [`ExecutorSpec`](@ref) defines a processing node in the workflow:

```julia
spec = ExecutorSpec(
    id = "summarizer",                   # Unique identifier
    description = "Summarize input",     # Human-readable description
    input_types = DataType[String],      # Expected input types
    output_types = DataType[String],     # Types sent via send_message
    yield_types = DataType[String],      # Types yielded as workflow output
    handler = my_handler_function,       # (data, context) -> nothing
)
```

### Handler Functions

A handler function receives the incoming message data and a [`WorkflowContext`](@ref):

```julia
function my_handler(data, ctx::WorkflowContext)
    # Process input
    result = process(data)

    # Send to downstream executors (follows edge routing)
    send_message(ctx, result)

    # Or yield as workflow output
    yield_output(ctx, result)

    # Access shared state
    count = get_state(ctx, "count", 0)
    set_state!(ctx, "count", count + 1)
end
```

### WorkflowContext API

| Function | Description |
|:---------|:------------|
| [`send_message`](@ref) | Send data to downstream executors |
| [`yield_output`](@ref) | Emit data as workflow output |
| [`get_state`](@ref) | Read from shared workflow state |
| [`set_state!`](@ref) | Write to shared workflow state |
| [`request_info`](@ref) | Pause for human input (human-in-the-loop) |

### Agent Executors

Use [`agent_executor`](@ref) to wrap an [`Agent`](@ref) as an executor node:

```julia
client = OllamaChatClient(model="qwen3:8b")

planner_agent = Agent(
    name = "Planner",
    instructions = "Break tasks into steps.",
    client = client,
)

planner = agent_executor("planner", planner_agent)
writer = @executor "writer" function(msg::String, ctx)
    yield_output(ctx, msg)
end

workflow = WorkflowBuilder(name="PlanAndExecute", start=planner) |>
    b -> add_executor(b, writer) |>
    b -> add_edge(b, planner.id, writer.id) |>
    build
```

### The @executor Macro

The [`@executor`](@ref) macro provides concise executor definitions:

```julia
upper = @executor "upper" function(msg::String, ctx)
    send_message(ctx, uppercase(msg))
end

formatter = @executor "formatter" "Format data as JSON" function(data::Dict{String, Any}, ctx)
    result = JSON3.write(data)
    yield_output(ctx, result)
end
```

## Edge Types

Edges define how messages flow between executors.

### Direct Edges

One-to-one routing between executors:

```julia
workflow = WorkflowBuilder(name="Pipeline", start=exec_a) |>
    b -> add_executor(b, exec_b) |>
    b -> add_edge(b, "a", "b") |>
    build
```

With optional conditions:

```julia
add_edge(b, "router", "handler_a",
    condition = data -> data isa String,
    condition_name = "is_string",
)
```

### Fan-Out Edges

One source broadcasts to multiple targets:

```julia
add_fan_out(b, "planner", ["writer", "coder", "reviewer"])
```

With selective routing:

```julia
add_fan_out(b, "router", ["fast", "medium", "slow"],
    selection_func = (data, targets) -> data["priority"] == "high" ? ["fast"] : targets,
)
```

### Fan-In Edges

Multiple sources converge to one target:

```julia
add_fan_in(b, ["writer", "coder", "reviewer"], "assembler")
```

### Switch Edges

Conditional routing based on data:

```julia
add_switch(b, "classifier", [
    (data -> get(data, "sentiment", 0.0) > 0.5) => "celebrate",
    (data -> get(data, "sentiment", 0.0) <= 0.5) => "console",
])
```

## Specialized Builders

AgentFramework.jl provides high-level builders for common orchestration patterns.

### SequentialBuilder

Chain executors in a linear pipeline:

```julia
workflow = SequentialBuilder(
    name = "Pipeline",
    participants = [step1, step2, step3],
) |> build
```

### ConcurrentBuilder

Run executors in parallel and aggregate results:

```julia
workflow = ConcurrentBuilder(
    name = "ParallelResearch",
    participants = [researcher1, researcher2, researcher3],
    aggregator = (results, ctx) -> yield_output(ctx, merge_results(results)),
) |> build
```

### GroupChatBuilder

Multi-agent round-robin or selector-based group chat:

```julia
workflow = GroupChatBuilder(
    name = "TeamDiscussion",
    participants = [agent_a, agent_b, agent_c],
) |> build
```

Variants include [`RoundRobinGroupChat`](@ref) and [`SelectorGroupChat`](@ref).

### MagenticBuilder

Implements the Magentic-One orchestration pattern with a manager that plans, selects participants, and tracks progress:

```julia
workflow = MagenticBuilder(
    name = "MagenticTeam",
    participants = [coder, reviewer, tester],
) |> build
```

The Magentic pattern includes:
- [`MagenticTaskLedger`](@ref) — Facts and plan tracking
- [`MagenticProgressLedger`](@ref) — Per-participant progress
- [`MagenticContext`](@ref) — Full orchestration state

## Running Workflows

Use [`run_workflow`](@ref) to execute a workflow with an initial input:

```julia
result = run_workflow(workflow, "hello world")
```

The returned [`WorkflowRunResult`](@ref) contains:

```julia
# Get workflow outputs
outputs = get_outputs(result)

# Get the final state
state = get_final_state(result)

# Get all events for observability
events = result.events

# Check for human-in-the-loop requests
info_requests = get_request_info_events(result)
```

### Workflow Events

The engine emits [`WorkflowEvent`](@ref) items for observability:

| Event | Description |
|:------|:------------|
| `EVT_STARTED` | Workflow execution started |
| `EVT_SUPERSTEP_STARTED` | Superstep began |
| `EVT_EXECUTOR_INVOKED` | Executor handler called |
| `EVT_EXECUTOR_COMPLETED` | Executor finished successfully |
| `EVT_EXECUTOR_FAILED` | Executor threw an error |
| `EVT_SUPERSTEP_COMPLETED` | Superstep ended |
| `EVT_OUTPUT` | Workflow output yielded |
| `EVT_REQUEST_INFO` | Human input requested |
| `EVT_FAILED` | Workflow failed |

### Streaming Events

Use event channels for real-time monitoring:

```julia
events_channel = run_workflow(workflow, "input"; stream = true)

# Process events as they arrive
for event in events_channel
    if event.type == EVT_OUTPUT
        println("Output: ", event.data)
    end
end
```

## Checkpointing

Checkpointing enables saving and restoring workflow state for resumable execution.

### WorkflowCheckpoint

A [`WorkflowCheckpoint`](@ref) captures the full workflow state at a superstep boundary:

```julia
checkpoint = WorkflowCheckpoint(
    workflow_name = "MyWorkflow",
    iteration = 5,
    state = Dict{String, Any}("progress" => 50),
    messages = Dict{String, Vector{WorkflowMessage}}(),
)
```

### Checkpoint Storage

#### InMemoryCheckpointStorage

```julia
storage = InMemoryCheckpointStorage()
```

#### FileCheckpointStorage

```julia
storage = FileCheckpointStorage(joinpath(storage_dir, "checkpoints"))
```

### Using Checkpoints

Attach checkpoint storage when building a workflow:

```julia
step1_exec = agent_executor("step1", step1)
step2_exec = agent_executor("step2", step2)

workflow = WorkflowBuilder(
    name = "ResumableWorkflow",
    start = step1_exec,
    checkpoint_storage = FileCheckpointStorage(joinpath(storage_dir, "checkpoints")),
) |>
    b -> add_executor(b, step2_exec) |>
    b -> add_edge(b, step1_exec.id, step2_exec.id) |>
    build
```

## Human-in-the-Loop

Executors can pause and request information from a human:

```julia
function review_handler(data, ctx::WorkflowContext)
    # Request human approval
    request_info(ctx, "Please review this output: $(data)")

    # The workflow pauses here until input is provided
    # When resumed, execution continues
    yield_output(ctx, data)
end
```

Check for pending requests:

```julia
result = run_workflow(workflow, input)
requests = get_request_info_events(result)
if !isempty(requests)
    println("Human input needed: ", requests[1].data)
end
```

## Next Steps

- [Agents](agents.md) — Create agents to use as workflow executors
- [Streaming](streaming.md) — Stream workflow events in real time
- [Advanced Topics](@ref) — Declarative workflows from YAML/JSON
