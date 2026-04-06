# Human-in-the-Loop Workflows
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [How `request_info` Works](#how-request_info-works)
- [Basic `request_info` in an
  Executor](#basic-request_info-in-an-executor)
- [Building a Workflow with HITL](#building-a-workflow-with-hitl)
- [Running the Workflow with Human
  Interaction](#running-the-workflow-with-human-interaction)
  - [Step 1: Initial Run](#step-1-initial-run)
  - [Step 2: Inspect Pending Requests](#step-2-inspect-pending-requests)
  - [Step 3: Resume with Human
    Response](#step-3-resume-with-human-response)
- [Function Tool Approval Pattern](#function-tool-approval-pattern)
- [Workflow States During HITL](#workflow-states-during-hitl)
- [Full Example: Multi-Step Approval
  Pipeline](#full-example-multi-step-approval-pipeline)
- [Summary](#summary)

## Overview

Some workflows need a human to approve, correct, or supply information
before execution can continue. AgentFramework supports
**human-in-the-loop** (HITL) patterns through the `request_info`
mechanism: an executor pauses the workflow, emits a request event, and
resumes only when a response arrives.

By the end you will know how to:

1.  Use `request_info` inside an executor to pause a workflow.
2.  Inspect `EVT_REQUEST_INFO` events to identify pending requests.
3.  Resume a paused workflow by supplying responses.
4.  Use the `FunctionApprovalRequest` content type for tool approval
    gates.
5.  Understand workflow state transitions during HITL interactions.

<!-- -->

    Executor runs ──► request_info(ctx, data) ──► Workflow pauses (IDLE)
                                                         │
                                                  Human provides input
                                                         │
                                                  resume with responses
                                                         │
                                                  Executor continues

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

## How `request_info` Works

When an executor calls `request_info(ctx, data)`, the workflow engine:

1.  Emits a `EVT_REQUEST_INFO` event containing the request data and a
    unique `request_id`.
2.  Transitions the workflow to the **`WF_IDLE`** state — execution
    stops.
3.  The caller inspects the result, finds pending requests, and collects
    human responses.
4.  The caller resumes the workflow by passing a `responses` dictionary
    mapping `request_id` values to response data.

The executor picks up exactly where it left off, with the response
available as the return value of `request_info`.

## Basic `request_info` in an Executor

Here is an executor that drafts content, then pauses to request human
review:

``` julia
function review_handler(messages, ctx)
    draft = first(messages)

    # Pause workflow and request human feedback
    feedback = request_info(ctx;
        request_data = Dict(
            "prompt" => "Please review this draft: $(first(draft, 100))...",
            "type" => "review",
        ),
        response_type = String,
    )

    # When resumed, feedback contains the human's response
    if lowercase(strip(feedback)) == "approve"
        yield_output(ctx, draft)
    else
        yield_output(ctx, "Revision needed: $feedback")
    end
end
```

    review_handler (generic function with 1 method)

The key points:

- **`request_data`** — arbitrary data sent to the human (prompt,
  context, etc.).
- **`response_type`** — the expected Julia type of the response.
- **Return value** — after resume, `request_info` returns the human’s
  response.

## Building a Workflow with HITL

We wire a writer agent into a review step that requires human approval:

``` julia
client = OllamaChatClient(model = "qwen3:8b")

writer_agent = Agent(
    name = "Writer",
    instructions = "Write a short paragraph about the given topic.",
    client = client,
)

writer_exec = agent_executor("writer", writer_agent, forward_response = true)

reviewer_exec = ExecutorSpec(
    id = "reviewer",
    description = "Human review gate",
    handler = review_handler,
)

workflow = WorkflowBuilder(name = "ReviewWorkflow", start = writer_exec) |>
    b -> add_executor(b, reviewer_exec) |>
    b -> add_edge(b, "writer", "reviewer") |>
    b -> add_output(b, "reviewer") |>
    build
```

## Running the Workflow with Human Interaction

### Step 1: Initial Run

The workflow runs until the reviewer executor calls `request_info`:

``` julia
result = run_workflow(workflow, "Write about the Julia programming language")
println("State: ", get_final_state(result))
```

**Expected output:**

    State: WF_IDLE

### Step 2: Inspect Pending Requests

Extract the `EVT_REQUEST_INFO` events:

``` julia
pending = get_request_info_events(result)
println("Pending requests: ", length(pending))

for req in pending
    println("  Request ID: ", req.request_id)
    println("  Data:       ", req.data)
end
```

**Expected output:**

    Pending requests: 1
      Request ID: a1b2c3d4-...
      Data:       Dict("prompt" => "Please review this draft: Julia is a high-...", "type" => "review")

### Step 3: Resume with Human Response

Supply the human’s feedback and resume:

``` julia
req_id = pending[1].request_id

# Human approves the draft
result2 = run_workflow(workflow;
    responses = Dict{String,Any}(req_id => "approve"),
)
println("Final state: ", get_final_state(result2))
println("Output:      ", first(get_outputs(result2)))
```

**Expected output:**

    Final state: WF_IDLE
    Output:      Julia is a high-level, high-performance programming language...

If the human rejects instead:

``` julia
result3 = run_workflow(workflow;
    responses = Dict{String,Any}(req_id => "Needs more detail on performance"),
)
println("Output: ", first(get_outputs(result3)))
```

**Expected output:**

    Output: Revision needed: Needs more detail on performance

## Function Tool Approval Pattern

For sensitive tool calls, you can require human approval before
execution using `FunctionApprovalRequest` content. This is useful when
an agent wants to call a destructive or costly function.

``` julia
# Define a tool that requires human approval
sensitive_tool = FunctionTool(
    name = "delete_file",
    description = "Delete a file from disk",
    handler = (path) -> begin
        rm(path)
        "Deleted $path"
    end,
    requires_approval = true,
)

agent = Agent(
    name = "FileManager",
    instructions = "You manage files. Use delete_file to remove files when asked.",
    client = client,
    tools = [sensitive_tool],
)
```

When the agent attempts to call `delete_file`, the framework emits a
`FunctionApprovalRequest` content item instead of executing the tool
immediately:

``` julia
response = run_agent(agent, "Please delete old_report.txt")

# Check for approval requests in the response
for content in response.contents
    if content.type == CONTENT_FUNCTION_APPROVAL_REQUEST
        println("Tool:      ", content.function_call.name)
        println("Arguments: ", content.function_call.arguments)

        # Approve or deny
        approval = to_function_approval_response(content, true)  # approved=true
        resumed = run_agent(agent, approval)
        println("Result:    ", resumed.text)
    end
end
```

**Expected output:**

    Tool:      delete_file
    Arguments: {"path": "old_report.txt"}
    Result:    Deleted old_report.txt

## Workflow States During HITL

A workflow progresses through these states during a human-in-the-loop
interaction:

    WF_STARTED ──► WF_IN_PROGRESS ──► WF_IDLE (waiting for human)
                                           │
                                      responses provided
                                           │
                                      WF_IN_PROGRESS (resumed)
                                           │
                                      WF_IDLE (completed)

| State            | Meaning                                       |
|------------------|-----------------------------------------------|
| `WF_STARTED`     | Workflow has been initialized                 |
| `WF_IN_PROGRESS` | Executors are actively running                |
| `WF_IDLE`        | Execution paused — check for pending requests |

When `WF_IDLE` is reached, use `get_request_info_events` to determine
whether the workflow is truly finished (no pending requests) or waiting
for human input (one or more pending requests).

## Full Example: Multi-Step Approval Pipeline

This example chains a writer, a fact-checker, and a human approval gate:

``` julia
client = OllamaChatClient(model = "qwen3:8b")

# Step 1: Writer produces a draft
writer = agent_executor("writer",
    Agent(
        name = "Writer",
        instructions = "Write a one-paragraph article about the topic.",
        client = client,
    ),
    forward_response = true,
)

# Step 2: Fact-checker reviews the draft
checker = agent_executor("checker",
    Agent(
        name = "FactChecker",
        instructions = "Review the text for factual accuracy. Output PASS or list corrections needed.",
        client = client,
    ),
    forward_response = true,
)

# Step 3: Human approval gate
approver = ExecutorSpec(
    id = "approver",
    description = "Requests final human approval",
    handler = (msg, ctx) -> begin
        feedback = request_info(ctx;
            request_data = Dict(
                "prompt" => "Fact-check result: $(first(msg, 200))\n\nApprove for publication?",
                "type" => "final_approval",
            ),
            response_type = String,
        )
        if lowercase(strip(feedback)) == "approve"
            yield_output(ctx, "✅ Published: $msg")
        else
            yield_output(ctx, "❌ Rejected: $feedback")
        end
    end,
)

pipeline = WorkflowBuilder(name = "PublishPipeline", start = writer) |>
    b -> add_executor(b, checker) |>
    b -> add_executor(b, approver) |>
    b -> add_edge(b, "writer", "checker") |>
    b -> add_edge(b, "checker", "approver") |>
    b -> add_output(b, "approver") |>
    build

# Run until human input is needed
result = run_workflow(pipeline, "The history of the Julia programming language")
println("State: ", get_final_state(result))

# Inspect and respond
pending = get_request_info_events(result)
if !isempty(pending)
    println("Approval requested:")
    println("  ", pending[1].data["prompt"])

    # Human approves
    final = run_workflow(pipeline;
        responses = Dict{String,Any}(pending[1].request_id => "approve"),
    )
    println(first(get_outputs(final)))
end
```

**Expected output:**

    State: WF_IDLE
    Approval requested:
      Fact-check result: PASS — the article is factually accurate...
      Approve for publication?
    ✅ Published: PASS — the article is factually accurate...

## Summary

| Concept | Description |
|----|----|
| `request_info(ctx; ...)` | Pause an executor and emit a request event |
| `EVT_REQUEST_INFO` | Event type for pending human-input requests |
| `get_request_info_events(result)` | Extract all pending request events |
| `run_workflow(wf; responses=...)` | Resume a paused workflow with responses |
| `WF_IDLE` | Workflow state when paused or completed |
| `FunctionApprovalRequest` | Content type for tool-call approval gates |
| `to_function_approval_response` | Convert an approval request to a response |
| `requires_approval` | FunctionTool option to gate execution |

Key takeaways:

1.  **`request_info`** is the core primitive — it pauses the workflow
    and emits an event that external systems (or humans) can respond to.
2.  **Resume is explicit** — the caller must collect responses and pass
    them back via the `responses` keyword argument.
3.  **Function approval** provides a built-in pattern for gating
    sensitive tool calls behind human review.
4.  **Workflow state** distinguishes between “idle and done” vs “idle
    and waiting for input” via pending request events.

Next, see [23 — Skills](../23_skills/23_skills.qmd) to learn how to
package reusable agent capabilities as composable skill modules.
