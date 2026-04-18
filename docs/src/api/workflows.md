# Workflows

The workflow system provides graph-based multi-agent orchestration. Executor
nodes (agents or plain functions) are connected by typed edges. The engine
runs superstep iterations, routing [`WorkflowMessage`](@ref) objects between
executors until the workflow reaches a terminal state.

## Messages

```@docs
AgentFramework.WorkflowMessage
AgentFramework.WorkflowMessageType
```

## Run State

```@docs
AgentFramework.WorkflowRunState
```

## Events

```@docs
AgentFramework.WorkflowEvent
AgentFramework.WorkflowEventType
AgentFramework.WorkflowErrorDetails
AgentFramework.WorkflowRunResult
```

### Event Type Constants

```@docs

```

### Event Factory Functions

```@docs
AgentFramework.event_started
AgentFramework.event_status
AgentFramework.event_failed
AgentFramework.event_output
AgentFramework.event_executor_invoked
AgentFramework.event_executor_completed
AgentFramework.event_executor_failed
AgentFramework.event_superstep_started
AgentFramework.event_superstep_completed
AgentFramework.event_request_info
AgentFramework.event_warning
AgentFramework.event_error
```

### Event Query Functions

```@docs
AgentFramework.get_outputs
AgentFramework.get_request_info_events
AgentFramework.get_final_state
```

## Executors

```@docs
AgentFramework.ExecutorSpec
AgentFramework.WorkflowContext
AgentFramework.send_message
AgentFramework.yield_output
AgentFramework.get_state
AgentFramework.set_state!
AgentFramework.request_info
AgentFramework.execute_handler
AgentFramework.agent_executor
AgentFramework.@executor
AgentFramework.@handler
```

## Agent Executor

```@docs
AgentFramework.AgentExecutor
AgentFramework.AgentExecutorRequest
AgentFramework.AgentExecutorResponse
AgentFramework.to_executor_spec
AgentFramework.handle_request!
AgentFramework.handle_string!
AgentFramework.handle_message!
AgentFramework.handle_response!
AgentFramework.reset!
AgentFramework.get_conversation
```

## Edges

```@docs
AgentFramework.EdgeKind
AgentFramework.Edge
AgentFramework.EdgeGroup
AgentFramework.direct_edge
AgentFramework.fan_out_edge
AgentFramework.fan_in_edge
AgentFramework.switch_edge
AgentFramework.should_route
AgentFramework.route_messages
AgentFramework.source_executor_ids
AgentFramework.target_executor_ids
```

## Builder

```@docs
AgentFramework.Workflow
AgentFramework.run_workflow
AgentFramework.WorkflowBuilder
AgentFramework.add_executor
AgentFramework.add_edge
AgentFramework.add_fan_out
AgentFramework.add_fan_in
AgentFramework.add_switch
AgentFramework.add_output
AgentFramework.build(::AgentFramework.WorkflowBuilder)
```

### Convenience Builders

```@docs
AgentFramework.SequentialBuilder
AgentFramework.ConcurrentBuilder
AgentFramework.GroupChatBuilder
AgentFramework.MagenticBuilder
```

## Group Chat

```@docs
AgentFramework.RoundRobinGroupChat
AgentFramework.SelectorGroupChat
AgentFramework.MagenticOneGroupChat
AgentFramework.ConcurrentParticipantResult
AgentFramework.GroupChatState
```

### Group Chat Configuration

```@docs
AgentFramework.with_aggregator
AgentFramework.with_selection_func
AgentFramework.with_termination
AgentFramework.with_plan_review
```

## Magentic-One

```@docs
AgentFramework.AbstractMagenticManager
AgentFramework.StandardMagenticManager
AgentFramework.MagenticTaskLedger
AgentFramework.MagenticProgressLedgerItem
AgentFramework.MagenticProgressLedger
AgentFramework.MagenticContext
AgentFramework.MagenticPlanReviewRequest
AgentFramework.MagenticPlanReviewResponse
AgentFramework.magentic_plan
AgentFramework.magentic_select
AgentFramework.magentic_finalize
```

## Checkpoints

```@docs
AgentFramework.WorkflowCheckpoint
AgentFramework.AbstractCheckpointStorage
AgentFramework.InMemoryCheckpointStorage
AgentFramework.FileCheckpointStorage
```

## Validation

```@docs
AgentFramework.ValidationCheck
AgentFramework.ValidationIssue
AgentFramework.FullValidationResult
AgentFramework.validate_workflow
```

## Protocol Introspection

```@docs
AgentFramework.ProtocolDescriptor
AgentFramework.TypeCompatibilityResult
AgentFramework.WorkflowValidationResult
AgentFramework.get_protocol
AgentFramework.can_handle
AgentFramework.can_output
AgentFramework.check_type_compatibility
AgentFramework.validate_workflow_types
AgentFramework.describe_protocol
```
